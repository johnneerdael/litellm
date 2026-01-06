import os
import secrets
import time
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

import httpx

from litellm._logging import verbose_logger
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse, Usage

from ..account_manager import AccountManager, get_account_manager
from ..common_utils import (
    ANTIGRAVITY_API_BASE,
    ANTIGRAVITY_ENDPOINT_FALLBACKS,
    ANTIGRAVITY_MODELS,
    AntigravityAuthError,
    AntigravityError,
    AntigravityNoAccountsError,
    AntigravityQuotaExhaustedError,
    AntigravityRateLimitError,
    DEFAULT_COOLDOWN_MS,
    GEMINI_MAX_OUTPUT_TOKENS,
    MAX_RETRIES,
    MAX_WAIT_BEFORE_ERROR_MS,
    get_antigravity_headers,
    get_fallback_model,
    get_model_family,
    is_auth_error,
    is_rate_limit_error,
    is_thinking_model,
    parse_reset_time_from_error,
)


def _convert_role(role: str) -> str:
    if role == "assistant":
        return "model"
    return "user"


def _convert_content_to_parts(
    content: Union[str, List[Dict[str, Any]]], is_claude: bool, is_gemini: bool
) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]

    if not isinstance(content, list):
        return [{"text": str(content)}]

    parts = []
    for block in content:
        if not block:
            continue

        block_type = block.get("type")

        if block_type == "text":
            text = block.get("text", "")
            if text and text.strip():
                parts.append({"text": text})

        elif block_type == "image":
            source = block.get("source", {})
            if source.get("type") == "base64":
                parts.append(
                    {
                        "inlineData": {
                            "mimeType": source.get("media_type", "image/jpeg"),
                            "data": source.get("data"),
                        }
                    }
                )
            elif source.get("type") == "url":
                parts.append(
                    {
                        "fileData": {
                            "mimeType": source.get("media_type", "image/jpeg"),
                            "fileUri": source.get("url"),
                        }
                    }
                )

        elif block_type == "image_url":
            url = block.get("image_url", {}).get("url", "")
            if url.startswith("data:"):
                import base64

                header, data = url.split(",", 1)
                mime_type = header.split(";")[0].replace("data:", "")
                parts.append(
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": data,
                        }
                    }
                )
            else:
                parts.append(
                    {
                        "fileData": {
                            "mimeType": "image/jpeg",
                            "fileUri": url,
                        }
                    }
                )

        elif block_type == "tool_use":
            function_call = {
                "name": block.get("name"),
                "args": block.get("input", {}),
            }
            if is_claude and block.get("id"):
                function_call["id"] = block["id"]

            part: Dict[str, Any] = {"functionCall": function_call}
            if is_gemini and block.get("thoughtSignature"):
                part["thoughtSignature"] = block["thoughtSignature"]

            parts.append(part)

        elif block_type == "tool_result":
            response_content = block.get("content", "")
            if isinstance(response_content, str):
                response_content = {"result": response_content}
            elif isinstance(response_content, list):
                texts = [c.get("text", "") for c in response_content if c.get("type") == "text"]
                response_content = {"result": "\n".join(texts) or ""}

            function_response = {
                "name": block.get("tool_use_id", "unknown"),
                "response": response_content,
            }
            if is_claude and block.get("tool_use_id"):
                function_response["id"] = block["tool_use_id"]

            parts.append({"functionResponse": function_response})

        elif block_type == "thinking":
            signature = block.get("signature", "")
            if signature and len(signature) >= 50:
                parts.append(
                    {
                        "text": block.get("thinking", ""),
                        "thought": True,
                        "thoughtSignature": signature,
                    }
                )

    return parts


def _convert_openai_messages_to_google(
    messages: List[Dict[str, Any]], model: str
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    model_family = get_model_family(model)
    is_claude = model_family == "claude"
    is_gemini = model_family == "gemini"

    contents = []
    system_instruction = None

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            if isinstance(content, str):
                system_instruction = {"parts": [{"text": content}]}
            elif isinstance(content, list):
                text_parts = [{"text": c.get("text", "")} for c in content if c.get("type") == "text"]
                if text_parts:
                    system_instruction = {"parts": text_parts}
            continue

        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "unknown")
            tool_content = content if isinstance(content, str) else str(content)
            parts = [
                {
                    "functionResponse": {
                        "name": tool_call_id,
                        "response": {"result": tool_content},
                    }
                }
            ]
            contents.append({"role": "user", "parts": parts})
            continue

        parts = _convert_content_to_parts(content, is_claude, is_gemini)

        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            func = tc.get("function", {})
            import json

            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}

            function_call = {
                "name": func.get("name", ""),
                "args": args,
            }
            if is_claude:
                function_call["id"] = tc.get("id", "")

            parts.append({"functionCall": function_call})

        if not parts:
            parts = [{"text": "."}]

        google_role = _convert_role(role)
        contents.append({"role": google_role, "parts": parts})

    return contents, system_instruction


def _convert_google_response_to_openai(google_response: Dict[str, Any], model: str) -> Dict[str, Any]:
    response = google_response.get("response", google_response)
    candidates = response.get("candidates", [])
    first_candidate = candidates[0] if candidates else {}
    content_obj = first_candidate.get("content", {})
    parts = content_obj.get("parts", [])

    content_blocks = []
    tool_calls = []
    has_tool_calls = False

    for part in parts:
        if "text" in part:
            if part.get("thought"):
                content_blocks.append(
                    {
                        "type": "thinking",
                        "thinking": part["text"],
                        "signature": part.get("thoughtSignature", ""),
                    }
                )
            else:
                content_blocks.append(
                    {
                        "type": "text",
                        "text": part["text"],
                    }
                )

        elif "functionCall" in part:
            fc = part["functionCall"]
            import json

            tool_calls.append(
                {
                    "id": fc.get("id", f"call_{secrets.token_hex(12)}"),
                    "type": "function",
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": json.dumps(fc.get("args", {})),
                    },
                }
            )
            has_tool_calls = True

    finish_reason = first_candidate.get("finishReason", "STOP")
    if finish_reason == "STOP":
        stop_reason = "stop"
    elif finish_reason == "MAX_TOKENS":
        stop_reason = "length"
    elif finish_reason == "TOOL_USE" or has_tool_calls:
        stop_reason = "tool_calls"
    else:
        stop_reason = "stop"

    usage_metadata = response.get("usageMetadata", {})
    prompt_tokens = usage_metadata.get("promptTokenCount", 0)
    cached_tokens = usage_metadata.get("cachedContentTokenCount", 0)
    completion_tokens = usage_metadata.get("candidatesTokenCount", 0)

    text_content = ""
    for block in content_blocks:
        if block.get("type") == "text":
            text_content += block.get("text", "")

    message: Dict[str, Any] = {
        "role": "assistant",
        "content": text_content or None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": f"chatcmpl-{secrets.token_hex(16)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": stop_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens - cached_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens - cached_tokens + completion_tokens,
        },
    }


def _sanitize_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object"}

    result = {}
    for key, value in schema.items():
        if key in ("$schema", "$id", "$ref", "definitions", "$defs", "examples", "default"):
            continue

        if key == "properties" and isinstance(value, dict):
            result[key] = {k: _sanitize_schema(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            result[key] = _sanitize_schema(value)
        elif key == "additionalProperties":
            if isinstance(value, dict):
                result[key] = _sanitize_schema(value)
            elif value is False:
                result[key] = False
        elif isinstance(value, dict):
            result[key] = _sanitize_schema(value)
        else:
            result[key] = value

    if "type" not in result:
        result["type"] = "object"

    return result


class AntigravityConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self._account_manager: Optional[AccountManager] = None

    @property
    def account_manager(self) -> AccountManager:
        if self._account_manager is None:
            self._account_manager = get_account_manager()
        return self._account_manager

    def get_supported_openai_params(self, model: str) -> List[str]:
        return [
            "temperature",
            "max_tokens",
            "top_p",
            "stop",
            "stream",
            "tools",
            "tool_choice",
        ]

    def map_openai_params(
        self,
        non_default_params: Dict[str, Any],
        optional_params: Dict[str, Any],
        model: str,
        drop_params: bool,
    ) -> Dict[str, Any]:
        for key in ["temperature", "max_tokens", "top_p", "stop", "tools", "tool_choice"]:
            if key in non_default_params:
                optional_params[key] = non_default_params[key]
        return optional_params

    def _get_openai_compatible_provider_info(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        resolved_api_base = api_base or get_secret_str("ANTIGRAVITY_API_BASE") or ANTIGRAVITY_API_BASE
        resolved_api_key = api_key or get_secret_str("ANTIGRAVITY_API_KEY") or "antigravity"
        return resolved_api_base, resolved_api_key

    def validate_environment(
        self,
        headers: Dict[str, str],
        model: str,
        messages: List[AllMessageValues],
        optional_params: Dict[str, Any],
        litellm_params: Dict[str, Any],
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.account_manager.get_account_count() == 0:
            raise AntigravityNoAccountsError(
                status_code=401,
                message="No Antigravity accounts configured. Run 'litellm antigravity add-account' to authenticate.",
            )
        return {"headers": headers, "api_base": api_base}

    def _build_request_payload(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        optional_params: Dict[str, Any],
        project_id: str,
    ) -> Dict[str, Any]:
        contents, system_instruction = _convert_openai_messages_to_google(messages, model)
        model_family = get_model_family(model)
        is_thinking = is_thinking_model(model)

        generation_config: Dict[str, Any] = {}

        if "max_tokens" in optional_params:
            generation_config["maxOutputTokens"] = optional_params["max_tokens"]
        if "temperature" in optional_params:
            generation_config["temperature"] = optional_params["temperature"]
        if "top_p" in optional_params:
            generation_config["topP"] = optional_params["top_p"]
        if "stop" in optional_params:
            stop = optional_params["stop"]
            if isinstance(stop, str):
                generation_config["stopSequences"] = [stop]
            elif isinstance(stop, list):
                generation_config["stopSequences"] = stop

        if is_thinking:
            if model_family == "claude":
                thinking_config = {"include_thoughts": True}
                budget = optional_params.get("thinking", {}).get("budget_tokens")
                if budget:
                    thinking_config["thinking_budget"] = budget
                    max_tokens = generation_config.get("maxOutputTokens", 0)
                    if max_tokens and max_tokens <= budget:
                        generation_config["maxOutputTokens"] = budget + 8192
                generation_config["thinkingConfig"] = thinking_config
            else:
                thinking_config = {
                    "includeThoughts": True,
                    "thinkingBudget": optional_params.get("thinking", {}).get("budget_tokens", 16000),
                }
                generation_config["thinkingConfig"] = thinking_config

        if model_family == "gemini":
            max_tokens = generation_config.get("maxOutputTokens", 0)
            if max_tokens > GEMINI_MAX_OUTPUT_TOKENS:
                generation_config["maxOutputTokens"] = GEMINI_MAX_OUTPUT_TOKENS

        google_request: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }

        if system_instruction:
            google_request["systemInstruction"] = system_instruction

        tools = optional_params.get("tools", [])
        if tools:
            function_declarations = []
            for tool in tools:
                func = tool.get("function", tool)
                name = func.get("name", "")
                description = func.get("description", "")
                parameters = func.get("parameters", func.get("input_schema", {"type": "object"}))
                parameters = _sanitize_schema(parameters)

                function_declarations.append(
                    {
                        "name": name,
                        "description": description,
                        "parameters": parameters,
                    }
                )
            google_request["tools"] = [{"functionDeclarations": function_declarations}]

        import hashlib

        first_user = next((m for m in messages if m.get("role") == "user"), None)
        if first_user:
            content = first_user.get("content", "")
            if isinstance(content, list):
                content = str(content)
            session_id = hashlib.sha256(content[:500].encode()).hexdigest()[:16]
        else:
            session_id = secrets.token_hex(8)

        google_request["sessionId"] = session_id

        return {
            "project": project_id,
            "model": model,
            "request": google_request,
            "userAgent": "antigravity-litellm",
            "requestId": f"agent-{secrets.token_hex(16)}",
        }

    def _build_headers(self, token: str, model: str) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            **get_antigravity_headers(),
        }

        model_family = get_model_family(model)
        if model_family == "claude" and is_thinking_model(model):
            headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

        return headers

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        return self._build_request_payload(
            model=model,
            messages=messages,  # type: ignore
            optional_params=optional_params,
            project_id=litellm_params.get("project_id", ""),
        )

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: Any,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        google_response = raw_response.json()
        openai_response = _convert_google_response_to_openai(google_response, model)

        model_response.id = openai_response["id"]
        model_response.created = openai_response["created"]
        model_response.model = openai_response["model"]
        model_response.choices = openai_response["choices"]
        setattr(model_response, "_hidden_params", {"usage": openai_response["usage"]})

        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> AntigravityError:
        if status_code == 429:
            return AntigravityRateLimitError(
                status_code=status_code,
                message=error_message,
                headers=headers,
            )
        if status_code == 401:
            return AntigravityAuthError(
                status_code=status_code,
                message=error_message,
                headers=headers,
            )
        return AntigravityError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    def completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: str,
        custom_prompt_dict: Dict[str, Any],
        model_response: ModelResponse,
        print_verbose: Any,
        encoding: Any,
        api_key: Optional[str],
        logging_obj: Any,
        optional_params: Dict[str, Any],
        acompletion: bool = False,
        litellm_params: Optional[Dict[str, Any]] = None,
        logger_fn: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ) -> Union[ModelResponse, Iterator[ModelResponse]]:
        max_attempts = max(MAX_RETRIES, self.account_manager.get_account_count() + 1)

        for attempt in range(max_attempts):
            account, wait_ms = self.account_manager.pick_sticky_account(model)

            if not account and wait_ms > 0 and wait_ms <= MAX_WAIT_BEFORE_ERROR_MS:
                verbose_logger.info(f"[Antigravity] Waiting {wait_ms}ms for sticky account...")
                time.sleep(wait_ms / 1000)
                self.account_manager.clear_expired_limits()
                account = self.account_manager.get_current_sticky_account(model)

            if not account:
                if self.account_manager.is_all_rate_limited(model):
                    all_wait_ms = self.account_manager.get_min_wait_time_ms(model)
                    if all_wait_ms > MAX_WAIT_BEFORE_ERROR_MS:
                        raise AntigravityQuotaExhaustedError(
                            status_code=429,
                            message=f"All accounts rate-limited for {model}. Wait {all_wait_ms}ms.",
                        )
                    verbose_logger.warning(f"[Antigravity] All accounts rate-limited, waiting {all_wait_ms}ms...")
                    time.sleep(all_wait_ms / 1000)
                    self.account_manager.clear_expired_limits()
                    account = self.account_manager.pick_next(model)

                if not account:
                    fallback_model = get_fallback_model(model)
                    if fallback_model:
                        verbose_logger.warning(f"[Antigravity] Trying fallback model: {fallback_model}")
                        return self.completion(
                            model=fallback_model,
                            messages=messages,
                            api_base=api_base,
                            custom_prompt_dict=custom_prompt_dict,
                            model_response=model_response,
                            print_verbose=print_verbose,
                            encoding=encoding,
                            api_key=api_key,
                            logging_obj=logging_obj,
                            optional_params=optional_params,
                            acompletion=acompletion,
                            litellm_params=litellm_params,
                            logger_fn=logger_fn,
                            headers=headers,
                            timeout=timeout,
                            client=client,
                        )
                    raise AntigravityNoAccountsError(
                        status_code=503,
                        message="No accounts available",
                    )

            try:
                token = self.account_manager.get_token_for_account(account)
                project = self.account_manager.get_project_for_account(account, token)
                payload = self._build_request_payload(model, messages, optional_params, project)
                request_headers = self._build_headers(token, model)

                sync_client = _get_httpx_client()
                last_error: Any = None

                for endpoint in ANTIGRAVITY_ENDPOINT_FALLBACKS:
                    url = f"{endpoint}/v1internal:generateContent"

                    try:
                        response = sync_client.post(
                            url,
                            headers=request_headers,
                            json=payload,
                            timeout=timeout or 600.0,
                        )

                        if response.status_code == 401:
                            verbose_logger.warning("[Antigravity] Auth error, refreshing token...")
                            self.account_manager.clear_token_cache(account["email"])
                            self.account_manager.clear_project_cache(account["email"])
                            continue

                        if response.status_code == 429:
                            error_text = response.text
                            reset_ms = parse_reset_time_from_error(error_text)
                            verbose_logger.info(f"[Antigravity] Rate limited at {endpoint}")
                            last_error = {"is_429": True, "reset_ms": reset_ms, "text": error_text}
                            continue

                        if response.status_code >= 500:
                            verbose_logger.warning(f"[Antigravity] Server error {response.status_code} at {endpoint}")
                            last_error = Exception(f"Server error {response.status_code}: {response.text}")
                            time.sleep(1)
                            continue

                        response.raise_for_status()
                        google_response = response.json()
                        openai_response = _convert_google_response_to_openai(google_response, model)

                        model_response.id = openai_response["id"]
                        model_response.created = openai_response["created"]
                        model_response.model = openai_response["model"]
                        model_response.choices = openai_response["choices"]
                        setattr(model_response, "_hidden_params", {"usage": openai_response["usage"]})

                        return model_response

                    except httpx.HTTPStatusError as e:
                        last_error = e
                        verbose_logger.warning(f"[Antigravity] HTTP error at {endpoint}: {e}")
                    except Exception as e:
                        last_error = e
                        verbose_logger.warning(f"[Antigravity] Error at {endpoint}: {e}")

                if last_error:
                    if isinstance(last_error, dict) and last_error.get("is_429"):
                        self.account_manager.mark_rate_limited(account["email"], last_error.get("reset_ms"), model)
                        raise AntigravityRateLimitError(
                            status_code=429,
                            message=f"Rate limited: {last_error.get('text', '')}",
                        )
                    raise AntigravityError(
                        status_code=500,
                        message=str(last_error),
                    )

            except AntigravityRateLimitError:
                verbose_logger.info(f"[Antigravity] Account {account['email']} rate-limited, trying next...")
                continue
            except AntigravityAuthError:
                self.account_manager.mark_invalid(account["email"], "Auth error")
                verbose_logger.warning(f"[Antigravity] Account {account['email']} invalid, trying next...")
                continue
            except Exception as e:
                if is_rate_limit_error(e):
                    self.account_manager.mark_rate_limited(account["email"], None, model)
                    continue
                if is_auth_error(e):
                    self.account_manager.mark_invalid(account["email"], str(e))
                    continue

                error_str = str(e).lower()
                if "500" in error_str or "503" in error_str or "server error" in error_str:
                    verbose_logger.warning(f"[Antigravity] Server error, trying next account...")
                    self.account_manager.pick_next(model)
                    continue

                raise

        raise AntigravityError(
            status_code=500,
            message="Max retries exceeded",
        )
