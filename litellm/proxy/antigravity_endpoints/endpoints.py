from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()

_pending_oauth_states: Dict[str, Dict[str, str]] = {}


class AntigravityAccountResponse(BaseModel):
    email: str
    project_id: Optional[str] = None
    is_rate_limited: bool = False
    is_invalid: bool = False


class AntigravityAccountsListResponse(BaseModel):
    total: int
    available: int
    rate_limited: int
    invalid: int
    accounts: List[AntigravityAccountResponse]


class AntigravityAuthStartResponse(BaseModel):
    auth_url: str
    state: str
    message: str


@router.get(
    "/antigravity/auth/start",
    dependencies=[Depends(user_api_key_auth)],
    tags=["antigravity"],
    response_model=AntigravityAuthStartResponse,
)
async def antigravity_auth_start(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> AntigravityAuthStartResponse:
    from litellm.llms.antigravity.authenticator import get_authorization_url

    try:
        auth_data = get_authorization_url()

        _pending_oauth_states[auth_data["state"]] = {
            "verifier": auth_data["verifier"],
            "state": auth_data["state"],
        }

        callback_url = str(request.base_url).rstrip("/") + "/antigravity/auth/callback"

        verbose_proxy_logger.info(f"[Antigravity] OAuth started, callback: {callback_url}")

        return AntigravityAuthStartResponse(
            auth_url=auth_data["url"],
            state=auth_data["state"],
            message=f"Open the auth_url in a browser to authenticate. Callback will be sent to {callback_url}",
        )
    except Exception as e:
        verbose_proxy_logger.error(f"[Antigravity] Auth start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/antigravity/auth/callback",
    tags=["antigravity"],
    response_class=HTMLResponse,
)
async def antigravity_auth_callback(
    request: Request,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
) -> HTMLResponse:
    from litellm.llms.antigravity.authenticator import (
        exchange_code,
        get_user_email,
        discover_project_id,
    )
    from litellm.llms.antigravity.account_manager import get_account_manager

    if error:
        return HTMLResponse(
            content=f"""
            <html><head><title>Authentication Failed</title></head>
            <body style="font-family: system-ui; padding: 40px; text-align: center;">
            <h1 style="color: #dc3545;">Authentication Failed</h1>
            <p>Error: {error}</p>
            <p>You can close this window.</p>
            </body></html>
            """,
            status_code=400,
        )

    if not code or not state:
        return HTMLResponse(
            content="""
            <html><head><title>Authentication Failed</title></head>
            <body style="font-family: system-ui; padding: 40px; text-align: center;">
            <h1 style="color: #dc3545;">Authentication Failed</h1>
            <p>Missing authorization code or state parameter.</p>
            <p>You can close this window.</p>
            </body></html>
            """,
            status_code=400,
        )

    pending = _pending_oauth_states.pop(state, None)
    if not pending:
        return HTMLResponse(
            content="""
            <html><head><title>Authentication Failed</title></head>
            <body style="font-family: system-ui; padding: 40px; text-align: center;">
            <h1 style="color: #dc3545;">Authentication Failed</h1>
            <p>Invalid or expired state. Please try again.</p>
            <p>You can close this window.</p>
            </body></html>
            """,
            status_code=400,
        )

    try:
        tokens = exchange_code(code, pending["verifier"])
        email = get_user_email(tokens["access_token"])
        project_id = discover_project_id(tokens["access_token"])

        manager = get_account_manager()
        accounts = manager._authenticator.get_accounts()

        existing = next((a for a in accounts if a["email"] == email), None)
        if existing:
            existing["refresh_token"] = tokens["refresh_token"]
            existing["project_id"] = project_id
            existing.pop("is_invalid", None)
        else:
            accounts.append(
                {
                    "email": email,
                    "refresh_token": tokens["refresh_token"],
                    "project_id": project_id,
                }
            )

        manager._authenticator._accounts = accounts
        manager._authenticator._save_accounts()
        manager._authenticator._token_cache.set(email, tokens["access_token"], tokens.get("expires_in", 3600))

        verbose_proxy_logger.info(f"[Antigravity] Account added: {email}")

        return HTMLResponse(
            content=f"""
            <html><head><title>Authentication Successful</title></head>
            <body style="font-family: system-ui; padding: 40px; text-align: center;">
            <h1 style="color: #28a745;">Authentication Successful!</h1>
            <p>Account <strong>{email}</strong> has been added.</p>
            <p>Project ID: {project_id or "auto-discovered"}</p>
            <p>You can close this window and return to the LiteLLM UI.</p>
            <script>setTimeout(() => window.close(), 3000);</script>
            </body></html>
            """,
            status_code=200,
        )
    except Exception as e:
        verbose_proxy_logger.error(f"[Antigravity] OAuth callback error: {e}")
        return HTMLResponse(
            content=f"""
            <html><head><title>Authentication Failed</title></head>
            <body style="font-family: system-ui; padding: 40px; text-align: center;">
            <h1 style="color: #dc3545;">Authentication Failed</h1>
            <p>Error: {str(e)}</p>
            <p>You can close this window.</p>
            </body></html>
            """,
            status_code=500,
        )


@router.get(
    "/antigravity/accounts",
    dependencies=[Depends(user_api_key_auth)],
    tags=["antigravity"],
    response_model=AntigravityAccountsListResponse,
)
async def list_antigravity_accounts(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> AntigravityAccountsListResponse:
    from litellm.llms.antigravity.account_manager import get_account_manager

    try:
        manager = get_account_manager()
        status = manager.get_status()

        return AntigravityAccountsListResponse(
            total=status["total"],
            available=status["available"],
            rate_limited=status["rate_limited"],
            invalid=status["invalid"],
            accounts=[
                AntigravityAccountResponse(
                    email=a["email"],
                    project_id=a.get("project_id"),
                    is_rate_limited=a.get("is_rate_limited", False),
                    is_invalid=a.get("is_invalid", False),
                )
                for a in status["accounts"]
            ],
        )
    except Exception as e:
        verbose_proxy_logger.error(f"[Antigravity] List accounts error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/antigravity/accounts/{email}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["antigravity"],
)
async def delete_antigravity_account(
    email: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, Any]:
    from litellm.llms.antigravity.account_manager import get_account_manager

    try:
        manager = get_account_manager()
        removed = manager._authenticator.remove_account(email)

        if not removed:
            raise HTTPException(status_code=404, detail=f"Account {email} not found")

        verbose_proxy_logger.info(f"[Antigravity] Account removed: {email}")

        return {"success": True, "message": f"Account {email} removed"}
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.error(f"[Antigravity] Delete account error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/antigravity/accounts/reset-rate-limits",
    dependencies=[Depends(user_api_key_auth)],
    tags=["antigravity"],
)
async def reset_antigravity_rate_limits(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, Any]:
    from litellm.llms.antigravity.account_manager import get_account_manager

    try:
        manager = get_account_manager()
        manager.reset_all_rate_limits()

        verbose_proxy_logger.info("[Antigravity] Rate limits reset")

        return {"success": True, "message": "All rate limits cleared"}
    except Exception as e:
        verbose_proxy_logger.error(f"[Antigravity] Reset rate limits error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
