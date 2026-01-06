import hashlib
import json
import os
import secrets
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import _get_httpx_client

from .common_utils import (
    ANTIGRAVITY_ACCOUNTS_FILE,
    ANTIGRAVITY_CONFIG_DIR,
    ANTIGRAVITY_ENDPOINT_FALLBACKS,
    AntigravityAuthError,
    AntigravityInvalidCredentialsError,
    DEFAULT_PROJECT_ID,
    OAUTH_CONFIG,
    OAUTH_REDIRECT_URI,
    TOKEN_REFRESH_INTERVAL_MS,
    get_antigravity_headers,
)


class TokenCache:
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get(self, email: str) -> Optional[str]:
        entry = self._cache.get(email)
        if entry and entry.get("expires_at", 0) > time.time():
            return entry.get("access_token")
        return None

    def set(self, email: str, access_token: str, expires_in: int = 3600):
        self._cache[email] = {
            "access_token": access_token,
            "expires_at": time.time() + expires_in - 60,  # 60s buffer
        }

    def clear(self, email: Optional[str] = None):
        if email:
            self._cache.pop(email, None)
        else:
            self._cache.clear()


class ProjectCache:
    def __init__(self):
        self._cache: Dict[str, str] = {}

    def get(self, email: str) -> Optional[str]:
        return self._cache.get(email)

    def set(self, email: str, project_id: str):
        self._cache[email] = project_id

    def clear(self, email: Optional[str] = None):
        if email:
            self._cache.pop(email, None)
        else:
            self._cache.clear()


def generate_pkce() -> Tuple[str, str]:
    verifier = secrets.token_urlsafe(32)
    challenge = hashlib.sha256(verifier.encode()).digest()
    challenge_b64 = challenge.hex().encode().decode().replace("+", "-").replace("/", "_").rstrip("=")
    import base64

    challenge_b64 = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    return verifier, challenge_b64


def get_authorization_url() -> Dict[str, str]:
    verifier, challenge = generate_pkce()
    state = secrets.token_hex(16)

    params = {
        "client_id": OAUTH_CONFIG["client_id"],
        "redirect_uri": OAUTH_REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(OAUTH_CONFIG["scopes"]),
        "access_type": "offline",
        "prompt": "consent",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
    }

    url = f"{OAUTH_CONFIG['auth_url']}?{urlencode(params)}"
    return {"url": url, "verifier": verifier, "state": state}


def exchange_code(code: str, verifier: str) -> Dict[str, Any]:
    client = _get_httpx_client()
    response = client.post(
        OAUTH_CONFIG["token_url"],
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "client_id": OAUTH_CONFIG["client_id"],
            "client_secret": OAUTH_CONFIG["client_secret"],
            "code": code,
            "code_verifier": verifier,
            "grant_type": "authorization_code",
            "redirect_uri": OAUTH_REDIRECT_URI,
        },
    )

    if not response.is_success:
        raise AntigravityAuthError(
            status_code=response.status_code,
            message=f"Token exchange failed: {response.text}",
        )

    tokens = response.json()
    if "access_token" not in tokens:
        raise AntigravityAuthError(
            status_code=400,
            message="No access token in response",
        )

    return {
        "access_token": tokens["access_token"],
        "refresh_token": tokens.get("refresh_token"),
        "expires_in": tokens.get("expires_in", 3600),
    }


def refresh_access_token(refresh_token: str) -> Dict[str, Any]:
    client = _get_httpx_client()
    response = client.post(
        OAUTH_CONFIG["token_url"],
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "client_id": OAUTH_CONFIG["client_id"],
            "client_secret": OAUTH_CONFIG["client_secret"],
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
    )

    if not response.is_success:
        raise AntigravityInvalidCredentialsError(
            status_code=response.status_code,
            message=f"Token refresh failed: {response.text}",
        )

    tokens = response.json()
    return {
        "access_token": tokens["access_token"],
        "expires_in": tokens.get("expires_in", 3600),
    }


def get_user_email(access_token: str) -> str:
    client = _get_httpx_client()
    response = client.get(
        OAUTH_CONFIG["user_info_url"],
        headers={"Authorization": f"Bearer {access_token}"},
    )

    if not response.is_success:
        raise AntigravityAuthError(
            status_code=response.status_code,
            message=f"Failed to get user info: {response.text}",
        )

    return response.json().get("email", "unknown")


def discover_project_id(access_token: str) -> Optional[str]:
    client = _get_httpx_client()
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        **get_antigravity_headers(),
    }

    for endpoint in ANTIGRAVITY_ENDPOINT_FALLBACKS:
        try:
            response = client.post(
                f"{endpoint}/v1internal:loadCodeAssist",
                headers=headers,
                json={
                    "metadata": {
                        "ideType": "IDE_UNSPECIFIED",
                        "platform": "PLATFORM_UNSPECIFIED",
                        "pluginType": "GEMINI",
                    }
                },
            )

            if not response.is_success:
                continue

            data = response.json()
            if isinstance(data.get("cloudaicompanionProject"), str):
                return data["cloudaicompanionProject"]
            if data.get("cloudaicompanionProject", {}).get("id"):
                return data["cloudaicompanionProject"]["id"]

        except Exception as e:
            verbose_logger.warning(f"Project discovery failed at {endpoint}: {e}")

    return DEFAULT_PROJECT_ID


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    authorization_code: Optional[str] = None
    received_state: Optional[str] = None
    error: Optional[str] = None

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/oauth-callback":
            self.send_response(404)
            self.end_headers()
            return

        params = parse_qs(parsed.query)

        if "error" in params:
            OAuthCallbackHandler.error = params["error"][0]
            self._send_error_page(params["error"][0])
            return

        if "code" not in params:
            OAuthCallbackHandler.error = "No authorization code"
            self._send_error_page("No authorization code received")
            return

        OAuthCallbackHandler.authorization_code = params["code"][0]
        OAuthCallbackHandler.received_state = params.get("state", [None])[0]
        self._send_success_page()

    def _send_success_page(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        html = """
        <html><head><title>Authentication Successful</title></head>
        <body style="font-family: system-ui; padding: 40px; text-align: center;">
        <h1 style="color: #28a745;">Authentication Successful!</h1>
        <p>You can close this window and return to the terminal.</p>
        </body></html>
        """
        self.wfile.write(html.encode())

    def _send_error_page(self, error: str):
        self.send_response(400)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        html = f"""
        <html><head><title>Authentication Failed</title></head>
        <body style="font-family: system-ui; padding: 40px; text-align: center;">
        <h1 style="color: #dc3545;">Authentication Failed</h1>
        <p>Error: {error}</p>
        </body></html>
        """
        self.wfile.write(html.encode())

    def log_message(self, format, *args):
        pass  # Suppress HTTP server logs


def start_oauth_callback_server(expected_state: str, timeout_seconds: int = 120) -> Optional[str]:
    OAuthCallbackHandler.authorization_code = None
    OAuthCallbackHandler.received_state = None
    OAuthCallbackHandler.error = None

    server = HTTPServer(("localhost", OAUTH_CONFIG["callback_port"]), OAuthCallbackHandler)
    server.timeout = timeout_seconds

    def serve():
        while OAuthCallbackHandler.authorization_code is None and OAuthCallbackHandler.error is None:
            server.handle_request()

    thread = Thread(target=serve, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    server.server_close()

    if OAuthCallbackHandler.error:
        raise AntigravityAuthError(
            status_code=400,
            message=f"OAuth error: {OAuthCallbackHandler.error}",
        )

    if OAuthCallbackHandler.received_state != expected_state:
        raise AntigravityAuthError(
            status_code=400,
            message="State mismatch - possible CSRF attack",
        )

    return OAuthCallbackHandler.authorization_code


class Authenticator:
    def __init__(self):
        self._token_cache = TokenCache()
        self._project_cache = ProjectCache()
        self._accounts: list = []
        self._ensure_config_dir()
        self._load_accounts()

    def _ensure_config_dir(self):
        os.makedirs(ANTIGRAVITY_CONFIG_DIR, exist_ok=True)

    def _load_accounts(self):
        if os.path.exists(ANTIGRAVITY_ACCOUNTS_FILE):
            try:
                with open(ANTIGRAVITY_ACCOUNTS_FILE, "r") as f:
                    data = json.load(f)
                    self._accounts = data.get("accounts", [])
            except (json.JSONDecodeError, IOError) as e:
                verbose_logger.warning(f"Failed to load accounts: {e}")
                self._accounts = []

    def _save_accounts(self):
        with open(ANTIGRAVITY_ACCOUNTS_FILE, "w") as f:
            json.dump({"accounts": self._accounts}, f, indent=2)

    def get_accounts(self) -> list:
        return self._accounts.copy()

    def add_account_interactive(self) -> Dict[str, Any]:
        auth_data = get_authorization_url()
        print(f"\nPlease visit this URL to authenticate:\n{auth_data['url']}\n")
        print("Waiting for authentication...")

        code = start_oauth_callback_server(auth_data["state"])
        if not code:
            raise AntigravityAuthError(
                status_code=400,
                message="OAuth callback timeout",
            )

        tokens = exchange_code(code, auth_data["verifier"])
        email = get_user_email(tokens["access_token"])
        project_id = discover_project_id(tokens["access_token"])

        existing = next((a for a in self._accounts if a["email"] == email), None)
        if existing:
            existing["refresh_token"] = tokens["refresh_token"]
            existing["project_id"] = project_id
        else:
            self._accounts.append(
                {
                    "email": email,
                    "refresh_token": tokens["refresh_token"],
                    "project_id": project_id,
                }
            )

        self._save_accounts()
        self._token_cache.set(email, tokens["access_token"], tokens["expires_in"])

        return {"email": email, "project_id": project_id or ""}

    def get_token_for_account(self, account: Dict[str, Any]) -> str:
        email = account["email"]

        cached = self._token_cache.get(email)
        if cached:
            return cached

        refresh_token = account.get("refresh_token")
        if not refresh_token:
            raise AntigravityInvalidCredentialsError(
                status_code=401,
                message=f"No refresh token for account {email}",
            )

        try:
            tokens = refresh_access_token(refresh_token)
            self._token_cache.set(email, tokens["access_token"], tokens["expires_in"])
            return tokens["access_token"]
        except AntigravityInvalidCredentialsError:
            raise
        except Exception as e:
            raise AntigravityAuthError(
                status_code=401,
                message=f"Failed to refresh token for {email}: {e}",
            )

    def get_project_for_account(self, account: Dict[str, Any], token: str) -> str:
        email = account["email"]

        cached = self._project_cache.get(email)
        if cached:
            return cached

        stored = account.get("project_id")
        if stored:
            self._project_cache.set(email, stored)
            return stored

        discovered = discover_project_id(token)
        if discovered:
            self._project_cache.set(email, discovered)
            account["project_id"] = discovered
            self._save_accounts()
            return discovered

        return DEFAULT_PROJECT_ID

    def clear_token_cache(self, email: Optional[str] = None):
        self._token_cache.clear(email)

    def clear_project_cache(self, email: Optional[str] = None):
        self._project_cache.clear(email)

    def remove_account(self, email: str) -> bool:
        before = len(self._accounts)
        self._accounts = [a for a in self._accounts if a["email"] != email]
        if len(self._accounts) < before:
            self._save_accounts()
            self._token_cache.clear(email)
            self._project_cache.clear(email)
            return True
        return False
