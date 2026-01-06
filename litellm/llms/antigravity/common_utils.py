"""
Constants and utilities for Antigravity Cloud Code API integration.

Antigravity provides access to Claude and Gemini models via Google's Cloud Code service.
This module contains API endpoints, OAuth configuration, headers, and error classes.
"""

import os
import platform
from typing import Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException


# Version info
ANTIGRAVITY_VERSION = "1.11.5"

# Cloud Code API endpoints (in fallback order: daily -> prod)
ANTIGRAVITY_ENDPOINT_DAILY = "https://daily-cloudcode-pa.sandbox.googleapis.com"
ANTIGRAVITY_ENDPOINT_PROD = "https://cloudcode-pa.googleapis.com"
ANTIGRAVITY_ENDPOINT_FALLBACKS = [
    ANTIGRAVITY_ENDPOINT_DAILY,
    ANTIGRAVITY_ENDPOINT_PROD,
]

# Default API base (primary endpoint)
ANTIGRAVITY_API_BASE = ANTIGRAVITY_ENDPOINT_DAILY

# Default project ID if none can be discovered
DEFAULT_PROJECT_ID = "rising-fact-p41fc"

# Token refresh interval (5 minutes)
TOKEN_REFRESH_INTERVAL_MS = 5 * 60 * 1000

# Rate limit thresholds
DEFAULT_COOLDOWN_MS = 60 * 1000  # 1 minute default cooldown
MAX_RETRIES = 5  # Max retry attempts across accounts
MAX_ACCOUNTS = 10  # Maximum number of accounts allowed
MAX_WAIT_BEFORE_ERROR_MS = 120000  # 2 minutes - throw error if wait exceeds this

# Thinking model constants
MIN_SIGNATURE_LENGTH = 50  # Minimum valid thinking signature length

# Gemini-specific limits
GEMINI_MAX_OUTPUT_TOKENS = 16384

# Config file location (configurable via env vars, like GitHub Copilot)
ANTIGRAVITY_CONFIG_DIR = os.getenv("ANTIGRAVITY_CONFIG_DIR", os.path.expanduser("~/.config/litellm/antigravity"))
ANTIGRAVITY_ACCOUNTS_FILE = os.path.join(
    ANTIGRAVITY_CONFIG_DIR, os.getenv("ANTIGRAVITY_ACCOUNTS_FILE", "accounts.json")
)


def get_platform_user_agent() -> str:
    """Generate platform-specific User-Agent string."""
    os_name = platform.system().lower()
    arch = platform.machine()
    return f"antigravity/{ANTIGRAVITY_VERSION} {os_name}/{arch}"


# Required headers for Antigravity API requests
def get_antigravity_headers() -> dict:
    """Get default headers for Antigravity API requests."""
    return {
        "User-Agent": get_platform_user_agent(),
        "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
        "Client-Metadata": '{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}',
    }


# Google OAuth configuration (same credentials as Antigravity app)
OAUTH_CONFIG = {
    "client_id": "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com",
    "client_secret": "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",
    "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
    "token_url": "https://oauth2.googleapis.com/token",
    "user_info_url": "https://www.googleapis.com/oauth2/v1/userinfo",
    "callback_port": 51121,
    "scopes": [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/cclog",
        "https://www.googleapis.com/auth/experimentsandconfigs",
    ],
}

OAUTH_REDIRECT_URI = f"http://localhost:{OAUTH_CONFIG['callback_port']}/oauth-callback"


# Model family detection
def get_model_family(model_name: str) -> str:
    """
    Get the model family from model name (dynamic detection).

    Args:
        model_name: The model name from the request

    Returns:
        'claude', 'gemini', or 'unknown'
    """
    lower = (model_name or "").lower()
    if "claude" in lower:
        return "claude"
    if "gemini" in lower:
        return "gemini"
    return "unknown"


def is_thinking_model(model_name: str) -> bool:
    """
    Check if a model supports thinking/reasoning output.

    Args:
        model_name: The model name from the request

    Returns:
        True if the model supports thinking blocks
    """
    lower = (model_name or "").lower()
    # Claude thinking models have "thinking" in the name
    if "claude" in lower and "thinking" in lower:
        return True
    # Gemini thinking models: explicit "thinking" in name, OR gemini version 3+
    if "gemini" in lower:
        if "thinking" in lower:
            return True
        # Check for gemini-3 or higher (e.g., gemini-3, gemini-3.5, gemini-4, etc.)
        import re

        version_match = re.search(r"gemini[.-]?(\d+)", lower)
        if version_match and int(version_match.group(1)) >= 3:
            return True
    return False


# Supported models (all functional via Antigravity)
ANTIGRAVITY_MODELS = [
    # Claude models
    "claude-sonnet-4.5-thinking",
    "claude-opus-4.5-thinking",
    "claude-sonnet-4.5",
    # Gemini 3 models
    "gemini-3-flash",
    "gemini-3-pro-high",
    "gemini-3-pro-low",
    # Gemini 2.5 models
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

# Model fallback mapping - maps primary model to fallback when quota exhausted
MODEL_FALLBACK_MAP = {
    "gemini-3-pro-high": "claude-opus-4.5-thinking",
    "gemini-3-pro-low": "claude-sonnet-4.5",
    "gemini-3-flash": "claude-sonnet-4.5-thinking",
    "gemini-2.5-flash": "claude-sonnet-4.5",
    "gemini-2.5-pro": "claude-opus-4.5-thinking",
    "claude-opus-4.5-thinking": "gemini-3-pro-high",
    "claude-sonnet-4.5-thinking": "gemini-3-flash",
    "claude-sonnet-4.5": "gemini-2.5-flash",
}


def get_fallback_model(model: str) -> Optional[str]:
    """Get fallback model for a given model when quota is exhausted."""
    return MODEL_FALLBACK_MAP.get(model)


# Error classes
class AntigravityError(BaseLLMException):
    """Base error class for Antigravity provider."""

    def __init__(
        self,
        status_code: int,
        message: str,
        request: Optional[httpx.Request] = None,
        response: Optional[httpx.Response] = None,
        headers: Optional[Union[httpx.Headers, dict]] = None,
        body: Optional[dict] = None,
    ):
        super().__init__(
            status_code=status_code,
            message=message,
            request=request,
            response=response,
            headers=headers,
            body=body,
        )


class AntigravityAuthError(AntigravityError):
    """Authentication error - OAuth token issues."""

    pass


class AntigravityRateLimitError(AntigravityError):
    """Rate limit exceeded for account/model."""

    pass


class AntigravityQuotaExhaustedError(AntigravityError):
    """Quota exhausted for all accounts."""

    pass


class AntigravityInvalidCredentialsError(AntigravityError):
    """Invalid or expired credentials."""

    pass


class AntigravityNoAccountsError(AntigravityError):
    """No accounts available/configured."""

    pass


def is_rate_limit_error(error: Exception) -> bool:
    """Check if an error is a rate limit error."""
    message = str(error).lower()
    return (
        "429" in message or "resource_exhausted" in message or "quota_exhausted" in message or "rate limit" in message
    )


def is_auth_error(error: Exception) -> bool:
    """Check if an error is an authentication error."""
    message = str(error).lower()
    return "401" in message or "unauthenticated" in message or "authentication" in message or "invalid_grant" in message


def parse_reset_time_from_error(error_text: str) -> Optional[int]:
    """
    Parse quota reset time from error message.

    Args:
        error_text: Error message text

    Returns:
        Reset time in milliseconds, or None if not found
    """
    import re

    # Try to extract time patterns like "5h30m", "2h", "30m", "45s"
    patterns = [
        r"reset after (\d+)h(\d+)m(\d+)s",
        r"reset after (\d+)h(\d+)m",
        r"reset after (\d+)h",
        r"reset after (\d+)m(\d+)s",
        r"reset after (\d+)m",
        r"reset after (\d+)s",
    ]

    for pattern in patterns:
        match = re.search(pattern, error_text, re.IGNORECASE)
        if match:
            groups = match.groups()
            ms = 0
            if len(groups) == 3:  # h, m, s
                ms = int(groups[0]) * 3600000 + int(groups[1]) * 60000 + int(groups[2]) * 1000
            elif len(groups) == 2:
                if "h" in pattern and "m" in pattern:  # h, m
                    ms = int(groups[0]) * 3600000 + int(groups[1]) * 60000
                else:  # m, s
                    ms = int(groups[0]) * 60000 + int(groups[1]) * 1000
            else:  # single unit
                if "h" in pattern:
                    ms = int(groups[0]) * 3600000
                elif "m" in pattern:
                    ms = int(groups[0]) * 60000
                else:
                    ms = int(groups[0]) * 1000
            return ms

    return None
