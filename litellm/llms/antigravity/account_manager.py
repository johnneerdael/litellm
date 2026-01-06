import time
from typing import Any, Dict, List, Optional, Tuple

from litellm._logging import verbose_logger

from .authenticator import Authenticator
from .common_utils import (
    DEFAULT_COOLDOWN_MS,
    MAX_WAIT_BEFORE_ERROR_MS,
    AntigravityNoAccountsError,
    AntigravityQuotaExhaustedError,
    AntigravityRateLimitError,
    is_auth_error,
    is_rate_limit_error,
    parse_reset_time_from_error,
)


class AccountManager:
    def __init__(self):
        self._authenticator = Authenticator()
        self._current_index = 0
        self._rate_limits: Dict[str, Dict[str, Any]] = {}

    def get_account_count(self) -> int:
        return len(self._authenticator.get_accounts())

    def get_accounts(self) -> List[Dict[str, Any]]:
        return self._authenticator.get_accounts()

    def _get_rate_limit_key(self, email: str, model_id: Optional[str] = None) -> str:
        if model_id:
            return f"{email}:{model_id}"
        return email

    def _is_account_rate_limited(self, account: Dict[str, Any], model_id: Optional[str] = None) -> bool:
        email = account["email"]

        model_key = self._get_rate_limit_key(email, model_id)
        if model_key in self._rate_limits:
            limit = self._rate_limits[model_key]
            if limit.get("reset_time", 0) > time.time() * 1000:
                return True

        global_key = self._get_rate_limit_key(email)
        if global_key in self._rate_limits:
            limit = self._rate_limits[global_key]
            if limit.get("reset_time", 0) > time.time() * 1000:
                return True

        return False

    def _is_account_invalid(self, account: Dict[str, Any]) -> bool:
        return account.get("is_invalid", False)

    def get_available_accounts(self, model_id: Optional[str] = None) -> List[Dict[str, Any]]:
        accounts = self._authenticator.get_accounts()
        return [
            a for a in accounts if not self._is_account_rate_limited(a, model_id) and not self._is_account_invalid(a)
        ]

    def is_all_rate_limited(self, model_id: Optional[str] = None) -> bool:
        available = self.get_available_accounts(model_id)
        return len(available) == 0 and self.get_account_count() > 0

    def clear_expired_limits(self) -> int:
        now = time.time() * 1000
        expired = [k for k, v in self._rate_limits.items() if v.get("reset_time", 0) <= now]
        for k in expired:
            del self._rate_limits[k]
        return len(expired)

    def mark_rate_limited(
        self,
        email: str,
        reset_ms: Optional[int] = None,
        model_id: Optional[str] = None,
    ):
        key = self._get_rate_limit_key(email, model_id)
        reset_time = time.time() * 1000 + (reset_ms or DEFAULT_COOLDOWN_MS)
        self._rate_limits[key] = {
            "reset_time": reset_time,
            "model_id": model_id,
        }
        verbose_logger.info(f"Account {email} rate-limited for model {model_id or 'all'} until {reset_time}")

    def mark_invalid(self, email: str, reason: str = "Unknown"):
        accounts = self._authenticator.get_accounts()
        for account in accounts:
            if account["email"] == email:
                account["is_invalid"] = True
                account["invalid_reason"] = reason
                break
        verbose_logger.warning(f"Account {email} marked invalid: {reason}")

    def get_min_wait_time_ms(self, model_id: Optional[str] = None) -> int:
        now = time.time() * 1000
        min_wait = float("inf")

        for key, limit in self._rate_limits.items():
            if model_id and f":{model_id}" not in key and ":" in key:
                continue
            reset_time = limit.get("reset_time", 0)
            if reset_time > now:
                wait = reset_time - now
                min_wait = min(min_wait, wait)

        return int(min_wait) if min_wait != float("inf") else 0

    def pick_next(self, model_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        accounts = self._authenticator.get_accounts()
        if not accounts:
            return None

        start_index = self._current_index
        for _ in range(len(accounts)):
            self._current_index = (self._current_index + 1) % len(accounts)
            account = accounts[self._current_index]
            if not self._is_account_rate_limited(account, model_id) and not self._is_account_invalid(account):
                return account

        self._current_index = start_index
        return None

    def get_current_sticky_account(self, model_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        accounts = self._authenticator.get_accounts()
        if not accounts:
            return None

        if self._current_index >= len(accounts):
            self._current_index = 0

        account = accounts[self._current_index]
        if self._is_account_rate_limited(account, model_id) or self._is_account_invalid(account):
            return None

        return account

    def should_wait_for_current_account(
        self, model_id: Optional[str] = None
    ) -> Tuple[bool, int, Optional[Dict[str, Any]]]:
        accounts = self._authenticator.get_accounts()
        if not accounts or self._current_index >= len(accounts):
            return False, 0, None

        account = accounts[self._current_index]
        email = account["email"]

        if self._is_account_invalid(account):
            return False, 0, None

        key = self._get_rate_limit_key(email, model_id)
        global_key = self._get_rate_limit_key(email)

        now = time.time() * 1000
        wait_ms = 0

        for k in [key, global_key]:
            if k in self._rate_limits:
                reset_time = self._rate_limits[k].get("reset_time", 0)
                if reset_time > now:
                    wait_ms = max(wait_ms, int(reset_time - now))

        if wait_ms > 0 and wait_ms <= MAX_WAIT_BEFORE_ERROR_MS:
            return True, wait_ms, account

        return False, 0, None

    def pick_sticky_account(self, model_id: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], int]:
        sticky = self.get_current_sticky_account(model_id)
        if sticky:
            return sticky, 0

        should_wait, wait_ms, account = self.should_wait_for_current_account(model_id)
        if should_wait and wait_ms <= MAX_WAIT_BEFORE_ERROR_MS // 2:
            return None, wait_ms

        next_account = self.pick_next(model_id)
        return next_account, 0

    def get_token_for_account(self, account: Dict[str, Any]) -> str:
        return self._authenticator.get_token_for_account(account)

    def get_project_for_account(self, account: Dict[str, Any], token: str) -> str:
        return self._authenticator.get_project_for_account(account, token)

    def clear_token_cache(self, email: Optional[str] = None):
        self._authenticator.clear_token_cache(email)

    def clear_project_cache(self, email: Optional[str] = None):
        self._authenticator.clear_project_cache(email)

    def add_account_interactive(self) -> Dict[str, str]:
        return self._authenticator.add_account_interactive()

    def reset_all_rate_limits(self):
        self._rate_limits.clear()

    def get_status(self) -> Dict[str, Any]:
        accounts = self._authenticator.get_accounts()
        available = self.get_available_accounts()
        rate_limited = [a for a in accounts if self._is_account_rate_limited(a) and not self._is_account_invalid(a)]
        invalid = [a for a in accounts if self._is_account_invalid(a)]

        return {
            "total": len(accounts),
            "available": len(available),
            "rate_limited": len(rate_limited),
            "invalid": len(invalid),
            "accounts": [
                {
                    "email": a["email"],
                    "is_rate_limited": self._is_account_rate_limited(a),
                    "is_invalid": self._is_account_invalid(a),
                }
                for a in accounts
            ],
        }


_account_manager: Optional[AccountManager] = None


def get_account_manager() -> AccountManager:
    global _account_manager
    if _account_manager is None:
        _account_manager = AccountManager()
    return _account_manager
