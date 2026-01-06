# Antigravity

Antigravity provides access to Claude and Gemini models via Google's Cloud Code service with multi-account support and automatic failover.

## Supported Models

| Model | Description |
|-------|-------------|
| `antigravity/claude-sonnet-4.5-thinking` | Claude Sonnet 4.5 with thinking |
| `antigravity/claude-opus-4.5-thinking` | Claude Opus 4.5 with thinking |
| `antigravity/claude-sonnet-4.5` | Claude Sonnet 4.5 |
| `antigravity/gemini-3-flash` | Gemini 3 Flash |
| `antigravity/gemini-3-pro-high` | Gemini 3 Pro (high quota) |
| `antigravity/gemini-3-pro-low` | Gemini 3 Pro (low quota) |
| `antigravity/gemini-2.5-flash` | Gemini 2.5 Flash |
| `antigravity/gemini-2.5-pro` | Gemini 2.5 Pro |

## Quick Start

```python
import litellm

response = litellm.completion(
    model="antigravity/claude-sonnet-4.5-thinking",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Authentication

Antigravity uses Google OAuth for authentication. Accounts are stored in `~/.config/litellm/antigravity/accounts.json`.

### Adding an Account

```python
from litellm.llms.antigravity import get_account_manager

manager = get_account_manager()
account = manager.add_account_interactive()
print(f"Added account: {account['email']}")
```

This will open a browser for Google OAuth authentication.

## Multi-Account Support

Antigravity supports multiple accounts with automatic failover when one account is rate-limited or out of quota.

### How It Works

1. **Sticky Selection**: The provider uses sticky account selection to maximize prompt caching efficiency
2. **Automatic Failover**: When an account is rate-limited, it automatically switches to the next available account
3. **Model-Specific Rate Limits**: Rate limits are tracked per model per account
4. **Fallback Models**: When all accounts are exhausted for a model, it can fallback to an equivalent model

### Checking Account Status

```python
from litellm.llms.antigravity import get_account_manager

manager = get_account_manager()
status = manager.get_status()
print(f"Total accounts: {status['total']}")
print(f"Available: {status['available']}")
print(f"Rate-limited: {status['rate_limited']}")
```

## Thinking/Reasoning Support

Models with `-thinking` suffix support extended thinking output:

```python
response = litellm.completion(
    model="antigravity/claude-sonnet-4.5-thinking",
    messages=[{"role": "user", "content": "Solve this step by step..."}],
    thinking={"budget_tokens": 16000}  # Optional thinking budget
)
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTIGRAVITY_API_BASE` | Override default API endpoint |
| `ANTIGRAVITY_API_KEY` | API key (usually not needed with OAuth) |

### LiteLLM Proxy Config

```yaml
model_list:
  - model_name: claude-thinking
    litellm_params:
      model: antigravity/claude-sonnet-4.5-thinking

  - model_name: gemini-3
    litellm_params:
      model: antigravity/gemini-3-flash
```

## Error Handling

```python
from litellm.llms.antigravity import (
    AntigravityRateLimitError,
    AntigravityQuotaExhaustedError,
    AntigravityNoAccountsError,
)

try:
    response = litellm.completion(
        model="antigravity/claude-sonnet-4.5",
        messages=[{"role": "user", "content": "Hello"}]
    )
except AntigravityNoAccountsError:
    print("No accounts configured. Run add_account_interactive()")
except AntigravityQuotaExhaustedError as e:
    print(f"All accounts exhausted: {e}")
except AntigravityRateLimitError as e:
    print(f"Rate limited: {e}")
```
