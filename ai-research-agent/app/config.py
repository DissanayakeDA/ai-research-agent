"""Configuration: read settings from environment variables (and .env) and validate them once.

Every other module receives a `Settings` object instead of calling os.getenv() itself.
That way a missing API key is reported once, at startup, with a helpful message -
instead of surfacing later as a confusing "401 Unauthorized" in the middle of the agent loop.
"""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


class Settings(BaseModel):
    openai_api_key: str
    search_api_key: str
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4.1-mini"
    max_iterations: int = Field(default=5, ge=2, le=20)  # >= 2: one tool round + one answer
    max_search_results: int = Field(default=5, ge=1, le=10)
    debug: bool = False


# Required variables and a hint shown to the user when one is missing.
REQUIRED_VARS = {
    "OPENAI_API_KEY": "API key for your OpenAI-compatible LLM provider",
    "SEARCH_API_KEY": "Tavily API key (free developer tier at https://tavily.com)",
}

# Optional variables -> Settings field. Empty values fall back to the defaults above.
OPTIONAL_VARS = {
    "LLM_BASE_URL": "llm_base_url",
    "LLM_MODEL": "llm_model",
    "MAX_ITERATIONS": "max_iterations",
    "MAX_SEARCH_RESULTS": "max_search_results",
    "DEBUG": "debug",
}


def load_settings(env: dict[str, str] | None = None) -> Settings:
    """Build Settings from `env` (defaults to os.environ after loading .env)."""
    if env is None:
        load_dotenv()
        env = dict(os.environ)

    missing = [name for name in REQUIRED_VARS if not env.get(name, "").strip()]
    if missing:
        details = "\n".join(f"  - {name}: {REQUIRED_VARS[name]}" for name in missing)
        raise ConfigError(
            f"Missing required environment variables:\n{details}\n\n"
            "Copy .env.example to .env and fill in your keys."
        )

    values = {
        "openai_api_key": env["OPENAI_API_KEY"].strip(),
        "search_api_key": env["SEARCH_API_KEY"].strip(),
    }
    for var, field in OPTIONAL_VARS.items():
        raw = env.get(var, "").strip()
        if raw:
            values[field] = raw  # Pydantic converts "5" -> 5 and "true" -> True

    try:
        return Settings(**values)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration:\n{exc}") from exc
