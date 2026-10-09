"""Configuration: read settings from environment variables (and .env) and validate them once.

Every other module receives a `Settings` object instead of calling os.getenv() itself.
A bad value (e.g. CHUNK_OVERLAP larger than CHUNK_SIZE) is reported once, at startup,
naming the variable to fix - instead of surfacing later as a confusing failure deep
inside chunking or retrieval.
"""

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, ValidationError, field_validator, model_validator

# The project folder. Relative paths (DATA_DIR, .env) are resolved against it, so uvicorn,
# streamlit and pytest all find the same files no matter which folder they are started from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""


class Settings(BaseModel):
    # --- Storage ---
    data_dir: Path = Field(default=Path("data"), validate_default=True)

    # --- Upload validation ---
    max_upload_mb: int = Field(default=20, ge=1, le=200)

    # --- Chunking (measured in characters, not tokens) ---
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=150, ge=0)

    # --- Embeddings: a local sentence-transformers model (free, runs on your machine) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- Retrieval ---
    top_k: int = Field(default=4, ge=1, le=20)

    # --- LLM: any OpenAI-compatible Chat Completions API ---
    # The fallback is a local server, so document text is only sent to a third party
    # (e.g. Groq) when .env explicitly says so.
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "llama3.2:3b"
    llm_api_key: SecretStr = SecretStr("")  # SecretStr prints as '**********', so it can't leak into logs
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, ge=64, le=8192)  # caps the answer length (and its cost)
    llm_timeout_seconds: float = Field(default=120.0, gt=0)

    # --- Frontend -> backend ---
    api_url: str = "http://127.0.0.1:8000"

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("data_dir")
    @classmethod
    def _resolve_data_dir(cls, path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path

    @field_validator("log_level", mode="before")
    @classmethod
    def _uppercase_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check_chunk_overlap(self) -> "Settings":
        # Each new chunk starts (chunk_size - chunk_overlap) characters after the previous one.
        # If that step is zero or negative, the chunker would never move forward.
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be smaller than "
                f"CHUNK_SIZE ({self.chunk_size})"
            )
        return self

    @property
    def uploads_dir(self) -> Path:
        """Where uploaded PDFs are kept, so the index can be rebuilt with different settings."""
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        """Where ChromaDB persists its vectors and metadata on disk."""
        return self.data_dir / "chroma"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def llm_is_local(self) -> bool:
        """True when the LLM runs on this machine: free, and documents never leave it."""
        return (urlparse(self.llm_base_url).hostname or "") in LOCAL_HOSTS


# Environment variable -> Settings field. Unset or empty variables keep the defaults above.
ENV_VARS = {
    "DATA_DIR": "data_dir",
    "MAX_UPLOAD_MB": "max_upload_mb",
    "CHUNK_SIZE": "chunk_size",
    "CHUNK_OVERLAP": "chunk_overlap",
    "EMBEDDING_MODEL": "embedding_model",
    "TOP_K": "top_k",
    "LLM_BASE_URL": "llm_base_url",
    "LLM_MODEL": "llm_model",
    "LLM_API_KEY": "llm_api_key",
    "LLM_TEMPERATURE": "llm_temperature",
    "LLM_MAX_TOKENS": "llm_max_tokens",
    "LLM_TIMEOUT_SECONDS": "llm_timeout_seconds",
    "API_URL": "api_url",
    "LOG_LEVEL": "log_level",
}


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build Settings from `env` (defaults to os.environ after loading the project's .env)."""
    if env is None:
        load_dotenv(PROJECT_ROOT / ".env")  # real environment variables still take priority
        env = os.environ

    values = {field: env[var].strip() for var, field in ENV_VARS.items() if env.get(var, "").strip()}
    try:
        return Settings(**values)  # Pydantic converts "1000" -> 1000, "data" -> Path("data")
    except ValidationError as exc:
        raise ConfigError(_describe(exc)) from None


def configure_logging(level: str = "INFO") -> None:
    """Print log records to the console in one format. Never log API keys or whole documents."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,  # replace any handlers a library installed before us
    )
    # Libraries that report routine steps at INFO: keep only their warnings and errors.
    for name in ("sentence_transformers", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _describe(exc: ValidationError) -> str:
    """List each problem by its environment variable name.

    Only Pydantic's message is shown, never the input value - so an invalid configuration
    can't print your API key.
    """
    field_to_var = {field: var for var, field in ENV_VARS.items()}
    lines = []
    for error in exc.errors():
        message = error["msg"].removeprefix("Value error, ")
        if error["loc"]:  # a single field failed, e.g. TOP_K=five
            lines.append(f"  - {field_to_var.get(error['loc'][0], error['loc'][0])}: {message}")
        else:  # a rule across fields failed, e.g. overlap >= chunk size (message names the variables)
            lines.append(f"  - {message}")
    return "Invalid configuration (check your .env):\n" + "\n".join(lines)
