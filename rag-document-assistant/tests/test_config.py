"""Tests for app/config.py: defaults, parsing, validation, and keeping the API key secret.

Each test passes an explicit dict to load_settings(), so your real .env is never read.
"""

import pytest

from app.config import PROJECT_ROOT, ConfigError, load_settings


def test_empty_environment_uses_defaults():
    settings = load_settings({})

    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 150
    assert settings.top_k == 4
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.llm_is_local


def test_values_are_parsed_from_strings():
    settings = load_settings(
        {"CHUNK_SIZE": "500", "CHUNK_OVERLAP": " 50 ", "TOP_K": "3", "LOG_LEVEL": "debug"}
    )

    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50
    assert settings.top_k == 3
    assert settings.log_level == "DEBUG"


def test_empty_value_falls_back_to_default():
    assert load_settings({"TOP_K": ""}).top_k == 4


def test_relative_data_dir_is_resolved_against_project_root():
    settings = load_settings({"DATA_DIR": "my_data"})

    assert settings.data_dir == PROJECT_ROOT / "my_data"
    assert settings.uploads_dir == PROJECT_ROOT / "my_data" / "uploads"
    assert settings.chroma_dir == PROJECT_ROOT / "my_data" / "chroma"


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ConfigError, match="CHUNK_OVERLAP"):
        load_settings({"CHUNK_SIZE": "200", "CHUNK_OVERLAP": "200"})


def test_invalid_value_names_the_environment_variable():
    with pytest.raises(ConfigError, match="TOP_K"):
        load_settings({"TOP_K": "five"})


def test_api_key_never_appears_when_settings_are_printed():
    settings = load_settings({"LLM_API_KEY": "sk-secret-123"})

    assert "sk-secret-123" not in repr(settings)
    assert "sk-secret-123" not in str(settings)
    assert settings.llm_api_key.get_secret_value() == "sk-secret-123"


def test_api_key_never_appears_in_config_errors():
    with pytest.raises(ConfigError) as excinfo:
        load_settings({"LLM_API_KEY": "sk-secret-123", "TOP_K": "five"})

    assert "sk-secret-123" not in str(excinfo.value)


@pytest.mark.parametrize(
    "url, is_local",
    [
        ("http://localhost:11434/v1", True),
        ("http://127.0.0.1:11434/v1", True),
        ("https://api.groq.com/openai/v1", False),
        ("https://api.openai.com/v1", False),
    ],
)
def test_llm_is_local(url, is_local):
    assert load_settings({"LLM_BASE_URL": url}).llm_is_local is is_local
