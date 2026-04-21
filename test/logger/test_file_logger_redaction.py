# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _redact() in autogen/logger/file_logger.py.

These tests verify the credential-redaction logic added by
fix: redact sensitive keys in FileLogger output.

No external dependencies (no API keys, no Azure, no Redis) are required.
The _redact function is a pure in-process utility.
"""

from typing import Any

from autogen.logger.logger_utils import redact as _redact


class TestRedactSensitiveKeys:
    """Direct unit tests for _redact()."""

    # ------------------------------------------------------------------
    # Flat dict cases
    # ------------------------------------------------------------------

    def test_api_key_redacted(self) -> None:
        data: dict[str, Any] = {"api_key": "sk-abc123", "model": "gpt-4"}
        result = _redact(data)
        assert result["api_key"] == "***REDACTED***"
        assert result["model"] == "gpt-4"

    def test_unknown_key_not_redacted(self) -> None:
        """Keys not in the exact sensitive list must pass through."""
        data: dict[str, Any] = {"openai_api_key": "sk-xyz", "custom_secret_key": "shh"}
        result = _redact(data)
        assert result["openai_api_key"] == "sk-xyz"
        assert result["custom_secret_key"] == "shh"

    def test_azure_ad_token_redacted(self) -> None:
        data: dict[str, Any] = {"azure_ad_token": "eyJ..."}
        result = _redact(data)
        assert result["azure_ad_token"] == "***REDACTED***"

    def test_hyphenated_api_key_redacted(self) -> None:
        """api-key (hyphen variant) must also be redacted."""
        data: dict[str, Any] = {"api-key": "sk-secret"}
        result = _redact(data)
        assert result["api-key"] == "***REDACTED***"

    def test_password_redacted(self) -> None:
        data: dict[str, Any] = {"password": "hunter2"}
        result = _redact(data)
        assert result["password"] == "***REDACTED***"

    def test_specific_token_keys_redacted(self) -> None:
        """Specific token keys (access_token, refresh_token) must be redacted."""
        data: dict[str, Any] = {"access_token": "abc", "refresh_token": "xyz", "auth_token": "tok"}
        result = _redact(data)
        assert result["access_token"] == "***REDACTED***"
        assert result["refresh_token"] == "***REDACTED***"
        assert result["auth_token"] == "***REDACTED***"

    def test_azure_ad_token_variants_redacted(self) -> None:
        """azure_ad_token and azure_ad_token_provider must be redacted."""
        data: dict[str, Any] = {"azure_ad_token": "eyJ...", "azure_ad_token_provider": "func"}
        result = _redact(data)
        assert result["azure_ad_token"] == "***REDACTED***"
        assert result["azure_ad_token_provider"] == "***REDACTED***"

    def test_base_url_not_redacted(self) -> None:
        """base_url is not sensitive -- useful for debugging."""
        data: dict[str, Any] = {"base_url": "https://custom.openai.azure.com"}
        result = _redact(data)
        assert result["base_url"] == "https://custom.openai.azure.com"

    def test_llm_token_params_not_redacted(self) -> None:
        """Common LLM parameters containing 'token' must NOT be redacted."""
        data: dict[str, Any] = {
            "max_tokens": 4096,
            "prompt_tokens": 150,
            "completion_tokens": 200,
            "total_tokens": 350,
            "token_count": 500,
            "num_tokens": 1024,
        }
        result = _redact(data)
        assert result == data

    def test_non_sensitive_keys_unchanged(self) -> None:
        data: dict[str, Any] = {"model": "gpt-4", "temperature": 0.7, "messages": []}
        result = _redact(data)
        assert result == data

    def test_exact_key_variants_redacted(self) -> None:
        """All exact key variants (api_key, api-key, apikey) must be redacted."""
        data: dict[str, Any] = {"api_key": "sk-1", "api-key": "sk-2", "apikey": "sk-3"}
        result = _redact(data)
        assert result["api_key"] == "***REDACTED***"
        assert result["api-key"] == "***REDACTED***"
        assert result["apikey"] == "***REDACTED***"

    def test_remaining_sensitive_keys_redacted(self) -> None:
        """All other keys in SENSITIVE_KEYS must be redacted."""
        data: dict[str, Any] = {
            "secret": "shh",
            "credential": "cred",
            "authorization": "Bearer sk-123",
            "bearer": "tok",
            "api_token": "tok",
            "azure_endpoint": "https://my.azure.com",
        }
        result = _redact(data)
        for key in data:
            assert result[key] == "***REDACTED***", f"{key} was not redacted"

    def test_case_insensitive_matching(self) -> None:
        """Key matching must be case-insensitive."""
        data: dict[str, Any] = {"API_KEY": "sk-1", "Password": "hunter2", "SECRET": "shh"}
        result = _redact(data)
        assert result["API_KEY"] == "***REDACTED***"
        assert result["Password"] == "***REDACTED***"
        assert result["SECRET"] == "***REDACTED***"

    # ------------------------------------------------------------------
    # Nested dict cases
    # ------------------------------------------------------------------

    def test_nested_sensitive_key_redacted(self) -> None:
        data: dict[str, Any] = {"config": {"api_key": "sk-abc123", "temperature": 0.7}}
        result = _redact(data)
        assert result["config"]["api_key"] == "***REDACTED***"
        assert result["config"]["temperature"] == 0.7

    def test_doubly_nested_redacted(self) -> None:
        data: dict[str, Any] = {"outer": {"inner": {"api_key": "secret"}}}
        result = _redact(data)
        assert result["outer"]["inner"]["api_key"] == "***REDACTED***"

    # ------------------------------------------------------------------
    # List / tuple / set cases
    # ------------------------------------------------------------------

    def test_list_with_dicts_redacted(self) -> None:
        data: list[dict[str, Any]] = [{"api_key": "secret"}, {"name": "safe"}]
        result = _redact(data)
        assert result[0]["api_key"] == "***REDACTED***"
        assert result[1]["name"] == "safe"

    def test_list_type_preserved(self) -> None:
        data: list[dict[str, Any]] = [{"api_key": "sk"}]
        result = _redact(data)
        assert isinstance(result, list)

    def test_tuple_type_preserved(self) -> None:
        data = ({"api_key": "sk"}, {"name": "x"})
        result = _redact(data)
        assert isinstance(result, tuple)
        assert result[0]["api_key"] == "***REDACTED***"

    # ------------------------------------------------------------------
    # Depth limit
    # ------------------------------------------------------------------

    def test_depth_limit_prevents_infinite_recursion(self) -> None:
        """Deeply nested structure (>10 levels) must not crash."""
        data: dict[str, Any] = {}
        current: dict[str, Any] = data
        for _ in range(20):
            current["next"] = {}
            current = current["next"]
        current["api_key"] = "deep_secret"
        # Must complete without raising RecursionError
        result = _redact(data)
        assert isinstance(result, dict)

    def test_default_depth_is_ten(self) -> None:
        """At depth 10, the value is returned as-is (no redaction)."""
        # Build a dict that puts api_key exactly at depth 11 (beyond default of 10)
        data: dict[str, Any] = {}
        current: dict[str, Any] = data
        for _ in range(10):
            current["child"] = {}
            current = current["child"]
        current["api_key"] = "should_not_be_redacted"
        result = _redact(data)
        # Navigate down to the leaf
        node = result
        for _ in range(10):
            node = node["child"]
        # At depth 10 the recursion limit is reached; original value is returned
        assert node["api_key"] == "should_not_be_redacted"

    # ------------------------------------------------------------------
    # Non-dict / scalar pass-through
    # ------------------------------------------------------------------

    def test_scalar_string_unchanged(self) -> None:
        assert _redact("plain string") == "plain string"

    def test_none_unchanged(self) -> None:
        assert _redact(None) is None

    def test_integer_unchanged(self) -> None:
        assert _redact(42) == 42
