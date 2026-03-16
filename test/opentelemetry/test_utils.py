# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for autogen.opentelemetry.utils module."""

from typing import Any
from unittest.mock import MagicMock

from autogen.opentelemetry.utils import (
    API_TYPE_TO_PROVIDER,
    aggregate_usage,
    get_model_from_config_list,
    get_model_name,
    get_provider_from_config_list,
    get_provider_name,
    message_to_otel,
    messages_to_otel,
    reply_to_otel_message,
    set_llm_request_params,
)


# ---------------------------------------------------------------------------
# message_to_otel
# ---------------------------------------------------------------------------
class TestMessageToOtel:
    """Tests for converting a single AG2/OpenAI message to OTEL format."""

    def test_simple_text_message(self) -> None:
        msg = {"role": "user", "content": "Hello, world!"}
        result = message_to_otel(msg)
        assert result == {
            "role": "user",
            "parts": [{"type": "text", "content": "Hello, world!"}],
        }

    def test_assistant_text_message(self) -> None:
        msg = {"role": "assistant", "content": "I can help with that."}
        result = message_to_otel(msg)
        assert result["role"] == "assistant"
        assert len(result["parts"]) == 1
        assert result["parts"][0] == {"type": "text", "content": "I can help with that."}

    def test_system_message(self) -> None:
        msg = {"role": "system", "content": "You are a helpful assistant."}
        result = message_to_otel(msg)
        assert result["role"] == "system"
        assert result["parts"] == [{"type": "text", "content": "You are a helpful assistant."}]

    def test_message_with_name(self) -> None:
        msg = {"role": "user", "content": "Hello", "name": "user_proxy"}
        result = message_to_otel(msg)
        assert result["role"] == "user"
        assert result["parts"] == [{"type": "text", "content": "Hello"}]

    def test_tool_calls_message(self) -> None:
        msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "London"}',
                    },
                }
            ],
        }
        result = message_to_otel(msg)
        assert result["role"] == "assistant"
        assert len(result["parts"]) == 1
        part = result["parts"][0]
        assert part["type"] == "tool_call"
        assert part["id"] == "call_123"
        assert part["name"] == "get_weather"
        assert part["arguments"] == {"city": "London"}

    def test_tool_calls_with_dict_arguments(self) -> None:
        msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_456",
                    "function": {
                        "name": "search",
                        "arguments": {"query": "test"},
                    },
                }
            ],
        }
        result = message_to_otel(msg)
        part = result["parts"][0]
        # Already a dict, should remain unchanged
        assert part["arguments"] == {"query": "test"}

    def test_tool_calls_with_invalid_json_arguments(self) -> None:
        msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_789",
                    "function": {
                        "name": "fn",
                        "arguments": "not valid json",
                    },
                }
            ],
        }
        result = message_to_otel(msg)
        part = result["parts"][0]
        # Should remain as string when JSON parsing fails
        assert part["arguments"] == "not valid json"

    def test_multiple_tool_calls(self) -> None:
        msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "fn_a", "arguments": "{}"},
                },
                {
                    "id": "call_2",
                    "function": {"name": "fn_b", "arguments": '{"x": 1}'},
                },
            ],
        }
        result = message_to_otel(msg)
        assert len(result["parts"]) == 2
        assert result["parts"][0]["name"] == "fn_a"
        assert result["parts"][1]["name"] == "fn_b"

    def test_tool_response_message(self) -> None:
        msg = {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "The weather is sunny.",
        }
        result = message_to_otel(msg)
        assert result["role"] == "tool"
        assert len(result["parts"]) == 1
        part = result["parts"][0]
        assert part["type"] == "tool_call_response"
        assert part["id"] == "call_123"
        assert part["response"] == "The weather is sunny."

    def test_multimodal_content_list(self) -> None:
        msg = {
            "role": "user",
            "content": [
                "Describe this image:",
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            ],
        }
        result = message_to_otel(msg)
        assert result["role"] == "user"
        assert len(result["parts"]) == 2
        assert result["parts"][0] == {"type": "text", "content": "Describe this image:"}
        assert result["parts"][1] == {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}

    def test_empty_content(self) -> None:
        msg = {"role": "assistant", "content": ""}
        result = message_to_otel(msg)
        assert result["role"] == "assistant"
        assert result["parts"] == []

    def test_none_content(self) -> None:
        msg = {"role": "assistant", "content": None}
        result = message_to_otel(msg)
        assert result["parts"] == []

    def test_missing_content_no_tool_calls(self) -> None:
        msg = {"role": "assistant"}
        result = message_to_otel(msg)
        assert result["parts"] == []

    def test_empty_tool_calls_list(self) -> None:
        msg = {"role": "assistant", "tool_calls": []}
        # Empty tool_calls list is falsy, so falls through to content check
        result = message_to_otel(msg)
        assert result["parts"] == []

    def test_default_role_is_user(self) -> None:
        msg = {"content": "no role specified"}
        result = message_to_otel(msg)
        assert result["role"] == "user"

    def test_tool_call_missing_function(self) -> None:
        msg = {
            "role": "assistant",
            "tool_calls": [{"id": "call_x"}],
        }
        result = message_to_otel(msg)
        part = result["parts"][0]
        assert part["name"] == ""
        assert part["arguments"] == {}

    def test_tool_response_missing_content(self) -> None:
        msg = {
            "role": "tool",
            "tool_call_id": "call_x",
        }
        result = message_to_otel(msg)
        part = result["parts"][0]
        assert part["response"] == ""


# ---------------------------------------------------------------------------
# messages_to_otel
# ---------------------------------------------------------------------------
class TestMessagesToOtel:
    """Tests for batch message conversion."""

    def test_empty_list(self) -> None:
        assert messages_to_otel([]) == []

    def test_single_message(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        result = messages_to_otel(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_multiple_messages(self) -> None:
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "how are you?"},
        ]
        result = messages_to_otel(msgs)
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "user"

    def test_mixed_message_types(self) -> None:
        msgs = [
            {"role": "user", "content": "call a function"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "c1", "function": {"name": "fn", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ]
        result = messages_to_otel(msgs)
        assert len(result) == 3
        assert result[0]["parts"][0]["type"] == "text"
        assert result[1]["parts"][0]["type"] == "tool_call"
        assert result[2]["parts"][0]["type"] == "tool_call_response"


# ---------------------------------------------------------------------------
# reply_to_otel_message
# ---------------------------------------------------------------------------
class TestReplyToOtelMessage:
    """Tests for converting agent replies to OTEL format."""

    def test_string_reply(self) -> None:
        result = reply_to_otel_message("Hello!")
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert msg["parts"] == [{"type": "text", "content": "Hello!"}]
        assert msg["finish_reason"] == "stop"

    def test_empty_string_reply(self) -> None:
        result = reply_to_otel_message("")
        assert len(result) == 1
        msg = result[0]
        assert msg["parts"] == [{"type": "text", "content": ""}]
        assert msg["finish_reason"] == "stop"

    def test_dict_reply_with_content(self) -> None:
        result = reply_to_otel_message({"content": "Some response"})
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert msg["parts"] == [{"type": "text", "content": "Some response"}]

    def test_dict_reply_with_tool_calls(self) -> None:
        reply = {
            "tool_calls": [
                {"id": "c1", "function": {"name": "fn", "arguments": '{"a": 1}'}},
            ]
        }
        result = reply_to_otel_message(reply)
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert msg["parts"][0]["type"] == "tool_call"

    def test_none_reply(self) -> None:
        result = reply_to_otel_message(None)
        assert result == []


# ---------------------------------------------------------------------------
# aggregate_usage
# ---------------------------------------------------------------------------
class TestAggregateUsage:
    """Tests for token usage aggregation."""

    def test_single_model(self) -> None:
        usage = {
            "gpt-4": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        result = aggregate_usage(usage)
        assert result is not None
        model, input_tokens, output_tokens = result
        assert model == "gpt-4"
        assert input_tokens == 100
        assert output_tokens == 50

    def test_multiple_models(self) -> None:
        usage = {
            "gpt-4": {"prompt_tokens": 100, "completion_tokens": 50},
            "gpt-3.5-turbo": {"prompt_tokens": 200, "completion_tokens": 100},
        }
        result = aggregate_usage(usage)
        assert result is not None
        model, input_tokens, output_tokens = result
        assert "gpt-4" in model
        assert "gpt-3.5-turbo" in model
        assert input_tokens == 300
        assert output_tokens == 150

    def test_empty_dict(self) -> None:
        result = aggregate_usage({})
        assert result is None

    def test_missing_token_keys(self) -> None:
        usage = {
            "model-x": {"some_other_field": 42},
        }
        result = aggregate_usage(usage)
        assert result is not None
        model, input_tokens, output_tokens = result
        assert model == "model-x"
        assert input_tokens == 0
        assert output_tokens == 0

    def test_partial_token_data(self) -> None:
        usage = {
            "model-a": {"prompt_tokens": 50},
        }
        result = aggregate_usage(usage)
        assert result is not None
        _, input_tokens, output_tokens = result
        assert input_tokens == 50
        assert output_tokens == 0


# ---------------------------------------------------------------------------
# API_TYPE_TO_PROVIDER mapping
# ---------------------------------------------------------------------------
class TestApiTypeToProvider:
    """Tests for the API type to provider mapping."""

    def test_known_providers(self) -> None:
        assert API_TYPE_TO_PROVIDER["openai"] == "openai"
        assert API_TYPE_TO_PROVIDER["azure"] == "azure.ai.openai"
        assert API_TYPE_TO_PROVIDER["anthropic"] == "anthropic"
        assert API_TYPE_TO_PROVIDER["bedrock"] == "aws.bedrock"
        assert API_TYPE_TO_PROVIDER["mistral"] == "mistral_ai"
        assert API_TYPE_TO_PROVIDER["groq"] == "groq"
        assert API_TYPE_TO_PROVIDER["cohere"] == "cohere"
        assert API_TYPE_TO_PROVIDER["deepseek"] == "deepseek"
        assert API_TYPE_TO_PROVIDER["together"] == "together"
        assert API_TYPE_TO_PROVIDER["ollama"] == "ollama"
        assert API_TYPE_TO_PROVIDER["cerebras"] == "cerebras"

    def test_unknown_provider_not_in_dict(self) -> None:
        assert "unknown_provider" not in API_TYPE_TO_PROVIDER


# ---------------------------------------------------------------------------
# get_provider_name / get_model_name
# ---------------------------------------------------------------------------
class TestGetProviderName:
    """Tests for extracting provider name from an agent."""

    def test_agent_without_llm_config(self) -> None:
        agent = MagicMock(spec=[])  # no llm_config attribute
        assert get_provider_name(agent) is None

    def test_agent_with_false_llm_config(self) -> None:
        agent = MagicMock()
        agent.llm_config = False
        assert get_provider_name(agent) is None

    def test_agent_with_none_llm_config(self) -> None:
        agent = MagicMock()
        agent.llm_config = None
        assert get_provider_name(agent) is None

    def test_agent_with_empty_config_list(self) -> None:
        agent = MagicMock()
        agent.llm_config.config_list = None
        assert get_provider_name(agent) is None

    def test_agent_with_openai_api_type(self) -> None:
        config_entry = MagicMock()
        config_entry.api_type = "openai"
        agent = MagicMock()
        agent.llm_config.config_list = [config_entry]
        assert get_provider_name(agent) == "openai"

    def test_agent_with_azure_api_type(self) -> None:
        config_entry = MagicMock()
        config_entry.api_type = "azure"
        agent = MagicMock()
        agent.llm_config.config_list = [config_entry]
        assert get_provider_name(agent) == "azure.ai.openai"

    def test_agent_with_unknown_api_type(self) -> None:
        config_entry = MagicMock()
        config_entry.api_type = "custom_provider"
        agent = MagicMock()
        agent.llm_config.config_list = [config_entry]
        assert get_provider_name(agent) == "custom_provider"

    def test_agent_with_no_api_type_on_object(self) -> None:
        """When api_type is None on the config object, provider should be None."""
        config_entry = MagicMock(spec=["api_type", "get"])
        config_entry.api_type = None
        config_entry.get = MagicMock(return_value=None)
        agent = MagicMock()
        agent.llm_config.config_list = [config_entry]
        assert get_provider_name(agent) is None

    def test_agent_with_dict_config_entry(self) -> None:
        """Test with a dict-based config entry (no api_type attribute)."""
        agent = MagicMock()
        config_entry = MagicMock(spec=[])  # No attributes
        # When getattr fails, it tries .get()
        config_entry.get = MagicMock(return_value="anthropic")
        agent.llm_config.config_list = [config_entry]
        result = get_provider_name(agent)
        assert result == "anthropic"


class TestGetModelName:
    """Tests for extracting model name from an agent."""

    def test_agent_without_llm_config(self) -> None:
        agent = MagicMock(spec=[])
        assert get_model_name(agent) is None

    def test_agent_with_false_llm_config(self) -> None:
        agent = MagicMock()
        agent.llm_config = False
        assert get_model_name(agent) is None

    def test_agent_with_model(self) -> None:
        config_entry = MagicMock()
        config_entry.model = "gpt-4"
        agent = MagicMock()
        agent.llm_config.config_list = [config_entry]
        assert get_model_name(agent) == "gpt-4"

    def test_agent_with_empty_config_list(self) -> None:
        agent = MagicMock()
        agent.llm_config.config_list = None
        assert get_model_name(agent) is None


# ---------------------------------------------------------------------------
# get_provider_from_config_list / get_model_from_config_list
# ---------------------------------------------------------------------------
class TestGetProviderFromConfigList:
    """Tests for extracting provider from a config list."""

    def test_empty_config_list(self) -> None:
        assert get_provider_from_config_list([]) == "openai"

    def test_dict_config_with_api_type(self) -> None:
        config_list = [{"api_type": "anthropic", "model": "claude-3"}]
        assert get_provider_from_config_list(config_list) == "anthropic"

    def test_dict_config_without_api_type(self) -> None:
        config_list = [{"model": "gpt-4"}]
        assert get_provider_from_config_list(config_list) == "openai"

    def test_dict_config_unknown_api_type(self) -> None:
        config_list = [{"api_type": "my_custom"}]
        assert get_provider_from_config_list(config_list) == "my_custom"

    def test_object_config_with_api_type(self) -> None:
        config = MagicMock()
        config.api_type = "bedrock"
        assert get_provider_from_config_list([config]) == "aws.bedrock"

    def test_object_config_without_api_type(self) -> None:
        config = MagicMock(spec=[])  # No api_type attribute
        assert get_provider_from_config_list([config]) == "openai"

    def test_uses_first_config_entry(self) -> None:
        config_list = [
            {"api_type": "anthropic"},
            {"api_type": "openai"},
        ]
        assert get_provider_from_config_list(config_list) == "anthropic"


class TestGetModelFromConfigList:
    """Tests for extracting model from a config list."""

    def test_empty_config_list(self) -> None:
        assert get_model_from_config_list([]) is None

    def test_dict_config_with_model(self) -> None:
        config_list = [{"model": "gpt-4o", "api_key": "test"}]
        assert get_model_from_config_list(config_list) == "gpt-4o"

    def test_dict_config_without_model(self) -> None:
        config_list = [{"api_key": "test"}]
        assert get_model_from_config_list(config_list) is None

    def test_object_config_with_model(self) -> None:
        config = MagicMock()
        config.model = "claude-3-opus"
        assert get_model_from_config_list([config]) == "claude-3-opus"

    def test_object_config_without_model(self) -> None:
        config = MagicMock(spec=[])
        assert get_model_from_config_list([config]) is None


# ---------------------------------------------------------------------------
# set_llm_request_params
# ---------------------------------------------------------------------------
class TestSetLlmRequestParams:
    """Tests for setting LLM request parameters on a span."""

    def test_all_params_present(self) -> None:
        span = MagicMock()
        config = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3,
        }
        set_llm_request_params(span, config)
        span.set_attribute.assert_any_call("gen_ai.request.temperature", 0.7)
        span.set_attribute.assert_any_call("gen_ai.request.max_tokens", 1000)
        span.set_attribute.assert_any_call("gen_ai.request.top_p", 0.9)
        span.set_attribute.assert_any_call("gen_ai.request.frequency_penalty", 0.5)
        span.set_attribute.assert_any_call("gen_ai.request.presence_penalty", 0.3)
        assert span.set_attribute.call_count == 5

    def test_no_params_present(self) -> None:
        span = MagicMock()
        config: dict[str, Any] = {"model": "gpt-4", "messages": []}
        set_llm_request_params(span, config)
        span.set_attribute.assert_not_called()

    def test_partial_params(self) -> None:
        span = MagicMock()
        config = {"temperature": 0.5}
        set_llm_request_params(span, config)
        span.set_attribute.assert_called_once_with("gen_ai.request.temperature", 0.5)

    def test_none_value_skipped(self) -> None:
        span = MagicMock()
        config = {"temperature": None, "max_tokens": 500}
        set_llm_request_params(span, config)
        span.set_attribute.assert_called_once_with("gen_ai.request.max_tokens", 500)

    def test_zero_value_included(self) -> None:
        span = MagicMock()
        config = {"temperature": 0, "frequency_penalty": 0.0}
        set_llm_request_params(span, config)
        assert span.set_attribute.call_count == 2
        span.set_attribute.assert_any_call("gen_ai.request.temperature", 0)
        span.set_attribute.assert_any_call("gen_ai.request.frequency_penalty", 0.0)

    def test_empty_config(self) -> None:
        span = MagicMock()
        set_llm_request_params(span, {})
        span.set_attribute.assert_not_called()
