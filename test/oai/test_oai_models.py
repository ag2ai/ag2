# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from autogen.import_utils import optional_import_block, run_for_optional_imports


def strip_descriptions(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively strip 'description' fields from a JSON schema.

    This is needed because the OpenAI SDK may include description fields
    in its Pydantic models' JSON schemas, which our local models don't have.
    We want to compare structural compatibility, not documentation.
    """
    if not isinstance(schema, dict):
        return schema

    result = {}
    for key, value in schema.items():
        if key == "description":
            continue
        elif isinstance(value, dict):
            result[key] = strip_descriptions(value)
        elif isinstance(value, list):
            result[key] = [strip_descriptions(item) if isinstance(item, dict) else item for item in value]
        else:
            result[key] = value
    return result


from autogen.oai.oai_models import (
    ChatCompletionMessage as ChatCompletionMessageLocal,
)
from autogen.oai.oai_models import (
    ChatCompletionMessageFunctionToolCall as ChatCompletionMessageFunctionToolCallLocal,
)
from autogen.oai.oai_models import (
    Choice as ChoiceLocal,
)
from autogen.oai.oai_models import (
    CompletionUsage as CompletionUsageLocal,
)
from autogen.oai.oai_models.chat_completion import ChatCompletion as ChatCompletionLocal

with optional_import_block():
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall
    from openai.types.completion_usage import CompletionUsage


@run_for_optional_imports(["openai"], "openai")
class TestOAIModels:
    def test_chat_completion_schema(self) -> None:
        local_schema = strip_descriptions(ChatCompletionLocal.model_json_schema())
        openai_schema = strip_descriptions(ChatCompletion.model_json_schema())
        assert local_schema == openai_schema

    def test_chat_completion_message_schema(self) -> None:
        local_schema = strip_descriptions(ChatCompletionMessageLocal.model_json_schema())
        openai_schema = strip_descriptions(ChatCompletionMessage.model_json_schema())
        assert local_schema == openai_schema

    def test_chat_completion_message_tool_call_schema(self) -> None:
        local_schema = strip_descriptions(ChatCompletionMessageFunctionToolCallLocal.model_json_schema())
        openai_schema = strip_descriptions(ChatCompletionMessageFunctionToolCall.model_json_schema())
        assert local_schema == openai_schema

    def test_choice_schema(self) -> None:
        local_schema = strip_descriptions(ChoiceLocal.model_json_schema())
        openai_schema = strip_descriptions(Choice.model_json_schema())
        assert local_schema == openai_schema

    def test_completion_usage_schema(self) -> None:
        local_schema = strip_descriptions(CompletionUsageLocal.model_json_schema())
        openai_schema = strip_descriptions(CompletionUsage.model_json_schema())
        assert local_schema == openai_schema
