# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.import_utils import optional_import_block, run_for_optional_imports
from autogen.oai.oai_models import (
    ChatCompletionMessage as ChatCompletionMessageLocal,
)
from autogen.oai.oai_models import (
    ChatCompletionMessageToolCall as ChatCompletionMessageToolCallLocal,
)
from autogen.oai.oai_models import (
    Choice as ChoiceLocal,
)
from autogen.oai.oai_models import (
    CompletionUsage as CompletionUsageLocal,
)
from autogen.oai.oai_models.chat_completion import ChatCompletion as ChatCompletionLocal

with optional_import_block():
    from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.completion_usage import CompletionUsage


@run_for_optional_imports(["openai"], "openai")
class TestOAIModels:
    def test_chat_completion_schema(self) -> None:
        assert ChatCompletionLocal.model_json_schema() == ChatCompletion.model_json_schema()

    def test_chat_completion_message_schema(self) -> None:
        assert ChatCompletionMessageLocal.model_json_schema() == ChatCompletionMessage.model_json_schema()

    def test_chat_completion_message_tool_call_schema(self) -> None:
        assert (
            ChatCompletionMessageToolCallLocal.model_json_schema() == ChatCompletionMessageToolCall.model_json_schema()
        )

    def test_choice_schema(self) -> None:
        assert ChoiceLocal.model_json_schema() == Choice.model_json_schema()

    def test_completion_usage_schema(self) -> None:
        assert CompletionUsageLocal.model_json_schema() == CompletionUsage.model_json_schema()
