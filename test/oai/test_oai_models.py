# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.import_utils import optional_import_block, skip_on_missing_imports
from autogen.oai.oai_models import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    Choice,
    CompletionUsage,
)

with optional_import_block():
    import openai


@skip_on_missing_imports(["openai"], "openai")
class TestOAIModels:
    def test_models(self) -> None:
        assert openai
        assert ChatCompletionMessage  # type: ignore
        assert ChatCompletionMessageToolCall  # type: ignore
        assert Choice  # type: ignore
        assert CompletionUsage  # type: ignore
        assert ChatCompletion  # type: ignore
