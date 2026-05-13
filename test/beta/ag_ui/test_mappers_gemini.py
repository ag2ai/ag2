# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("google.genai")

from google.genai import types

from autogen.beta.ag_ui.mappers import call_from_agui
from autogen.beta.ag_ui.mappers.gemini import gemini_call_from_agui
from autogen.beta.config.gemini.events import GeminiServerToolCallEvent
from autogen.beta.tools.builtin.code_execution import CODE_EXECUTION_TOOL_NAME
from autogen.beta.tools.builtin.web_fetch import WEB_FETCH_TOOL_NAME
from autogen.beta.tools.builtin.web_search import WEB_SEARCH_TOOL_NAME


class TestExecutableCode:
    def test_round_trips(self) -> None:
        part = types.Part(executable_code=types.ExecutableCode(code="print(1)", language=types.Language.PYTHON))

        forward = GeminiServerToolCallEvent.from_executable_code(part)
        restored = call_from_agui(forward.name, forward.id, forward.arguments)

        assert restored == forward


class TestGroundingWebSearch:
    def test_round_trips(self) -> None:
        gm = types.GroundingMetadata(web_search_queries=["bitcoin price", "btc usd"])

        forward = GeminiServerToolCallEvent.from_grounding(gm, name=WEB_SEARCH_TOOL_NAME)
        restored = call_from_agui(forward.name, forward.id, forward.arguments)

        # grounding_metadata is shaped by the mapper from `queries` only, so
        # it's logically equal but not identical to forward.grounding_metadata
        # (which may carry chunks/supports the wire format doesn't preserve).
        # We compare the round-trippable fields directly.
        assert restored == GeminiServerToolCallEvent(
            id=forward.id,
            name=WEB_SEARCH_TOOL_NAME,
            arguments=forward.arguments,
            grounding_metadata=types.GroundingMetadata(web_search_queries=["bitcoin price", "btc usd"]),
        )


class TestGroundingWebFetch:
    def test_round_trips(self) -> None:
        gm = types.GroundingMetadata(web_search_queries=["example.com"])

        forward = GeminiServerToolCallEvent.from_grounding(gm, name=WEB_FETCH_TOOL_NAME)
        restored = call_from_agui(forward.name, forward.id, forward.arguments)

        assert restored == GeminiServerToolCallEvent(
            id=forward.id,
            name=WEB_FETCH_TOOL_NAME,
            arguments=forward.arguments,
            grounding_metadata=types.GroundingMetadata(web_search_queries=["example.com"]),
        )


class TestNonMatching:
    def test_unknown_name_returns_none(self) -> None:
        assert gemini_call_from_agui("my_tool", "id_1", {"x": 1}) is None

    def test_code_execution_without_language_returns_none(self) -> None:
        # OpenAI / Anthropic code_execution payload — no `language` field.
        assert gemini_call_from_agui(CODE_EXECUTION_TOOL_NAME, "ci_x", {"code": "print(1)"}) is None

    def test_web_search_without_queries_returns_none(self) -> None:
        # OpenAI / Anthropic web_search payload — no `queries` field.
        assert gemini_call_from_agui(WEB_SEARCH_TOOL_NAME, "ws_x", {"type": "search"}) is None
