# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("anthropic")

from anthropic.types import ServerToolUseBlock

from autogen.beta.ag_ui.mappers import call_from_agui
from autogen.beta.ag_ui.mappers.anthropic import anthropic_call_from_agui
from autogen.beta.config.anthropic.events import AnthropicServerToolCallEvent
from autogen.beta.tools.builtin.code_execution import CODE_EXECUTION_TOOL_NAME
from autogen.beta.tools.builtin.web_fetch import WEB_FETCH_TOOL_NAME
from autogen.beta.tools.builtin.web_search import WEB_SEARCH_TOOL_NAME


class TestWebSearch:
    def test_round_trips(self) -> None:
        block = ServerToolUseBlock(
            id="w1",
            name="web_search",
            input={"query": "bitcoin"},
            type="server_tool_use",
        )

        forward = AnthropicServerToolCallEvent.from_block(block)
        restored = call_from_agui(forward.name, forward.id, forward.arguments)

        assert restored == forward


class TestWebFetch:
    def test_round_trips(self) -> None:
        block = ServerToolUseBlock(
            id="f1",
            name="web_fetch",
            input={"url": "https://example.com"},
            type="server_tool_use",
        )

        forward = AnthropicServerToolCallEvent.from_block(block)
        restored = call_from_agui(forward.name, forward.id, forward.arguments)

        assert restored == forward


class TestCodeExecution:
    def test_code_execution_kind_round_trips(self) -> None:
        block = ServerToolUseBlock(
            id="c1",
            name="code_execution",
            input={"code": "print(1)"},
            type="server_tool_use",
        )

        forward = AnthropicServerToolCallEvent.from_block(block)
        restored = call_from_agui(forward.name, forward.id, forward.arguments)

        assert restored == forward

    def test_bash_code_execution_kind_round_trips(self) -> None:
        block = ServerToolUseBlock(
            id="b1",
            name="bash_code_execution",
            input={"cmd": "ls"},
            type="server_tool_use",
        )

        forward = AnthropicServerToolCallEvent.from_block(block)
        restored = call_from_agui(forward.name, forward.id, forward.arguments)

        assert restored == forward

    def test_text_editor_code_execution_kind_round_trips(self) -> None:
        block = ServerToolUseBlock(
            id="t1",
            name="text_editor_code_execution",
            input={"command": "view", "path": "/tmp/f"},
            type="server_tool_use",
        )

        forward = AnthropicServerToolCallEvent.from_block(block)
        restored = call_from_agui(forward.name, forward.id, forward.arguments)

        assert restored == forward


class TestNonMatching:
    def test_unknown_name_returns_none(self) -> None:
        assert anthropic_call_from_agui("my_tool", "id_1", {"x": 1}) is None

    def test_web_search_without_query_returns_none(self) -> None:
        # OpenAI web_search payload (has `type`, not `query`) — Anthropic mapper
        # must not claim it.
        assert anthropic_call_from_agui(WEB_SEARCH_TOOL_NAME, "ws_x", {"type": "search"}) is None

    def test_web_fetch_without_url_returns_none(self) -> None:
        # Gemini grounding-derived web_fetch carries `queries` instead of `url`
        # — Anthropic mapper must defer to the Gemini one.
        assert anthropic_call_from_agui(WEB_FETCH_TOOL_NAME, "wf_x", {"queries": ["x"]}) is None

    def test_code_execution_without_kind_returns_none(self) -> None:
        # Gemini / OpenAI code_execution payload — no `_kind` field, must skip.
        assert anthropic_call_from_agui(CODE_EXECUTION_TOOL_NAME, "ci_x", {"code": "x = 1"}) is None

    def test_code_execution_with_unknown_kind_returns_none(self) -> None:
        # Defensive: future SDK might add a new code-execution variant we don't
        # know yet — fall back rather than validating against a wrong shape.
        assert anthropic_call_from_agui(CODE_EXECUTION_TOOL_NAME, "ci_x", {"_kind": "telekinetic"}) is None
