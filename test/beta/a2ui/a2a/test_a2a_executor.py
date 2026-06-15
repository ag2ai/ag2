# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest
from a2a.types import Message, Part, Role

from autogen.beta.a2a.mappers import ParsedMessage
from autogen.beta.a2ui import A2UIAction, A2UIAgent
from autogen.beta.a2ui.a2a import create_a2ui_parts, get_a2ui_data, is_a2ui_part
from autogen.beta.a2ui.a2a.executor import A2UIAgentExecutor, _extract_a2ui_envelopes
from autogen.beta.context import ConversationContext
from autogen.beta.events import TextInput
from autogen.beta.stream import MemoryStream
from autogen.beta.testing import TestConfig

VERSION = "v0.9"
CATALOG = "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json"

DELETE_SURFACE_MSG = {"version": VERSION, "deleteSurface": {"surfaceId": "s1"}}

ACTION_ENVELOPE = {
    "version": VERSION,
    "action": {
        "name": "submit",
        "surfaceId": "s1",
        "sourceComponentId": "submit_btn",
        "timestamp": "2026-06-14T00:00:00Z",
        "context": {"email": "user@example.com"},
    },
}

ERROR_ENVELOPE = {
    "version": VERSION,
    "error": {
        "code": "VALIDATION_FAILED",
        "surfaceId": "s1",
        "message": "bad component",
        "path": "/components/0",
    },
}

FUNCTION_RESPONSE_ENVELOPE = {
    "version": "v1.0",
    "functionResponse": {
        "functionCallId": "fc-1",
        "call": "openUrl",
        "value": True,
    },
}

# A server→client callFunction the LLM may emit inside <a2ui-json>. ``openUrl``
# is a basic-catalog function, so this validates under v1.0.
_CALL_FUNCTION_BLOCK = (
    '[{"version":"v1.0","functionCallId":"fc-1","wantResponse":true,'
    '"callFunction":{"call":"openUrl","args":{"url":"https://example.com"}}}]'
)


def _make_executor(actions: tuple[A2UIAction, ...] = ()) -> A2UIAgentExecutor:
    agent = A2UIAgent(
        name="ui_agent",
        config=TestConfig("ok"),
        validate_responses=False,
        actions=actions,
    )
    return A2UIAgentExecutor(agent)


class TestExtractA2UIEnvelopes:
    """Envelope decode — the three on-the-wire shapes from get_a2ui_data."""

    def test_returns_empty_for_text_part(self) -> None:
        assert _extract_a2ui_envelopes(Part(text="hello")) == []

    def test_canonical_list_payload(self) -> None:
        [part] = create_a2ui_parts([ACTION_ENVELOPE])
        assert _extract_a2ui_envelopes(part) == [ACTION_ENVELOPE]

    def test_legacy_single_dict_payload(self) -> None:
        [part] = create_a2ui_parts(ACTION_ENVELOPE, legacy_split=True)
        assert _extract_a2ui_envelopes(part) == [ACTION_ENVELOPE]

    def test_filters_entries_without_action_or_error(self) -> None:
        surface_only = {"version": VERSION, "createSurface": {"surfaceId": "s1", "catalogId": CATALOG}}
        [part] = create_a2ui_parts([surface_only, ACTION_ENVELOPE])
        assert _extract_a2ui_envelopes(part) == [ACTION_ENVELOPE]

    def test_decodes_error_envelope(self) -> None:
        [part] = create_a2ui_parts([ERROR_ENVELOPE])
        assert _extract_a2ui_envelopes(part) == [ERROR_ENVELOPE]

    def test_decodes_function_response_envelope(self) -> None:
        [part] = create_a2ui_parts([FUNCTION_RESPONSE_ENVELOPE])
        assert _extract_a2ui_envelopes(part) == [FUNCTION_RESPONSE_ENVELOPE]


class TestBuildA2UIMessage:
    """Splitting the completed turn into prose text + canonical A2UI DataPart.

    The A2UI messages arrive as a collected list (gathered from the stream's
    A2UIMessageEvents), and the prose is already stripped by the middleware.
    """

    def test_splits_text_and_a2ui_datapart(self) -> None:
        executor = _make_executor()
        updater = MagicMock()
        executor._build_a2ui_message(updater, "Here is your UI.", [DELETE_SURFACE_MSG], {})

        parts = updater.new_agent_message.call_args.kwargs["parts"]
        assert len(parts) == 2
        assert parts[0].text == "Here is your UI."
        assert is_a2ui_part(parts[1])
        assert get_a2ui_data(parts[1]) == [DELETE_SURFACE_MSG]

    def test_a2ui_only_without_prose(self) -> None:
        executor = _make_executor()
        updater = MagicMock()
        executor._build_a2ui_message(updater, "", [DELETE_SURFACE_MSG], {})

        parts = updater.new_agent_message.call_args.kwargs["parts"]
        assert len(parts) == 1
        assert is_a2ui_part(parts[0])
        assert get_a2ui_data(parts[0]) == [DELETE_SURFACE_MSG]

    def test_plain_text_single_part(self) -> None:
        executor = _make_executor()
        updater = MagicMock()
        executor._build_a2ui_message(updater, "Just text.", [], {})

        parts = updater.new_agent_message.call_args.kwargs["parts"]
        assert len(parts) == 1
        assert parts[0].text == "Just text."

    def test_returns_none_when_no_parts_and_no_variables(self) -> None:
        executor = _make_executor()
        updater = MagicMock()
        assert executor._build_a2ui_message(updater, "", [], {}) is None
        updater.new_agent_message.assert_not_called()

    def test_context_update_variables_go_to_metadata(self) -> None:
        executor = _make_executor()
        updater = MagicMock()
        executor._build_a2ui_message(updater, "Just text.", [], {"count": 3})

        metadata = updater.new_agent_message.call_args.kwargs["metadata"]
        assert metadata is not None
        assert any(v == {"count": 3} for v in metadata.values())


class TestActionToPrompt:
    """Prompt synthesis from incoming client actions."""

    def test_tool_action_synthesizes_tool_call_prompt(self) -> None:
        executor = _make_executor((A2UIAction("submit", tool_name="submit_form"),))
        msg = MagicMock()
        msg.parts = [create_a2ui_parts([ACTION_ENVELOPE])[0]]
        ctx = MagicMock()
        ctx.message = msg

        executor._rewrite_incoming_a2ui_parts(ctx)

        texts = [p.text for p in msg.parts if p.text]
        assert len(texts) == 1
        assert "submit" in texts[0]
        assert "submit_form" in texts[0]
        assert "user@example.com" in texts[0]

    def test_unregistered_action_is_not_synthesized(self) -> None:
        executor = _make_executor()  # no registered actions
        msg = MagicMock()
        msg.parts = [create_a2ui_parts([ACTION_ENVELOPE])[0]]
        ctx = MagicMock()
        ctx.message = msg

        executor._rewrite_incoming_a2ui_parts(ctx)

        # No matching A2UIAction → the action is dropped, never turned into a
        # text prompt (the original DataPart is left untouched).
        assert [p.text for p in msg.parts if p.text] == []

    def test_error_envelope_synthesizes_corrective_prompt(self) -> None:
        executor = _make_executor()
        msg = MagicMock()
        msg.parts = [create_a2ui_parts([ERROR_ENVELOPE])[0]]
        ctx = MagicMock()
        ctx.message = msg

        executor._rewrite_incoming_a2ui_parts(ctx)

        texts = [p.text for p in msg.parts if p.text]
        assert len(texts) == 1
        assert "VALIDATION_FAILED" in texts[0]
        assert "/components/0" in texts[0]

    def test_function_response_synthesizes_continuation_prompt(self) -> None:
        executor = _make_executor()
        msg = MagicMock()
        msg.parts = [create_a2ui_parts([FUNCTION_RESPONSE_ENVELOPE])[0]]
        ctx = MagicMock()
        ctx.message = msg

        executor._rewrite_incoming_a2ui_parts(ctx)

        texts = [p.text for p in msg.parts if p.text]
        assert len(texts) == 1
        assert "openUrl" in texts[0]
        assert "fc-1" in texts[0]


@pytest.mark.asyncio
class TestCallFunctionPause:
    """A server callFunction(wantResponse=true) pauses the task awaiting the
    client's functionResponse, delivering the callFunction DataPart on the
    input-required transition instead of completing the task."""

    @staticmethod
    def _run_turn_collaborators(stream: MemoryStream) -> "tuple[MagicMock, Message, ConversationContext]":
        agent_msg = Message(role=Role.ROLE_AGENT, parts=[Part(text="ui")], message_id="m1")
        updater = MagicMock()
        updater.requires_input = AsyncMock()
        updater.complete = AsyncMock()
        updater.new_agent_message = MagicMock(return_value=agent_msg)
        lifecycle_ctx = ConversationContext(stream)
        return updater, agent_msg, lifecycle_ctx

    async def test_call_function_with_want_response_pauses(self) -> None:
        agent = A2UIAgent(
            name="ui_agent",
            config=TestConfig(f"Opening the link.\n<a2ui-json>\n{_CALL_FUNCTION_BLOCK}\n</a2ui-json>"),
            protocol_version="v1.0",
            validate_responses=True,
        )
        executor = A2UIAgentExecutor(agent)
        stream = MemoryStream()
        updater, agent_msg, lifecycle_ctx = self._run_turn_collaborators(stream)

        await executor._run_one_turn(
            ParsedMessage(inputs=[TextInput("open the link")]),
            updater,
            stream,
            lifecycle_ctx,
            text_pieces=[],
            pending_client_calls=[],
            task_id="t1",
            context_id="c1",
        )

        updater.requires_input.assert_awaited_once()
        updater.complete.assert_not_awaited()
        # The callFunction DataPart rides the input-required transition.
        assert updater.requires_input.await_args.kwargs["message"] is agent_msg

    async def test_no_call_function_completes_normally(self) -> None:
        agent = A2UIAgent(
            name="ui_agent",
            config=TestConfig("Just a plain reply."),
            protocol_version="v1.0",
            validate_responses=True,
        )
        executor = A2UIAgentExecutor(agent)
        stream = MemoryStream()
        updater, _agent_msg, lifecycle_ctx = self._run_turn_collaborators(stream)

        await executor._run_one_turn(
            ParsedMessage(inputs=[TextInput("hi")]),
            updater,
            stream,
            lifecycle_ctx,
            text_pieces=[],
            pending_client_calls=[],
            task_id="t1",
            context_id="c1",
        )

        updater.complete.assert_awaited_once()
        updater.requires_input.assert_not_awaited()
