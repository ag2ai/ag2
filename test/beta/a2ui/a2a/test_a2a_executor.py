# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

from a2a.types import Part

from autogen.beta.a2ui import A2UIAction, A2UIAgent
from autogen.beta.a2ui.a2a import create_a2ui_parts, get_a2ui_data, is_a2ui_part
from autogen.beta.a2ui.a2a.executor import A2UIAgentExecutor, _extract_a2ui_envelopes
from autogen.beta.a2ui.incoming import A2UIIncomingAction
from autogen.beta.testing import TestConfig

VERSION = "v0.9"
CATALOG = "https://a2ui.org/specification/v0_9/basic_catalog.json"

A2UI_RESPONSE = (
    f'Here is your UI.\n---a2ui_JSON---\n[{{"version": "{VERSION}", "deleteSurface": {{"surfaceId": "s1"}}}}]'
)

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


class TestBuildA2UIMessage:
    """Splitting the final response into text + canonical A2UI DataPart."""

    def test_splits_text_and_a2ui_datapart(self) -> None:
        executor = _make_executor()
        updater = MagicMock()
        executor._build_a2ui_message(updater, A2UI_RESPONSE, {})

        parts = updater.new_agent_message.call_args.kwargs["parts"]
        assert len(parts) == 2
        assert parts[0].text == "Here is your UI."
        assert is_a2ui_part(parts[1])
        assert get_a2ui_data(parts[1]) is not None

    def test_plain_text_single_part(self) -> None:
        executor = _make_executor()
        updater = MagicMock()
        executor._build_a2ui_message(updater, "Just text.", {})

        parts = updater.new_agent_message.call_args.kwargs["parts"]
        assert len(parts) == 1
        assert parts[0].text == "Just text."

    def test_invalid_json_falls_back_to_full_text(self) -> None:
        executor = _make_executor()
        updater = MagicMock()
        executor._build_a2ui_message(updater, "Text\n---a2ui_JSON---\n{not json}", {})

        parts = updater.new_agent_message.call_args.kwargs["parts"]
        assert len(parts) == 1
        assert "---a2ui_JSON---" in parts[0].text

    def test_returns_none_when_no_parts_and_no_variables(self) -> None:
        executor = _make_executor()
        updater = MagicMock()
        assert executor._build_a2ui_message(updater, "", {}) is None
        updater.new_agent_message.assert_not_called()

    def test_context_update_variables_go_to_metadata(self) -> None:
        executor = _make_executor()
        updater = MagicMock()
        executor._build_a2ui_message(updater, "Just text.", {"count": 3})

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

    def test_unregistered_action_returns_none(self) -> None:
        executor = _make_executor()  # no registered actions
        incoming = A2UIIncomingAction(name="submit", surface_id="s1", source_component_id="btn", timestamp="t")
        assert executor._action_to_prompt(incoming) is None

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
