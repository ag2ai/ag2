# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2.a2ui import A2UIClientCapabilities
from ag2.a2ui.actions import A2UIEventAction
from ag2.a2ui.request import parse_request
from ag2.events import ModelRequest, ModelResponse


def _no_actions(name: str) -> None:
    return None


class TestParseRequestCapabilities:
    def test_absent_capabilities_is_none(self) -> None:
        req = parse_request({"messages": []}, resolve_action=_no_actions)
        assert req.client_capabilities is None

    def test_capabilities_parsed_for_version(self) -> None:
        body = {
            "messages": [],
            "a2uiClientCapabilities": {"v1.0": {"supportedCatalogIds": ["https://x.example/c.json"]}},
        }
        req = parse_request(body, resolve_action=_no_actions, version_key="v1.0")
        assert req.client_capabilities == A2UIClientCapabilities(
            supported_catalog_ids=["https://x.example/c.json"], inline_catalogs=[]
        )

    def test_capabilities_version_mismatch_is_none(self) -> None:
        body = {"messages": [], "a2uiClientCapabilities": {"v1.0": {"supportedCatalogIds": ["x"]}}}
        req = parse_request(body, resolve_action=_no_actions)  # default version_key "v0.9"
        assert req.client_capabilities is None


class TestParseRequestShape:
    def test_rejects_non_object_body(self) -> None:
        with pytest.raises(ValueError, match="JSON object"):
            parse_request("[1, 2, 3]", resolve_action=_no_actions)

    def test_rejects_invalid_json(self) -> None:
        with pytest.raises(ValueError, match="not valid JSON"):
            parse_request("{not json", resolve_action=_no_actions)

    def test_rejects_non_list_messages(self) -> None:
        with pytest.raises(ValueError, match="'messages' must be a list"):
            parse_request({"messages": {}}, resolve_action=_no_actions)

    def test_rejects_non_dict_variables(self) -> None:
        with pytest.raises(ValueError, match="'variables' must be an object"):
            parse_request({"messages": [], "variables": []}, resolve_action=_no_actions)

    def test_rejects_non_list_a2ui(self) -> None:
        with pytest.raises(ValueError, match="'a2ui' must be a list"):
            parse_request({"messages": [], "a2ui": {}}, resolve_action=_no_actions)

    def test_rejects_unknown_role(self) -> None:
        with pytest.raises(ValueError, match="unsupported message role"):
            parse_request({"messages": [{"role": "ghost", "content": "x"}]}, resolve_action=_no_actions)

    def test_rejects_non_string_content(self) -> None:
        with pytest.raises(ValueError, match="must be a string"):
            parse_request({"messages": [{"role": "user", "content": 5}]}, resolve_action=_no_actions)

    def test_accepts_bytes_body(self) -> None:
        req = parse_request(b'{"messages": [{"role": "user", "content": "hi"}]}', resolve_action=_no_actions)
        assert [p.content for p in req.current_inputs] == ["hi"]


class TestParseRequestMapping:
    def test_trailing_user_run_is_current_turn(self) -> None:
        req = parse_request(
            {"messages": [{"role": "user", "content": "show a form"}]},
            resolve_action=_no_actions,
        )
        assert req.history == []
        assert [type(i).__name__ for i in req.current_inputs] == ["TextInput"]
        assert req.current_inputs[0].content == "show a form"

    def test_prior_turns_become_history(self) -> None:
        req = parse_request(
            {
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "answer"},
                    {"role": "user", "content": "second"},
                ]
            },
            resolve_action=_no_actions,
        )
        assert isinstance(req.history[0], ModelRequest)
        assert isinstance(req.history[1], ModelResponse)
        assert [i.content for i in req.current_inputs] == ["second"]

    def test_system_and_developer_become_prompt(self) -> None:
        req = parse_request(
            {
                "messages": [
                    {"role": "system", "content": "be terse"},
                    {"role": "developer", "content": "use v0.9"},
                    {"role": "user", "content": "go"},
                ]
            },
            resolve_action=_no_actions,
        )
        assert req.prompt == ["be terse", "use v0.9"]

    def test_variables_passthrough(self) -> None:
        req = parse_request(
            {"messages": [], "variables": {"locale": "en"}},
            resolve_action=_no_actions,
        )
        assert req.variables == {"locale": "en"}


class TestParseRequestActions:
    def test_registered_action_is_skipped_for_server_handling(self) -> None:
        # A registered action is handled on the server (its handler runs in the
        # turn core), so it is NOT rewritten into a prompt for the agent.
        actions = {"confirm": A2UIEventAction(name="confirm", description="Confirm it")}
        req = parse_request(
            {
                "messages": [],
                "a2ui": [
                    {
                        "version": "v0.9",
                        "action": {
                            "name": "confirm",
                            "surfaceId": "s1",
                            "sourceComponentId": "btn",
                            "timestamp": "2026-06-15T00:00:00Z",
                            "context": {"id": 1},
                        },
                    }
                ],
            },
            resolve_action=actions.get,
        )
        assert req.current_inputs == []

    def test_unregistered_action_becomes_generic_prompt(self) -> None:
        # An unmatched click no longer drops: buttons are rendered dynamically by
        # the LLM, so the click is rewritten generically and the LLM continues.
        req = parse_request(
            {
                "messages": [],
                "a2ui": [
                    {
                        "version": "v0.9",
                        "action": {"name": "ghost", "surfaceId": "", "sourceComponentId": "", "timestamp": ""},
                    }
                ],
            },
            resolve_action=_no_actions,
        )
        assert len(req.current_inputs) == 1
        assert "ghost" in req.current_inputs[0].content

    def test_error_envelope_becomes_corrective_prompt(self) -> None:
        req = parse_request(
            {
                "messages": [],
                "a2ui": [
                    {
                        "version": "v0.9",
                        "error": {"code": "VALIDATION_FAILED", "surfaceId": "s1", "message": "bad", "path": "/x"},
                    }
                ],
            },
            resolve_action=_no_actions,
        )
        assert len(req.current_inputs) == 1
        assert "error" in req.current_inputs[0].content.lower()
        assert "VALIDATION_FAILED" in req.current_inputs[0].content

    def test_unregistered_action_appended_after_trailing_user_text(self) -> None:
        # An unregistered click becomes a generic prompt appended after the
        # trailing user message (a registered action would be skipped instead).
        req = parse_request(
            {
                "messages": [{"role": "user", "content": "and also"}],
                "a2ui": [
                    {
                        "version": "v0.9",
                        "action": {"name": "go", "surfaceId": "", "sourceComponentId": "", "timestamp": ""},
                    }
                ],
            },
            resolve_action=_no_actions,
        )
        assert [i.content for i in req.current_inputs][0] == "and also"
        assert len(req.current_inputs) == 2

    def test_function_response_envelope_becomes_continuation_prompt(self) -> None:
        req = parse_request(
            {
                "messages": [],
                "a2ui": [
                    {
                        "version": "v1.0",
                        "functionResponse": {
                            "functionCallId": "fc-1",
                            "call": "getScreenResolution",
                            "value": [1920, 1080],
                        },
                    }
                ],
            },
            resolve_action=_no_actions,
        )
        assert len(req.current_inputs) == 1
        content = req.current_inputs[0].content
        assert "getScreenResolution" in content
        assert "fc-1" in content

    def test_function_response_without_call_id_is_dropped(self) -> None:
        req = parse_request(
            {
                "messages": [],
                "a2ui": [{"version": "v1.0", "functionResponse": {"call": "openUrl", "value": True}}],
            },
            resolve_action=_no_actions,
        )
        assert req.current_inputs == []
