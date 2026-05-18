# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.a2ui import (
    A2UIIncomingAction,
    A2UIIncomingError,
    parse_incoming_message,
)


class TestParseIncomingAction:
    def test_full_action(self) -> None:
        result = parse_incoming_message({
            "version": "v0.9",
            "action": {
                "name": "submit",
                "surfaceId": "s1",
                "sourceComponentId": "btn1",
                "timestamp": "2026-05-18T12:00:00Z",
                "context": {"email": "a@b.c"},
            },
        })
        assert result.kind == "action"
        assert result.action == A2UIIncomingAction(
            name="submit",
            surface_id="s1",
            source_component_id="btn1",
            timestamp="2026-05-18T12:00:00Z",
            context={"email": "a@b.c"},
        )
        assert result.error is None
        assert result.parse_error is None

    def test_missing_fields_default_to_empty(self) -> None:
        result = parse_incoming_message({"action": {"name": "click"}})
        assert result.kind == "action"
        assert result.action == A2UIIncomingAction(
            name="click",
            surface_id="",
            source_component_id="",
            timestamp="",
            context={},
        )


class TestParseIncomingError:
    def test_validation_failed(self) -> None:
        result = parse_incoming_message({
            "version": "v0.9",
            "error": {
                "code": "VALIDATION_FAILED",
                "surfaceId": "s1",
                "path": "/components/0/text",
                "message": "Expected string, got null.",
            },
        })
        assert result.kind == "error"
        assert result.error == A2UIIncomingError(
            code="VALIDATION_FAILED",
            surface_id="s1",
            message="Expected string, got null.",
            path="/components/0/text",
        )

    def test_generic_error_without_path(self) -> None:
        result = parse_incoming_message({
            "error": {"code": "RUNTIME_ERROR", "surfaceId": "s1", "message": "Oops"},
        })
        assert result.kind == "error"
        assert result.error is not None
        assert result.error.code == "RUNTIME_ERROR"
        assert result.error.path is None


class TestParseIncomingUnknown:
    def test_neither_action_nor_error(self) -> None:
        result = parse_incoming_message({"version": "v0.9", "foo": "bar"})
        assert result.kind == "unknown"
        assert result.parse_error is not None
        assert result.action is None
        assert result.error is None

    def test_non_dict_input(self) -> None:
        result = parse_incoming_message(["not", "a", "dict"])
        assert result.kind == "unknown"
        assert result.parse_error is not None
