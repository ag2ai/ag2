# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2.a2ui import _types
from ag2.a2ui._types import A2UIVersion
from ag2.a2ui.diagnostics import A2UIIssue, MessageChecker
from ag2.a2ui.schema_manager import A2UISchemaManager

VERSIONS: list[A2UIVersion] = ["v0.9", "v0.9.1", "v1.0"]


def _checker(version: A2UIVersion, **manager_kwargs: object) -> MessageChecker:
    manager = A2UISchemaManager(version, **manager_kwargs)  # type: ignore[arg-type]
    return MessageChecker(manager.server_to_client_schema, manager.build_schema_registry())


def _render(checker: MessageChecker, message: object) -> list[str]:
    return [issue.render(message) for issue in checker.issues(message)]


def _components(version: A2UIVersion, *components: dict) -> dict:
    return {"version": version, "updateComponents": {"surfaceId": "s", "components": list(components)}}


@pytest.mark.parametrize("version", VERSIONS)
class TestComponentIssues:
    def test_unknown_property_is_named_on_its_component(self, version: A2UIVersion) -> None:
        slider = {"id": "b", "component": "Slider", "min": 1, "max": 2, "step": 1, "value": 0}

        assert _render(_checker(version), _components(version, slider)) == [
            "Component 'b' (Slider): Unevaluated properties are not allowed ('step' was unexpected)"
        ]

    def test_unknown_component_type(self, version: A2UIVersion) -> None:
        message = _components(version, {"id": "root", "component": "Nope"})

        assert _render(_checker(version), message) == ["Component 'root' (Nope): unknown component 'Nope'"]

    def test_missing_property_is_not_buried_in_unevaluated_noise(self, version: A2UIVersion) -> None:
        # A failing branch records no annotations, so ``child`` would otherwise be reported as unexpected.
        message = _components(version, {"id": "root", "component": "Button", "child": "label"})

        assert _render(_checker(version), message) == ["Component 'root' (Button): 'action' is a required property"]

    def test_valid_message_has_no_issues(self, version: A2UIVersion) -> None:
        message = _components(version, {"id": "root", "component": "Text", "text": "hi"})

        assert _checker(version).issues(message) == []


@pytest.mark.parametrize("version", VERSIONS)
class TestEnvelopeIssues:
    def test_missing_payload_property_is_located(self, version: A2UIVersion) -> None:
        message = {"version": version, "deleteSurface": {}}

        assert _render(_checker(version), message) == ["'surfaceId' is a required property (at /deleteSurface)"]

    def test_message_without_a_payload_key(self, version: A2UIVersion) -> None:
        [issue] = _render(_checker(version), {"version": version, "bogus": {}})

        assert issue.startswith("Not an A2UI message: expected an object with exactly one of [")

    def test_non_object(self, version: A2UIVersion) -> None:
        assert _render(_checker(version), [1]) == ["Expected operation to be an object, got list"]


class TestV1Issues:
    def test_missing_update_data_model_value(self) -> None:
        message = {"version": "v1.0", "updateDataModel": {"surfaceId": "s", "path": "/a"}}

        assert _render(_checker("v1.0"), message) == ["'value' is a required property (at /updateDataModel)"]

    def test_unknown_function_names_the_tag(self) -> None:
        message = {
            "version": "v1.0",
            "callRendererFunction": {
                "functionCallId": "f",
                "callFunction": {"@call": "nope", "catalogId": "https://a2ui.org/c.json"},
            },
        }

        assert _render(_checker("v1.0"), message) == ["unknown @call 'nope' (at /callRendererFunction/callFunction)"]

    def test_function_argument_error_is_located(self) -> None:
        message = {
            "version": "v1.0",
            "callRendererFunction": {
                "functionCallId": "f",
                "callFunction": {"@call": "openUrl", "catalogId": "https://a2ui.org/c.json", "args": {}},
            },
        }

        assert _render(_checker("v1.0"), message) == [
            "'url' is a required property (at /callRendererFunction/callFunction/args)"
        ]

    def test_every_payload_key_has_a_definition(self) -> None:
        checker = _checker("v1.0")
        [issue] = checker.issues({"version": "v1.0", "bogus": 1})

        for key in _types.server_to_client_payload_keys():
            assert f"'{key}'" in issue.message


def test_custom_catalog_component_is_checked_against_its_own_schema() -> None:
    catalog = {
        "$id": "https://example.com/cat.json",
        "components": {
            "MyWidget": {
                "type": "object",
                "properties": {"component": {"const": "MyWidget"}, "foo": {"type": "string"}},
                "required": ["component", "foo"],
            }
        },
    }
    checker = _checker("v1.0", custom_catalog=catalog)

    assert _render(checker, _components("v1.0", {"id": "root", "component": "MyWidget"})) == [
        "Component 'root' (MyWidget): 'foo' is a required property"
    ]


def test_issue_without_a_component_reports_its_location() -> None:
    issue = A2UIIssue(("createSurface", "surfaceId"), "5 is not of type 'string'")

    assert (
        issue.render({"createSurface": {"surfaceId": 5}}) == "5 is not of type 'string' (at /createSurface/surfaceId)"
    )
