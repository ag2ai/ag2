# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``A2UIServer`` configuration that is rejected at construction, and the vendored v1.0 catalog's own examples."""

import json
import re
from pathlib import Path
from typing import Any

import pytest

import ag2.a2ui
from ag2 import Agent
from ag2.a2ui import A2UIServer
from ag2.a2ui.transports import RestTransport

from .seam import a2ui_reply, serve_turn


def _server(**kwargs: Any) -> A2UIServer:
    return A2UIServer(Agent("ui"), transport=RestTransport(), **kwargs)


def test_unsupported_protocol_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported A2UI protocol version"):
        _server(protocol_version="v0.7")


@pytest.mark.parametrize("version", ["v0.9", "v0.9.1", "v1.0"])
@pytest.mark.parametrize(
    "catalog",
    [{"components": {}}, {"$id": "https://example.com/catalog.json", "components": {}}, {"catalogId": ""}],
    ids=["no-id", "only-dollar-id", "empty-catalog-id"],
)
def test_custom_catalog_is_identified_by_its_catalog_id(version: str, catalog: dict[str, Any]) -> None:
    # v0_9 a2ui_protocol.md L317 / v1_0 L543: "Each catalog is identified by a `catalogId` string".
    with pytest.raises(ValueError, match="Custom catalog must include a non-empty string 'catalogId'"):
        _server(protocol_version=version, custom_catalog={"protocolVersion": "1.0", **catalog})


def test_custom_catalog_dollar_id_must_match_its_catalog_id() -> None:
    # The spec asks for both "set to the same URI"; $refs resolve against $id, so ag2 requires it.
    with pytest.raises(ValueError, match="'\\$id' 'https://example.com/other.json' differs from its 'catalogId'"):
        _server(custom_catalog={"$id": "https://example.com/other.json", "catalogId": _CATALOG_ID, "components": {}})


@pytest.mark.asyncio
@pytest.mark.parametrize("with_dollar_id", [False, True])
async def test_custom_catalog_is_served_under_its_catalog_id(with_dollar_id: bool) -> None:
    catalog = {
        "catalogId": _CATALOG_ID,
        **({"$id": _CATALOG_ID} if with_dollar_id else {}),
        "protocolVersion": "1.0",
        "components": {"Label": _component("Label", text={"type": "string"})},
    }
    messages = [
        {"version": "1.0", "createSurface": {"surfaceId": "s", "catalogId": _CATALOG_ID}},
        {
            "version": "1.0",
            "updateComponents": {"surfaceId": "s", "components": [{"id": "root", "component": "Label", "text": "x"}]},
        },
    ]

    turn = await serve_turn(a2ui_reply(messages), protocol_version="v1.0", custom_catalog=catalog)

    assert turn.feedback == []
    assert turn.ui == messages


@pytest.mark.parametrize(
    ("version", "catalog", "error"),
    [
        pytest.param(
            "v1.0",
            {},
            'has no protocolVersion, so it is written for A2UI 0.9, but this agent serves v1.0: set "protocolVersion": "1.0"',
            id="v1-missing",
        ),
        pytest.param(
            "v0.9",
            {"protocolVersion": "1.0"},
            "declares protocolVersion '1.0', so it is written for A2UI 1.0, but this agent serves v0.9",
            id="v09-declares-1.0",
        ),
        pytest.param("v1.0", {"protocolVersion": "one"}, "'protocolVersion' 'one' is not a version", id="malformed"),
        # v0.9 has no catalog_definition.json, so only the v1.0 schema's semver pattern guards these.
        pytest.param(
            "v0.9", {"protocolVersion": "0.9.x"}, "'protocolVersion' '0.9.x' is not a version", id="bad-patch"
        ),
        pytest.param("v0.9", {"protocolVersion": "0.9-"}, "'protocolVersion' '0.9-' is not a version", id="empty-pre"),
    ],
)
def test_custom_catalog_protocol_version_must_match_the_served_version(
    version: str, catalog: dict[str, Any], error: str
) -> None:
    # v1_0 a2ui_protocol.md L635: "defaults to "0.9" if omitted, required for catalogs targeting 1.0 and beyond".
    with pytest.raises(ValueError, match=re.escape(error)):
        _server(protocol_version=version, custom_catalog={"catalogId": _CATALOG_ID, **catalog})


@pytest.mark.parametrize(
    ("version", "protocol_version"),
    [
        ("v0.9", None),
        ("v0.9", "0.9"),
        ("v0.9", "v0.9"),
        ("v0.9.1", "0.9"),
        ("v0.9.1", "0.9.1"),
        ("v0.9.1", "0.9.1-rc.1+build.5"),
        ("v1.0", "1.0"),
        ("v1.0", "1.0.0"),
    ],
)
def test_custom_catalog_protocol_version_is_matched_on_major_and_minor(
    version: str, protocol_version: str | None
) -> None:
    declared = {"protocolVersion": protocol_version} if protocol_version is not None else {}
    _server(protocol_version=version, custom_catalog={"catalogId": _CATALOG_ID, **declared, "components": {}})


def test_v1_custom_catalog_defs_need_any_component_and_any_function() -> None:
    # v1_0 catalog_definition.json: "$defs" requires "anyComponent" and "anyFunction".
    with pytest.raises(
        ValueError, match=re.escape("at /$defs (catalog_definition.json): 'anyComponent' is a required")
    ):
        _server(
            protocol_version="v1.0",
            custom_catalog={"catalogId": _CATALOG_ID, "protocolVersion": "1.0", "$defs": {"Local": {"type": "string"}}},
        )


def test_v1_basic_catalog_is_a_valid_custom_catalog() -> None:
    catalog_path = Path(ag2.a2ui.__file__).parent / "v1_0" / "spec" / "basic_catalog.json"
    basic = json.loads(catalog_path.read_text(encoding="utf-8"))

    _server(protocol_version="v1.0", custom_catalog={**basic, "$id": _CATALOG_ID, "catalogId": _CATALOG_ID})


def _catalog_with_ref(ref: str) -> dict[str, Any]:
    return {
        "catalogId": "https://example.com/catalog.json",
        "protocolVersion": "1.0",
        "$defs": {
            "anyComponent": {"oneOf": [{"$ref": "#/components/Label"}]},
            "anyFunction": {"not": {}},
            "Local": {"type": "string"},
        },
        "components": {
            "Label": {
                "type": "object",
                "properties": {"component": {"const": "Label"}, "text": {"$ref": ref}},
                "required": ["component", "text"],
            },
        },
    }


@pytest.mark.parametrize(
    "ref",
    [
        "basic_catalog.json#/components/Text",
        "other_catalog.json#/$defs/Shared",
        "https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json#/components/Text",
        "my_common_types.json#/$defs/DynamicString",
        "catalog.json#/components/Text",
        "common_types.json#/$defs/Surface",
        "common_types.json#DynamicString",
        "https://a2ui.org/specification/v1_0/common_types.json#/$defs/DynamicString",
        "https://evil.example/x/common_types.json#/$defs/DynamicString",
        "DynamicString",
        "#/$defs/anyComponent",
        "#/$defs/anyFunction",
        "#/$defs/Missing",
        "#/components/Missing",
        "#/functions/required",
        "#/properties/text",
    ],
)
def test_v1_custom_catalog_rejects_references_the_spec_does_not_allow(ref: str) -> None:
    # a2ui_protocol.md, "Restricted $ref Targets".
    with pytest.raises(ValueError, match=re.escape(f"'{ref}' at /components/Label/properties/text/$ref")):
        _server(protocol_version="v1.0", custom_catalog=_catalog_with_ref(ref))


def _component(name: str, **properties: Any) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"component": {"const": name}, **properties},
        "required": ["component"],
    }


def _function(name: str, **arguments: Any) -> dict[str, Any]:
    return {
        "type": "object",
        "returnType": "string",
        "properties": {
            "@call": {"const": name},
            "args": {"type": "object", "properties": arguments},
        },
        "required": ["@call"],
    }


_CATALOG_ID = "https://example.com/catalog.json"


@pytest.mark.parametrize(
    ("catalog", "error"),
    [
        pytest.param(
            {"components": {"my-widget": _component("my-widget")}},
            "component name 'my-widget' is not a UAX #31 identifier",
            id="component",
        ),
        pytest.param(
            {"components": {"widget/~definition": _component("Widget")}},
            "component name 'widget/~definition' is not a UAX #31 identifier",
            id="escaped-component",
        ),
        pytest.param(
            {"components": {"Widget": _component("Widget", **{"font-size": {"type": "number"}})}},
            "property name 'font-size' of component 'Widget' is not a UAX #31 identifier",
            id="property",
        ),
        pytest.param(
            {"functions": {"look-up": _function("look-up")}},
            "function name 'look-up' is not a UAX #31 identifier",
            id="function",
        ),
        pytest.param(
            {"functions": {"lookup": _function("lookup", **{"2nd": {"type": "string"}})}},
            "argument name '2nd' of function 'lookup' is not a UAX #31 identifier",
            id="argument",
        ),
        pytest.param(
            {"components": {"Surface": _component("Surface")}},
            "component name 'Surface' is reserved",
            id="reserved-surface",
        ),
    ],
)
def test_v1_custom_catalog_names_must_follow_the_entity_naming_rules(catalog: dict[str, Any], error: str) -> None:
    # v1_0 a2ui_protocol.md, "Catalog Entity Naming Rules".
    with pytest.raises(ValueError, match=re.escape(error)):
        _server(protocol_version="v1.0", custom_catalog={"catalogId": _CATALOG_ID, "protocolVersion": "1.0", **catalog})


@pytest.mark.parametrize(
    ("catalog", "error"),
    [
        pytest.param(
            {"components": {"Card": {**_component("Card"), "allowedParents": "Surface"}}},
            "at /components/Card/allowedParents (catalog_definition.json): 'Surface' is not of type 'array'",
            id="parents-not-an-array",
        ),
        pytest.param(
            {"components": {"Card": {**_component("Card"), "allowedChildren": [1]}}},
            "at /components/Card/allowedChildren/0 (catalog_definition.json): 1 is not of type 'string'",
            id="children-not-strings",
        ),
        pytest.param(
            {"functions": {"lookup": {**_function("lookup"), "allowedCallers": "anyone"}}},
            "at /functions/lookup/allowedCallers (catalog_definition.json): 'anyone' is not one of "
            "['rendererOnly', 'agentOnly', 'rendererOrAgent']",
            id="unknown-callers",
        ),
        pytest.param(
            {
                "functions": {
                    "lookup": {**_function("lookup"), "allowedCallers": "agentOnly", "requiresUserActivation": True}
                }
            },
            "at /functions/lookup/allowedCallers (catalog_definition.json): 'agentOnly' is not one of ['rendererOnly']",
            id="activation-not-renderer-only",
        ),
    ],
)
def test_v1_custom_catalog_constraints_follow_the_catalog_definition(catalog: dict[str, Any], error: str) -> None:
    # v1_0 catalog_definition.json; upstream composition_constraints.json and function_definitions.json.
    with pytest.raises(ValueError, match=re.escape(error)):
        _server(protocol_version="v1.0", custom_catalog={"catalogId": _CATALOG_ID, "protocolVersion": "1.0", **catalog})


def test_v1_custom_catalog_component_const_must_match_its_key() -> None:
    # v1_0 a2ui_protocol.md, "Catalog Schema Rules", 4. Component Discriminator Rule: the `component` const must be
    # "matching the key under which it is defined".
    catalog = {"catalogId": _CATALOG_ID, "protocolVersion": "1.0", "components": {"Gauge": _component("Meter")}}

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid custom catalog: component 'Gauge' declares \"component\": 'Meter'; it must be 'Gauge'"
        ),
    ):
        _server(protocol_version="v1.0", custom_catalog=catalog)


def test_v09_custom_catalog_names_are_not_restricted() -> None:
    # The v0.9 spec sets no naming rule for catalog entities.
    _server(custom_catalog={"catalogId": _CATALOG_ID, "components": {"my-widget": _component("my-widget")}})


def _v1_catalog_instruction_examples() -> list[Any]:
    catalog_path = Path(ag2.a2ui.__file__).parent / "v1_0" / "spec" / "basic_catalog.json"
    instructions = json.loads(catalog_path.read_text(encoding="utf-8"))["instructions"]
    return [json.loads(example) for example in re.findall(r"```json\n(.*?)\n```", instructions, flags=re.DOTALL)]


@pytest.mark.asyncio
@pytest.mark.parametrize("example", _v1_catalog_instruction_examples())
async def test_v1_basic_catalog_instruction_examples_validate(example: Any) -> None:
    messages = example if isinstance(example, list) else [example]

    turn = await serve_turn(a2ui_reply(messages), protocol_version="v1.0")

    assert turn.feedback == []
    assert turn.ui == messages


def test_retry_count_is_not_checked_when_validation_is_off() -> None:
    _server(validate_responses=False, validation_retries=-1)
