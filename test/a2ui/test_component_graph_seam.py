# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""The component-tree and data-path checks, seen through the retry feedback ``A2UIServer`` sends the model."""

from typing import Any

import pytest

from .seam import BASIC_CATALOG, WIRE_VERSION, a2ui_reply, serve_turn

VERSIONS = ["v0.9", "v0.9.1", "v1.0"]


def _create(version: str, surface_id: str = "s", catalog_id: str | None = None) -> dict[str, Any]:
    return {
        "version": WIRE_VERSION[version],
        "createSurface": {"surfaceId": surface_id, "catalogId": catalog_id or BASIC_CATALOG[version]},
    }


def _update(version: str, *components: dict[str, Any], surface_id: str = "s") -> dict[str, Any]:
    return {
        "version": WIRE_VERSION[version],
        "updateComponents": {"surfaceId": surface_id, "components": list(components)},
    }


def _text(component_id: str) -> dict[str, Any]:
    return {"id": component_id, "component": "Text", "text": "hi"}


@pytest.mark.asyncio
@pytest.mark.parametrize("version", VERSIONS)
class TestNewSurface:
    async def test_every_kind_of_child_reference_counts_towards_reachability(self, version: str) -> None:
        messages = [
            _create(version),
            _update(
                version,
                {"id": "root", "component": "Column", "children": ["tabs", "modal", "list"]},
                {"id": "tabs", "component": "Tabs", "tabs": [{"title": "One", "child": "one"}]},
                {"id": "modal", "component": "Modal", "trigger": "open", "content": "body"},
                {"id": "open", "component": "Button", "child": "open_label", "action": {"event": {"name": "go"}}},
                {"id": "list", "component": "List", "children": {"componentId": "row", "path": "/items"}},
                _text("one"),
                _text("body"),
                _text("open_label"),
                _text("row"),
            ),
        ]

        turn = await serve_turn(a2ui_reply(messages), protocol_version=version)

        assert turn.feedback == []
        assert turn.ui == messages

    async def test_one_retry_lists_every_graph_error(self, version: str) -> None:
        messages = [
            _create(version),
            _update(
                version,
                {"id": "root", "component": "Card", "child": "missing"},
                _text("orphan"),
                _text("orphan"),
            ),
        ]

        turn = await serve_turn(a2ui_reply(messages), "Plain text.", protocol_version=version)

        assert len(turn.feedback) == 1
        for error in (
            "Surface 's': duplicate component id 'orphan'",
            "Surface 's': component 'root' references missing component 'missing'",
            "Surface 's': component 'orphan' is not reachable from 'root'",
        ):
            assert f"- {error}\n" in turn.feedback[0]

    async def test_surfaces_are_checked_separately(self, version: str) -> None:
        messages = [
            _create(version, "a"),
            _update(version, {"id": "root", "component": "Card", "child": "x"}, _text("x"), surface_id="a"),
            _create(version, "b"),
            _update(version, {"id": "root", "component": "Card", "child": "x"}, _text("x"), surface_id="b"),
        ]

        turn = await serve_turn(a2ui_reply(messages), protocol_version=version)

        assert turn.feedback == []


@pytest.mark.asyncio
@pytest.mark.parametrize("version", VERSIONS)
class TestEarlierSurface:
    async def test_partial_update_may_reference_earlier_components(self, version: str) -> None:
        messages = [_update(version, {"id": "root", "component": "Card", "child": "from_last_turn"}, _text("loose"))]

        turn = await serve_turn(a2ui_reply(messages), protocol_version=version)

        assert turn.feedback == []
        assert turn.ui == messages

    async def test_partial_update_still_rejects_a_cycle(self, version: str) -> None:
        messages = [
            _update(
                version,
                {"id": "root", "component": "Card", "child": "col"},
                {"id": "col", "component": "Column", "children": ["root"]},
            )
        ]

        turn = await serve_turn(a2ui_reply(messages), "Plain text.", protocol_version=version)

        assert len(turn.feedback) == 1
        assert "- Surface 's': components form a cycle: root -> col -> root\n" in turn.feedback[0]

    async def test_partial_update_still_rejects_duplicate_ids(self, version: str) -> None:
        messages = [_update(version, _text("root"), _text("root"))]

        turn = await serve_turn(a2ui_reply(messages), "Plain text.", protocol_version=version)

        assert len(turn.feedback) == 1
        assert "- Surface 's': duplicate component id 'root'\n" in turn.feedback[0]


@pytest.mark.asyncio
async def test_custom_component_references_come_from_its_schema() -> None:
    catalog = {
        "catalogId": "https://example.com/cat.json",
        "protocolVersion": "1.0",
        "components": {
            "Panel": {
                "type": "object",
                "properties": {
                    "component": {"const": "Panel"},
                    "header": {"$ref": "common_types.json#/$defs/ComponentId"},
                    "body": {"$ref": "common_types.json#/$defs/ChildList"},
                },
                "required": ["component", "header", "body"],
            }
        },
    }
    # Mixable catalogs: the custom Panel names its catalog on a basic surface (a2ui_protocol.md L511).
    messages = [
        _create("v1.0"),
        _update(
            "v1.0",
            {
                "id": "root",
                "component": "Panel",
                "@catalogId": "https://example.com/cat.json",
                "header": "title",
                "body": ["line"],
            },
            _text("title"),
            _text("line"),
        ),
    ]

    turn = await serve_turn(a2ui_reply(messages), protocol_version="v1.0", custom_catalog=catalog)

    assert turn.feedback == []


@pytest.mark.asyncio
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("path", ["", "/", "/a~0b/c~1d", "/items/0"])
async def test_data_model_path_may_be_any_json_pointer(version: str, path: str) -> None:
    messages = [{"version": WIRE_VERSION[version], "updateDataModel": {"surfaceId": "s", "path": path, "value": 1}}]

    turn = await serve_turn(a2ui_reply(messages), protocol_version=version)

    assert turn.feedback == []


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["a/b", "/a~2", "/a~", "~0"])
async def test_data_model_path_must_be_a_json_pointer(path: str) -> None:
    messages = [{"version": "v0.9", "updateDataModel": {"surfaceId": "s", "path": path, "value": 1}}]

    turn = await serve_turn(a2ui_reply(messages), "Plain text.")

    assert len(turn.feedback) == 1
    assert f"- Operation 0: {path!r} is not a JSON Pointer (at /updateDataModel/path)\n" in turn.feedback[0]


@pytest.mark.asyncio
async def test_nothing_is_rejected_when_validation_is_off() -> None:
    messages = [
        _create("v0.9"),
        _update("v0.9", {"id": "root", "component": "Card", "child": "missing"}, _text("orphan")),
        {"version": "v0.9", "updateDataModel": {"surfaceId": "s", "path": "a/b", "value": 1}},
    ]

    turn = await serve_turn(a2ui_reply(messages), validate_responses=False)

    assert turn.feedback == []
    assert turn.ui == messages


@pytest.mark.asyncio
@pytest.mark.parametrize("version", VERSIONS)
async def test_a_later_message_may_update_a_component_from_an_earlier_one(version: str) -> None:
    messages = [
        _create(version),
        _update(version, {"id": "root", "component": "Card", "child": "x"}, _text("x")),
        _update(version, {"id": "x", "component": "Text", "text": "updated"}),
    ]

    turn = await serve_turn(a2ui_reply(messages), protocol_version=version)

    assert turn.feedback == []


@pytest.mark.asyncio
@pytest.mark.parametrize("version", VERSIONS)
async def test_a_new_surface_needs_a_root_component(version: str) -> None:
    messages = [_create(version), _update(version, _text("t"))]

    turn = await serve_turn(a2ui_reply(messages), "Plain text.", protocol_version=version)

    assert len(turn.feedback) == 1
    assert "- No component with id 'root' found on surface 's'" in turn.feedback[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("version", VERSIONS)
async def test_a_partial_update_to_an_earlier_surface_needs_no_root(version: str) -> None:
    messages = [_update(version, _text("t"))]

    turn = await serve_turn(a2ui_reply(messages), protocol_version=version)

    assert turn.feedback == []
    assert turn.ui == messages


def _panel_catalog(child_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "catalogId": "https://example.com/cat.json",
        "protocolVersion": "1.0",
        "$defs": {
            "anyComponent": {"oneOf": [{"$ref": "#/components/Panel"}]},
            "anyFunction": {"not": {}},
            "Kid": {"$ref": "common_types.json#/$defs/ComponentId"},
        },
        "components": {
            "Panel": {
                "type": "object",
                "properties": {"component": {"const": "Panel"}, "child": child_schema},
                "required": ["component", "child"],
            }
        },
    }


@pytest.mark.asyncio
async def test_custom_child_references_are_found_through_a_local_definition() -> None:
    messages = [
        _create("v1.0"),
        _update(
            "v1.0",
            {"id": "root", "component": "Panel", "@catalogId": "https://example.com/cat.json", "child": "t"},
            _text("t"),
        ),
    ]

    turn = await serve_turn(
        a2ui_reply(messages), protocol_version="v1.0", custom_catalog=_panel_catalog({"$ref": "#/$defs/Kid"})
    )

    assert turn.feedback == []


@pytest.mark.asyncio
async def test_a_plain_string_property_is_static_text_even_when_named_child() -> None:
    # The spec: a raw string type for an id is treated as static text, whatever the property is called.
    messages = [
        _create("v1.0", catalog_id="https://example.com/cat.json"),
        _update("v1.0", {"id": "root", "component": "Panel", "child": "not a component id"}),
    ]

    turn = await serve_turn(
        a2ui_reply(messages), protocol_version="v1.0", custom_catalog=_panel_catalog({"type": "string"})
    )

    assert turn.feedback == []


@pytest.mark.asyncio
async def test_an_inline_template_schema_is_a_child_reference() -> None:
    template = {"type": "object", "properties": {"componentId": {"type": "string"}, "path": {"type": "string"}}}
    messages = [
        _create("v1.0", catalog_id="https://example.com/cat.json"),
        _update("v1.0", {"id": "root", "component": "Panel", "child": {"componentId": "missing", "path": "/items"}}),
    ]

    turn = await serve_turn(
        a2ui_reply(messages), "Plain text.", protocol_version="v1.0", custom_catalog=_panel_catalog(template)
    )

    assert len(turn.feedback) == 1
    assert "- Surface 's': component 'root' references missing component 'missing'\n" in turn.feedback[0]


@pytest.mark.asyncio
async def test_schema_and_graph_errors_are_reported_in_one_retry() -> None:
    messages = [_create("v0.9"), _update("v0.9", {"id": "root", "component": "Column", "children": "a"}, _text("a"))]

    turn = await serve_turn(a2ui_reply(messages), "Plain text.")

    assert len(turn.feedback) == 1
    assert "not of type 'array'" in turn.feedback[0]
    assert "- Surface 's': component 'a' is not reachable from 'root'\n" in turn.feedback[0]


def _delete(version: str, surface_id: str = "s") -> dict[str, Any]:
    return {"version": WIRE_VERSION[version], "deleteSurface": {"surfaceId": surface_id}}


@pytest.mark.asyncio
@pytest.mark.parametrize("version", VERSIONS)
class TestRecreatedSurface:
    # "to reconfigure them, the surface must be deleted and recreated" (a2ui_protocol.md, createSurface): the new
    # surface starts empty, so its tree is checked on its own.
    async def test_is_checked_without_the_deleted_surfaces_components(self, version: str) -> None:
        messages = [
            _create(version),
            _update(version, {"id": "root", "component": "Column", "children": ["old"]}, _text("old")),
            _delete(version),
            _create(version),
            _update(version, _text("root")),
        ]

        turn = await serve_turn(a2ui_reply(messages), "Plain text.", protocol_version=version)

        assert turn.feedback == []
        assert turn.ui == messages

    async def test_needs_its_own_root(self, version: str) -> None:
        messages = [
            _create(version),
            _update(version, _text("root")),
            _delete(version),
            _create(version),
            _update(version, {"id": "card", "component": "Card", "child": "label"}, _text("label")),
        ]

        turn = await serve_turn(a2ui_reply(messages), "Plain text.", protocol_version=version)

        assert turn.errors == [
            [
                "- No component with id 'root' found on surface 's' — the A2UI spec requires the component tree to "
                "have a 'root' node.",
            ]
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("version", "hint"),
    [
        ("v0.9", "send deleteSurface before creating it again, or use a new surfaceId"),
        ("v0.9.1", "send deleteSurface before creating it again"),
        ("v1.0", "send deleteSurface before creating it again, or use a new surfaceId"),
    ],
)
async def test_creating_an_existing_surface_names_the_versions_remedy(version: str, hint: str) -> None:
    # v0.9 and v1.0: "`surfaceId` must be globally unique for the renderer's lifetime"; v0.9.1 relaxed that to the
    # active surfaces (v0_9_1 evolution_guide.md, "Surface ID Uniqueness Relaxation").
    messages = [_create(version), _update(version, _text("root")), _create(version)]

    turn = await serve_turn(a2ui_reply(messages), "Plain text.", protocol_version=version)

    assert turn.errors == [[f"- Operation 2: surface 's' already exists (operation 0 created it); {hint}"]]


_COLUMN_CATALOG_ID = "https://example.com/columns.json"
_COLUMN_CATALOG = {
    "catalogId": _COLUMN_CATALOG_ID,
    "protocolVersion": "1.0",
    "components": {
        "Column": {
            "type": "object",
            "properties": {
                "component": {"const": "Column"},
                "children": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["component"],
        }
    },
}


@pytest.mark.asyncio
class TestSameTypeInTwoCatalogs:
    # A component resolves against its own `@catalogId` or its surface's `catalogId`, with no fallback (v1.0
    # a2ui_protocol.md, "Mixable catalogs and component resolution logic"); a custom `Column` is another type.
    async def test_a_basic_surface_uses_the_basic_definition(self) -> None:
        messages = [
            _create("v1.0"),
            _update("v1.0", {"id": "root", "component": "Column", "children": ["t"]}, _text("t")),
        ]

        turn = await serve_turn(a2ui_reply(messages), protocol_version="v1.0", custom_catalog=_COLUMN_CATALOG)

        assert turn.feedback == []
        assert turn.ui == messages

    async def test_a_basic_surface_reports_a_dangling_basic_reference(self) -> None:
        messages = [_create("v1.0"), _update("v1.0", {"id": "root", "component": "Column", "children": ["missing"]})]

        turn = await serve_turn(
            a2ui_reply(messages), "Plain text.", protocol_version="v1.0", custom_catalog=_COLUMN_CATALOG
        )

        assert turn.errors == [["- Surface 's': component 'root' references missing component 'missing'"]]

    async def test_a_custom_surface_uses_the_custom_definition(self) -> None:
        # The custom `children` is a plain string list, not a ChildList: its items are not component references.
        messages = [
            _create("v1.0", catalog_id=_COLUMN_CATALOG_ID),
            _update("v1.0", {"id": "root", "component": "Column", "children": ["just text"]}),
        ]

        turn = await serve_turn(a2ui_reply(messages), protocol_version="v1.0", custom_catalog=_COLUMN_CATALOG)

        assert turn.feedback == []
        assert turn.ui == messages
