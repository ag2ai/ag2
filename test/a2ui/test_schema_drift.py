# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""The vendored A2UI schemas against the official SDK's (``a2ui-core``), which ag2 does not depend on at runtime.

Annotations (descriptions, titles, examples) and the order of unordered lists are ignored; every other
difference must be listed in ``KNOWN_DRIFT``. A new difference fails the test (upstream moved: update the
vendored file or record why not), and so does a listed one that disappeared (drop it from the list). See ADR 0039.
v0.9.1 is not compared on its own: it reuses the v0.9 files. Only the documents the SDK publishes are compared:
the message schema, the common types and the basic catalog.
"""

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

import ag2.a2ui

a2ui_core = pytest.importorskip("a2ui.core")
sdk_basic_catalog = pytest.importorskip("a2ui.core.basic_catalog")
sdk_schema = pytest.importorskip("a2ui.core.schema")

SPEC_ROOT = Path(ag2.a2ui.__file__).parent

KNOWN_DRIFT: dict[tuple[str, str], set[str]] = {
    # a2ui-project/a2ui#4044 (2026-11-11) renamed the v1.0 component envelope keys to `@...`; the vendored spec has
    # it, `a2ui-core` 0.6.0 predates it. Drop these once a release ships it.
    ("1.0", "common_types"): {
        "ag2-only /$defs/Checkable/properties/@checks",
        "ag2-only /$defs/ComponentCommon/properties/@accessibility",
        "ag2-only /$defs/ComponentCommon/properties/@catalogId",
        "ag2-only /$defs/ComponentCommon/properties/@metadata",
        "sdk-only /$defs/Checkable/properties/checks",
        "sdk-only /$defs/ComponentCommon/properties/accessibility",
        "sdk-only /$defs/ComponentCommon/properties/catalogId",
        "sdk-only /$defs/ComponentCommon/properties/metadata",
    },
    # The same change rewrote the catalog's prompt instructions to the `@` keys.
    ("1.0", "basic_catalog"): {"changed /instructions"},
}

_ANNOTATIONS = frozenset({"description", "title", "examples", "$comment"})
_UNORDERED = frozenset({"enum", "required", "type", "allOf", "anyOf", "oneOf"})
# Maps from a name to a schema: their keys are names (a property may be called `description`), not keywords.
_NAME_MAPS = frozenset({"properties", "patternProperties", "$defs", "definitions", "components", "functions"})


def _normalize(node: Any, key: str | None = None) -> Any:
    if isinstance(node, dict):
        if key in _NAME_MAPS:
            return {name: _normalize(schema) for name, schema in node.items()}
        return {k: _normalize(v, k) for k, v in node.items() if k not in _ANNOTATIONS}
    if isinstance(node, list):
        items = [_normalize(item) for item in node]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True)) if key in _UNORDERED else items
    return node


def _differences(ours: Any, theirs: Any, path: str = "") -> Iterator[str]:
    if isinstance(ours, dict) and isinstance(theirs, dict):
        for key in sorted(ours.keys() | theirs.keys()):
            if key not in ours:
                yield f"sdk-only {path}/{key}"
            elif key not in theirs:
                yield f"ag2-only {path}/{key}"
            else:
                yield from _differences(ours[key], theirs[key], f"{path}/{key}")
    elif isinstance(ours, list) and isinstance(theirs, list) and len(ours) == len(theirs):
        for index, (mine, other) in enumerate(zip(ours, theirs, strict=True)):
            yield from _differences(mine, other, f"{path}[{index}]")
    elif ours != theirs:
        yield f"changed {path}"


def _vendored(version_dir: str, file_name: str) -> Any:
    return json.loads((SPEC_ROOT / version_dir / "spec" / file_name).read_text(encoding="utf-8"))


def _sdk_message_schema(version: str) -> Any:
    return json.loads(a2ui_core.get_agent_to_renderer_schema_json(sdk_schema.ProtocolVersion(version)))


def _sdk_common_types(version: str) -> Any:
    return json.loads(a2ui_core.get_common_types_schema_json(sdk_schema.ProtocolVersion(version)))


def _sdk_basic_catalog(version: str) -> Any:
    return sdk_basic_catalog.BasicCatalog(sdk_schema.ProtocolVersion(version)).to_json()


# (SDK version, document, ag2 version dir, ag2 file, how the SDK publishes it)
CASES: list[tuple[str, str, str, str, Callable[[str], Any]]] = [
    ("v0.9", "server_to_client", "v0_9", "server_to_client.json", _sdk_message_schema),
    ("v0.9", "common_types", "v0_9", "common_types.json", _sdk_common_types),
    ("v0.9", "basic_catalog", "v0_9", "basic_catalog.json", _sdk_basic_catalog),
    ("1.0", "server_to_client", "v1_0", "agent_to_renderer.json", _sdk_message_schema),
    ("1.0", "common_types", "v1_0", "common_types.json", _sdk_common_types),
    ("1.0", "basic_catalog", "v1_0", "basic_catalog.json", _sdk_basic_catalog),
]


@pytest.mark.parametrize(
    ("sdk_version", "document", "version_dir", "file_name", "published"),
    [pytest.param(*case, id=f"{case[0]}-{case[1]}") for case in CASES],
)
def test_vendored_schema_matches_the_sdk(
    sdk_version: str, document: str, version_dir: str, file_name: str, published: Callable[[str], Any]
) -> None:
    ours = _normalize(_vendored(version_dir, file_name))
    theirs = _normalize(published(sdk_version))

    assert set(_differences(ours, theirs)) == KNOWN_DRIFT.get((sdk_version, document), set())
