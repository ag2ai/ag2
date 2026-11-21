# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""The vendored A2UI files of one protocol version, read once per process."""

from dataclasses import dataclass
from functools import cache

from .._types import JsonSchema, JsonValue
from .._versions import A2UIProtocol
from ._json import _PACKAGE_DIR, _as_object, _load_json, _load_object


def _instructions_of(catalog: JsonSchema) -> str:
    """Return a catalog's embedded ``instructions`` text (v1.0), or ``""``."""
    instructions = catalog.get("instructions")
    return instructions.strip() if isinstance(instructions, str) else ""


def _accept_wire_versions(schema: JsonValue, versions: tuple[str, ...]) -> JsonValue:
    """Widen every message's ``version: {const}`` to ``{enum: versions}``.

    v0.9.1 reuses the v0.9 spec files, whose messages pin ``"v0.9"``; v0.9.1 also accepts ``"v0.9.1"``.
    """
    if isinstance(schema, list):
        return [_accept_wire_versions(item, versions) for item in schema]
    if not isinstance(schema, dict):
        return schema
    result = {key: _accept_wire_versions(value, versions) for key, value in schema.items()}
    version = _as_object(_as_object(result.get("properties")).get("version"))
    if "const" in version:
        widened = {key: value for key, value in version.items() if key != "const"}
        widened["enum"] = list(versions)
        result["properties"] = {**_as_object(result["properties"]), "version": widened}
    return result


@dataclass(frozen=True, slots=True)
class _VendoredSpec:
    """One protocol version's vendored files, read once per process; each schema manager works on a deep copy."""

    server_to_client: JsonSchema
    common_types: JsonSchema
    basic_catalog: JsonSchema
    catalog_rules: str
    prompt_example: JsonValue
    prompt_message_types: str


@cache
def _vendored_spec(protocol: A2UIProtocol) -> _VendoredSpec:
    version_dir = _PACKAGE_DIR / protocol.package_dir
    spec_dir = version_dir / "spec"
    server_to_client = _load_object(spec_dir / protocol.server_to_client_file)
    if len(protocol.accepted_wire_versions) > 1:
        server_to_client = _as_object(_accept_wire_versions(server_to_client, protocol.accepted_wire_versions))
    basic_catalog = _load_object(spec_dir / "basic_catalog.json")
    rules = (
        (spec_dir / protocol.rules_file).read_text(encoding="utf-8").strip()
        if protocol.rules_file
        else _instructions_of(basic_catalog)
    )
    return _VendoredSpec(
        server_to_client=server_to_client,
        common_types=_load_object(spec_dir / "common_types.json"),
        basic_catalog=basic_catalog,
        catalog_rules=rules,
        prompt_example=_load_json(version_dir / "prompt_example.json"),
        prompt_message_types=(version_dir / "prompt_message_types.md").read_text(encoding="utf-8").strip(),
    )
