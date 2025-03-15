# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from jsonschema.exceptions import _RefResolutionError

from autogen.json_utils import resolve_json_references  # Replace 'your_module' with the actual module name


def test_resolve_json_references_no_refs() -> None:
    schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
    resolved_schema = resolve_json_references(schema)
    assert resolved_schema == schema


def test_resolve_json_references_with_refs() -> None:
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "address": {"$ref": "#/definitions/address"}},
        "definitions": {
            "address": {"type": "object", "properties": {"street": {"type": "string"}, "city": {"type": "string"}}}
        },
    }
    expected_resolved_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "address": {"type": "object", "properties": {"street": {"type": "string"}, "city": {"type": "string"}}},
        },
        "definitions": {
            "address": {"type": "object", "properties": {"street": {"type": "string"}, "city": {"type": "string"}}}
        },
    }
    resolved_schema = resolve_json_references(schema)
    assert resolved_schema == expected_resolved_schema


def test_resolve_json_references_invalid_ref() -> None:
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "address": {"$ref": "#/definitions/non_existent"}},
        "definitions": {
            "address": {"type": "object", "properties": {"street": {"type": "string"}, "city": {"type": "string"}}}
        },
    }
    with pytest.raises(_RefResolutionError, match="Unresolvable JSON pointer: 'definitions/non_existent'"):
        resolve_json_references(schema)
