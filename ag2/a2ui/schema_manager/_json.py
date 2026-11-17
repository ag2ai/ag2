# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Reading JSON documents and walking JSON Schema trees: helpers every schema-manager module shares."""

import json
from pathlib import Path

from referencing import Resource
from referencing.jsonschema import DRAFT202012

from .._types import JsonObject, JsonSchema, JsonValue

_PACKAGE_DIR = Path(__file__).parent.parent
"""``ag2/a2ui``, under which each protocol version's vendored files live."""


def _as_object(value: "JsonValue | None") -> JsonObject:
    """Narrow a ``JsonValue`` to a JSON object, returning ``{}`` when it is not a dict.

    JSON Schema documents are dynamic ``dict[str, JsonValue]`` trees, so reading a
    nested member (``components`` / ``functions`` / ``$defs`` / ``properties``)
    yields a broad ``JsonValue``. Funnelling those reads through this helper keeps
    the schema-munging code statically typed without scattering ``isinstance``
    checks or ``cast``s at every access.
    """
    return value if isinstance(value, dict) else {}


def _pointer_token(key: str) -> str:
    return key.replace("~", "~0").replace("/", "~1")


def _pointer_key(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _load_json(path: Path) -> JsonValue:
    """Load a JSON file, raising a clear error on a missing or malformed one."""
    try:
        with open(path, encoding="utf-8") as f:
            result: JsonValue = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"A2UI file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"A2UI file {path} is not valid JSON: {e}") from e
    return result


def _load_object(path: Path) -> JsonSchema:
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"A2UI file {path} must be a JSON object")
    return data


def _resource(contents: JsonSchema) -> Resource:
    return Resource.from_contents(contents, default_specification=DRAFT202012)
