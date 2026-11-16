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


def test_custom_catalog_without_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="Custom catalog must include"):
        _server(custom_catalog={"components": {}})


def _catalog_with_ref(ref: str) -> dict[str, Any]:
    return {
        "$id": "https://example.com/catalog.json",
        "$defs": {"Local": {"type": "string"}},
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
