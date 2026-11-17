# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""The system prompt ``A2UIServer`` sends the model, pinned against golden files.

Regenerate a golden after an intended prompt change with
``A2UI_UPDATE_PROMPTS=1 pytest test/a2ui/test_prompt_seam.py`` and review the diff.
"""

import os
from pathlib import Path
from typing import Any

import pytest

from ag2.a2ui import a2ui_action

from .seam import serve_turn

_GOLDEN_DIR = Path(__file__).parent / "prompts"

_WIDGET_CATALOG = {
    "catalogId": "https://example.com/cat.json",
    "components": {
        "MyWidget": {
            "type": "object",
            "description": "A widget that shows foo.",
            "properties": {"component": {"const": "MyWidget"}, "foo": {"type": "string"}},
            "required": ["component", "foo"],
        },
        "Bare": {
            "type": "object",
            "properties": {"component": {"const": "Bare"}, "bar": {"type": "string"}},
            "required": ["component"],
        },
    },
}

_V1_CATALOG = {
    "catalogId": "https://mycompany.com/cat.json",
    "protocolVersion": "1.0",
    "instructions": "Prefer MyWidget for greetings.",
    "components": {
        "MyWidget": {
            "type": "object",
            "properties": {"component": {"const": "MyWidget"}, "foo": {"type": "string"}},
            "required": ["component", "foo"],
        },
    },
    "functions": {
        "lookup": {
            "type": "object",
            "returnType": "string",
            "allowedCallers": "agentOnly",
            "properties": {"@call": {"const": "lookup"}},
            "required": ["@call"],
        },
    },
}


@a2ui_action(description="Schedule all posts")
def schedule_posts(time: str) -> None: ...


_CASES: dict[str, dict[str, Any]] = {
    "v0_9_default": {"protocol_version": "v0.9"},
    "v0_9_1_default": {"protocol_version": "v0.9.1"},
    "v1_0_default": {"protocol_version": "v1.0"},
    "v0_9_custom": {
        "protocol_version": "v0.9",
        "custom_catalog": _WIDGET_CATALOG,
        "custom_catalog_rules": "Use MyWidget sparingly.",
        "actions": [schedule_posts],
        "include_schema_in_prompt": False,
    },
    "v1_0_custom": {"protocol_version": "v1.0", "custom_catalog": _V1_CATALOG},
    "v1_0_minimal": {
        "protocol_version": "v1.0",
        "include_schema_in_prompt": False,
        "include_rules_in_prompt": False,
        "system_message": "Be brief.",
    },
}

_CAPABILITIES_BODY = {
    "messages": [{"role": "user", "content": "show ui"}],
    "a2uiRendererCapabilities": {
        "1.0": {
            "supportedCatalogIds": [
                "https://example.com/other.json",
                "https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json",
            ],
            "inlineCatalogs": [{"catalogId": "inline", "components": {"Chip": {}}}],
        }
    },
    "a2uiRendererDataModel": {"surfaces": {"s1": {"name": "Ada"}}},
}


def _check_golden(name: str, prompt: tuple[str, ...]) -> None:
    text = "\n\n=====\n\n".join(prompt) + "\n"
    path = _GOLDEN_DIR / f"{name}.txt"
    if os.environ.get("A2UI_UPDATE_PROMPTS"):
        path.parent.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
    assert text == path.read_text(encoding="utf-8")


@pytest.mark.asyncio
@pytest.mark.parametrize("name", list(_CASES))
async def test_system_prompt(name: str) -> None:
    turn = await serve_turn("ok", **_CASES[name])

    _check_golden(name, turn.model_calls[0].prompt)


@pytest.mark.asyncio
async def test_renderer_capabilities_and_data_model_join_the_prompt() -> None:
    turn = await serve_turn("ok", body=_CAPABILITIES_BODY, protocol_version="v1.0", include_schema_in_prompt=False)

    _check_golden("v1_0_capabilities", turn.model_calls[0].prompt)
