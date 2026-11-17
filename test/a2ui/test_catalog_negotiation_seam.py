# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Per-turn catalog negotiation at the ``A2UIServer`` seam.

As the official SDK's ``resolve_catalogs`` does, a renderer that lists ``supportedCatalogIds`` is offered only the
agent's catalogs it names: the prompt and the validator both narrow to them. With none in common the turn runs
without A2UI, and nothing A2UI reaches the renderer.
"""

import logging
from typing import Any

import pytest

from ag2.a2ui import a2ui_action

from .seam import BASIC_CATALOG, a2ui_reply, serve_turn

_CUSTOM_ID = "https://example.com/cat.json"
_CUSTOM = {
    "$id": _CUSTOM_ID,
    "components": {
        "MyWidget": {
            "type": "object",
            "properties": {"component": {"const": "MyWidget"}, "foo": {"type": "string"}},
            "required": ["component", "foo"],
        },
    },
}
_BASIC_ID = BASIC_CATALOG["v1.0"]
_WIDGET = {"id": "root", "component": "MyWidget", "foo": "x"}
_TEXT = {"id": "root", "component": "Text", "text": "x"}


def _body(*supported: str) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": "show ui"}],
        "a2uiRendererCapabilities": {"1.0": {"supportedCatalogIds": list(supported)}},
    }


def _surface(catalog_id: str, component: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"version": "1.0", "createSurface": {"surfaceId": "s", "catalogId": catalog_id}},
        {"version": "1.0", "updateComponents": {"surfaceId": "s", "components": [component]}},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("supported", "offered", "withheld", "component"),
    [
        (_CUSTOM_ID, _CUSTOM_ID, _BASIC_ID, _WIDGET),
        (_BASIC_ID, _BASIC_ID, _CUSTOM_ID, _TEXT),
    ],
    ids=["custom-only", "basic-only"],
)
async def test_prompt_and_validator_narrow_to_the_catalog_the_renderer_names(
    supported: str, offered: str, withheld: str, component: dict[str, Any]
) -> None:
    accepted = await serve_turn(
        a2ui_reply(_surface(offered, component)),
        body=_body(supported, "https://unknown.example/c.json"),
        protocol_version="v1.0",
        custom_catalog=_CUSTOM,
    )
    rejected = await serve_turn(
        a2ui_reply(_surface(withheld, component)),
        "Plain text.",
        body=_body(supported),
        protocol_version="v1.0",
        custom_catalog=_CUSTOM,
    )

    prompt = "\n".join(accepted.model_calls[0].prompt)
    assert offered in prompt
    assert withheld not in prompt
    assert accepted.feedback == []
    assert accepted.ui == _surface(offered, component)
    assert rejected.errors == [
        [
            f"- Operation 0: unknown catalog '{withheld}' (at /createSurface/catalogId); use one of: '{offered}'",
        ]
    ]


@pytest.mark.asyncio
async def test_a_renderer_naming_every_catalog_is_offered_all_of_them() -> None:
    turn = await serve_turn(
        a2ui_reply(_surface(_CUSTOM_ID, _WIDGET)),
        body=_body(_CUSTOM_ID, _BASIC_ID),
        protocol_version="v1.0",
        custom_catalog=_CUSTOM,
    )

    prompt = "\n".join(turn.model_calls[0].prompt)
    assert _CUSTOM_ID in prompt
    assert _BASIC_ID in prompt
    assert turn.ui == _surface(_CUSTOM_ID, _WIDGET)


@pytest.mark.asyncio
async def test_no_catalog_in_common_runs_the_turn_without_a2ui(caplog: pytest.LogCaptureFixture) -> None:
    clicked: list[str] = []

    @a2ui_action
    def refresh() -> dict[str, Any]:
        clicked.append("refresh")
        return {"updateDataModel": {"surfaceId": "s", "path": "/n", "value": 1}}

    body = {
        **_body("https://unknown.example/c.json"),
        "a2ui": [
            {
                "version": "1.0",
                "action": {
                    "name": "refresh",
                    "surfaceId": "s",
                    "sourceComponentId": "b",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "context": {},
                },
            }
        ],
    }

    with caplog.at_level(logging.WARNING, logger="ag2.a2ui"):
        turn = await serve_turn(
            a2ui_reply(_surface(_BASIC_ID, _TEXT), prose="Done."),
            body=body,
            actions=[refresh],
            protocol_version="v1.0",
        )

    # The handler still runs; neither its messages nor the model's A2UI reach the renderer.
    assert clicked == ["refresh"]
    assert turn.frames == [{"text": "Done."}]
    prompt = "\n".join(turn.model_calls[0].prompt)
    assert _BASIC_ID not in prompt
    assert "A2UI Client Capabilities" not in prompt
    assert "supports none of the agent's catalogs" in caplog.text
