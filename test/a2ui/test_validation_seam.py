# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Validation, retry and extraction behaviour observed at the ``A2UIServer`` seam."""

import json
from collections.abc import Mapping
from typing import Any

import pytest

from ag2 import Agent, tool
from ag2.a2ui import A2UIServer
from ag2.a2ui.transports import RestTransport
from ag2.events import ModelMessage, ModelResponse, ToolCallEvent, ToolCallsEvent

from .seam import BASIC_CATALOG, WIRE_VERSION, a2ui_reply, serve_turn

SURFACE = {"version": "v0.9", "createSurface": {"surfaceId": "s1", "catalogId": BASIC_CATALOG["v0.9"]}}
MISSING_CATALOG = {"version": "v0.9", "createSurface": {"surfaceId": "s1"}}
BROKEN_JSON = "Text.\n<a2ui-json>\n{not valid json}\n</a2ui-json>"


def _with_tool_call(text: str) -> ModelResponse:
    return ModelResponse(
        message=ModelMessage(text),
        tool_calls=ToolCallsEvent([ToolCallEvent(name="ping", arguments="{}")]),
    )


@tool
def ping() -> str:
    """Answer pong."""
    return "pong"


@pytest.mark.asyncio
class TestValidated:
    async def test_valid_reply_streams_prose_then_messages(self) -> None:
        turn = await serve_turn(a2ui_reply([SURFACE], prose="Here is your UI."))

        assert turn.frames == [{"text": "Here is your UI."}, SURFACE]
        assert turn.feedback == []

    async def test_jsonl_block_streams_one_frame_per_line(self) -> None:
        delete = {"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}}
        reply = f"Here.\n<a2ui-json>\n```json\n{_line(SURFACE)}\n{_line(delete)}\n```\n</a2ui-json>"

        turn = await serve_turn(reply)

        assert turn.ui == [SURFACE, delete]

    async def test_plain_text_is_not_validated(self) -> None:
        turn = await serve_turn("Just text.")

        assert turn.frames == [{"text": "Just text."}]
        assert len(turn.model_calls) == 1

    async def test_invalid_reply_is_retried_with_the_bad_turn_and_feedback(self) -> None:
        bad = a2ui_reply([MISSING_CATALOG])

        turn = await serve_turn(bad, a2ui_reply([SURFACE], prose="Fixed."))

        assert turn.frames == [{"text": "Fixed."}, SURFACE]
        retry = turn.model_calls[1].messages
        assert retry[-2] == ModelResponse(ModelMessage(bad))
        assert turn.feedback == [
            "Your A2UI output had validation errors:\n"
            "- Operation 0: 'catalogId' is a required property (at /createSurface)\n"
            "\n"
            'Please fix these errors and regenerate the A2UI JSON. Make sure each message includes "version": "v0.9" '
            "and all required properties. The user never sees this exchange: do not mention the errors or the "
            "correction in your reply."
        ]

    async def test_unparsable_block_is_retried(self) -> None:
        turn = await serve_turn(BROKEN_JSON, a2ui_reply([SURFACE]))

        assert turn.ui == [SURFACE]
        assert turn.feedback[0].count("Invalid JSON") == 1

    async def test_exhausted_retries_degrade_to_prose_and_emit_failure(self) -> None:
        bad = a2ui_reply([MISSING_CATALOG], prose="Here.")

        turn = await serve_turn(bad, bad, validation_retries=1)

        assert turn.frames == [{"text": "Here."}]
        assert [(event.errors, event.attempts) for event in turn.failures] == [
            (["Operation 0: 'catalogId' is a required property (at /createSurface)"], 2)
        ]

    async def test_zero_retries_fails_on_the_first_attempt(self) -> None:
        turn = await serve_turn(a2ui_reply([MISSING_CATALOG]), validation_retries=0)

        assert len(turn.model_calls) == 1
        assert [event.attempts for event in turn.failures] == [1]

    async def test_no_failure_event_on_success(self) -> None:
        turn = await serve_turn(a2ui_reply([SURFACE]))

        assert turn.failures == []

    async def test_tool_calling_reply_publishes_its_ui_and_runs_the_tool(self) -> None:
        turn = await serve_turn(_with_tool_call(a2ui_reply([SURFACE])), "Done.", tools=[ping])

        assert turn.frames == [{"text": "Done."}, SURFACE]

    async def test_invalid_ui_on_a_tool_calling_reply_is_not_retried(self) -> None:
        turn = await serve_turn(_with_tool_call(a2ui_reply([MISSING_CATALOG])), "Done.", tools=[ping])

        assert turn.frames == [{"text": "Done."}]
        assert turn.feedback == []
        assert [event.attempts for event in turn.failures] == [1]


def test_negative_retries_are_rejected() -> None:
    with pytest.raises(ValueError, match="validation_retries must be >= 0"):
        A2UIServer(Agent("ui"), transport=RestTransport(), validation_retries=-1)


@pytest.mark.asyncio
class TestValidationDisabled:
    async def test_valid_block_is_extracted(self) -> None:
        turn = await serve_turn(a2ui_reply([SURFACE], prose="Here is your UI."), validate_responses=False)

        assert turn.frames == [{"text": "Here is your UI."}, SURFACE]

    async def test_invalid_block_is_published_as_is(self) -> None:
        turn = await serve_turn(a2ui_reply([MISSING_CATALOG]), validate_responses=False)

        assert turn.frames == [{"text": "Here."}, MISSING_CATALOG]
        assert len(turn.model_calls) == 1
        assert turn.failures == []

    async def test_unparsable_block_is_stripped_from_prose(self) -> None:
        turn = await serve_turn(BROKEN_JSON, validate_responses=False)

        assert turn.frames == [{"text": "Text."}]

    async def test_tool_calling_reply_still_extracts_ui(self) -> None:
        turn = await serve_turn(_with_tool_call(a2ui_reply([SURFACE])), "Done.", tools=[ping], validate_responses=False)

        assert turn.frames == [{"text": "Done."}, SURFACE]


def _line(message: Mapping[str, object]) -> str:
    return json.dumps(message)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "server_kwargs"),
    [
        pytest.param(SURFACE, {}, id="validated"),
        pytest.param(SURFACE, {"validate_responses": False}, id="extracted-without-validation"),
        pytest.param(MISSING_CATALOG, {"validation_retries": 0}, id="retries-exhausted"),
    ],
)
async def test_stripping_the_a2ui_block_keeps_the_message_metadata(
    message: dict[str, Any], server_kwargs: dict[str, Any]
) -> None:
    reply = ModelResponse(ModelMessage(a2ui_reply([message], prose="Here."), metadata={"trace": "t-1"}))

    turn = await serve_turn(reply, **server_kwargs)

    [response] = turn.responses
    assert response.message == ModelMessage("Here.", metadata={"trace": "t-1"})


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol_version", ["v0.9", "v0.9.1", "v1.0"])
async def test_surface_on_an_unknown_catalog_is_reported_once(protocol_version: str) -> None:
    # The SDK answers CATALOG_NOT_FOUND; the surface's components have no catalog to be checked against,
    # so neither their schema, their references, their reachability nor the root are judged by the agent's catalog.
    unknown = "https://example.com/nope.json"
    version = WIRE_VERSION[protocol_version]
    messages = [
        {"version": version, "createSurface": {"surfaceId": "s", "catalogId": unknown}},
        {
            "version": version,
            "updateComponents": {
                "surfaceId": "s",
                "components": [
                    {"id": "top", "component": "Column", "children": ["a", "b"]},
                    {"id": "orphan", "component": "Text"},
                    {"id": "odd", "component": "NotInTheBasicCatalog"},
                ],
            },
        },
    ]

    turn = await serve_turn(a2ui_reply(messages), "Plain text.", protocol_version=protocol_version)

    assert turn.errors == [
        [
            f"- Operation 0: unknown catalog '{unknown}' (at /createSurface/catalogId); "
            f"use one of: '{BASIC_CATALOG[protocol_version]}'"
        ]
    ]


@pytest.mark.asyncio
async def test_surface_component_is_reported_once() -> None:
    # a2ui_protocol.md, createSurface: the Surface container "cannot be modified using updateComponents".
    messages = [
        {"version": "1.0", "createSurface": {"surfaceId": "s", "catalogId": BASIC_CATALOG["v1.0"]}},
        {
            "version": "1.0",
            "updateComponents": {
                "surfaceId": "s",
                "components": [{"id": "root", "component": "Surface", "child": "root"}],
            },
        },
    ]

    turn = await serve_turn(a2ui_reply(messages), "Plain text.", protocol_version="v1.0")

    assert turn.errors == [
        [
            "- Operation 1: Component 'root' (Surface): component type 'Surface' is reserved for the container "
            "createSurface creates and cannot be sent as a component"
        ]
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("protocol_version", "kinds"),
    [
        ("v0.9", "['createSurface', 'deleteSurface', 'updateComponents', 'updateDataModel']"),
        ("v0.9.1", "['createSurface', 'deleteSurface', 'updateComponents', 'updateDataModel']"),
        (
            "v1.0",
            "['agentFunctionResponse', 'callRendererFunction', 'createSurface', 'deleteSurface', "
            "'updateComponents', 'updateDataModel']",
        ),
    ],
)
async def test_not_a_message_lists_the_served_versions_message_kinds(protocol_version: str, kinds: str) -> None:
    messages = [{"version": WIRE_VERSION[protocol_version], "showSurface": {"surfaceId": "s"}}]

    turn = await serve_turn(a2ui_reply(messages), "Plain text.", protocol_version=protocol_version)

    assert turn.errors == [[f"- Operation 0 is not an A2UI message: expected an object with exactly one of {kinds}"]]


@pytest.mark.asyncio
async def test_v0_9_custom_catalog_keeps_the_basic_components_it_references() -> None:
    # v0.9: a custom catalog is used in place of the basic one (a2ui_protocol.md L157), and may $ref its definitions.
    custom_id = "https://example.com/cat.json"
    catalog = {
        "$id": custom_id,
        "components": {
            "Column": {"$ref": "basic_catalog.json#/components/Column"},
            "Text": {"$ref": "basic_catalog.json#/components/Text"},
        },
    }
    messages = [
        {"version": "v0.9", "createSurface": {"surfaceId": "s", "catalogId": custom_id}},
        {
            "version": "v0.9",
            "updateComponents": {
                "surfaceId": "s",
                "components": [
                    {"id": "root", "component": "Column", "children": ["t"]},
                    {"id": "t", "component": "Text", "text": "hi"},
                ],
            },
        },
    ]

    turn = await serve_turn(a2ui_reply(messages), custom_catalog=catalog)

    assert turn.feedback == []
    assert turn.ui == messages
