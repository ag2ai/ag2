# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Input material this server does not recognise is stripped with a warning, and the run is served.

A malformed *known* value still refuses the input; see ``served/test_run_input_parsing.py``.
"""

import json
import logging
from typing import Any

import pytest

from ag2 import Agent, ToolResult
from ag2.ag_ui import AGUIStream, read_run_input
from ag2.ag_ui.stream import AGStreamInput, map_agui_messages_to_events
from ag2.events import ModelRequest, TextInput, ToolResultEvent, ToolResultsEvent
from ag2.testing import TestConfig, TrackingConfig
from test.ag_ui.harness import dispatch_run, outcome_of

pytestmark = pytest.mark.asyncio


def _body(*messages: dict[str, Any], **extra: Any) -> str:
    return json.dumps({"threadId": "t1", "runId": "r1", "protocolVersion": "1.0", "messages": list(messages), **extra})


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name.startswith("ag2.ag_ui") and r.levelno == logging.WARNING]


async def _served(body: str) -> tuple[list[dict[str, Any]], TrackingConfig]:
    tracking = TrackingConfig(TestConfig("seen"))
    frames = await dispatch_run(AGUIStream(Agent("test_agent", config=tracking)), read_run_input(body))
    return frames, tracking


async def test_a_part_of_an_unknown_kind_is_stripped_and_the_rest_reaches_the_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    body = _body({
        "id": "m1",
        "role": "user",
        "content": [{"type": "text", "text": "look"}, {"type": "hologram", "value": "?"}],
    })

    with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
        frames, tracking = await _served(body)

    assert outcome_of(frames) == {"type": "success"}
    [(sent,)] = [call.args for call in tracking.mock.call_args_list]
    assert sent == ModelRequest([TextInput("look")])
    [warning] = _warnings(caplog)
    assert "/messages/0/content/1" in warning


async def test_an_unknown_top_level_property_is_stripped_and_the_run_served(caplog: pytest.LogCaptureFixture) -> None:
    body = _body({"id": "m1", "role": "user", "content": "hi"}, fromTheFuture=True)

    with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
        frames, _ = await _served(body)

    assert outcome_of(frames) == {"type": "success"}
    [warning] = _warnings(caplog)
    assert "/fromTheFuture" in warning


async def test_an_unknown_nested_property_is_stripped(caplog: pytest.LogCaptureFixture) -> None:
    body = _body({"id": "m1", "role": "user", "content": [{"type": "text", "text": "hi", "glow": "blue"}]})

    with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
        incoming = read_run_input(body)

    assert incoming.model_dump(by_alias=True, exclude_none=True)["messages"] == [
        {"id": "m1", "role": "user", "content": [{"type": "text", "text": "hi"}]}
    ]
    [warning] = _warnings(caplog)
    assert "/messages/0/content/0/glow" in warning


async def test_a_message_of_an_unknown_role_is_stripped(caplog: pytest.LogCaptureFixture) -> None:
    body = _body({"id": "m0", "role": "oracle", "content": "?"}, {"id": "m1", "role": "user", "content": "hi"})

    with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
        frames, tracking = await _served(body)

    assert outcome_of(frames) == {"type": "success"}
    [(sent,)] = [call.args for call in tracking.mock.call_args_list]
    assert sent == ModelRequest([TextInput("hi")])
    [warning] = _warnings(caplog)
    assert "/messages/0" in warning


async def test_open_objects_keep_what_the_protocol_does_not_describe(caplog: pytest.LogCaptureFixture) -> None:
    """`state`, `forwardedProps` and `metadata` are open by key: nothing in them is unrecognised."""
    body = _body(
        {"id": "m1", "role": "user", "content": [{"type": "text", "text": "hi", "metadata": {"mine": None}}]},
        state={"draft": {"anything": None}},
        forwardedProps={"app": [1, None]},
    )

    with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
        incoming = read_run_input(body)

    assert (incoming.state, incoming.forwarded_props) == ({"draft": {"anything": None}}, {"app": [1, None]})
    assert _warnings(caplog) == []


async def test_a_part_whose_source_is_of_an_unknown_kind_is_stripped_whole(caplog: pytest.LogCaptureFixture) -> None:
    """A part left without its source would be malformed, so the whole part goes."""
    body = _body({
        "id": "m1",
        "role": "user",
        "content": [{"type": "text", "text": "look"}, {"type": "image", "source": {"type": "ipfs", "value": "Qm..."}}],
    })

    with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
        frames, tracking = await _served(body)

    assert outcome_of(frames) == {"type": "success"}
    [(sent,)] = [call.args for call in tracking.mock.call_args_list]
    assert sent == ModelRequest([TextInput("look")])
    [warning] = _warnings(caplog)
    assert "/messages/0/content/1" in warning


async def test_a_tool_message_whose_only_part_is_stripped_is_answered_with_the_empty_string(
    caplog: pytest.LogCaptureFixture,
) -> None:
    body = _body(
        {"id": "m1", "role": "user", "content": "weather?"},
        {
            "id": "a1",
            "role": "assistant",
            "toolCalls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}],
        },
        {
            "id": "t1",
            "role": "tool",
            "toolCallId": "c1",
            "content": [{"type": "image", "source": {"type": "ipfs", "value": "Qm..."}}],
        },
    )

    with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
        incoming = read_run_input(body)

    _, history, _ = map_agui_messages_to_events(AGStreamInput(incoming=incoming, variables={}))
    assert history[-1] == ToolResultsEvent([ToolResultEvent(parent_id="c1", name="get_weather", result=ToolResult(""))])
    [warning] = _warnings(caplog)
    assert "/messages/2/content/0" in warning
