# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Every run that started ends with one terminal event, and then its stream closes."""

import logging

import pytest
from ag_ui.core import DataSource, ImagePart, TextPart, ToolMessage, UserMessage

from ag2 import Agent
from ag2.ag_ui import AGUIStream
from ag2.testing import TestConfig
from test.ag_ui.harness import dispatch_run, exploding_agent, run_input, types_of

pytestmark = pytest.mark.asyncio

# Not base64: decoding it fails while AG-UI input becomes ag2 input.
_MALFORMED = ImagePart(source=DataSource(value="a", mime_type="image/png"))


async def test_a_user_part_that_cannot_be_decoded_ends_the_run_with_run_error() -> None:
    agent = Agent("test_agent", config=TestConfig("never reached"))
    incoming = run_input(UserMessage(id="m1", content=[TextPart(text="look"), _MALFORMED]))

    frames = await dispatch_run(AGUIStream(agent), incoming)

    assert types_of(frames) == ["RUN_STARTED", "RUN_ERROR"]


async def test_a_tool_part_that_cannot_be_decoded_ends_the_run_with_run_error() -> None:
    agent = Agent("test_agent", config=TestConfig("never reached"))
    incoming = run_input(
        UserMessage(id="m1", content="go"),
        ToolMessage(id="m2", tool_call_id="c1", content=[_MALFORMED]),
        UserMessage(id="m3", content="and now?"),
    )

    frames = await dispatch_run(AGUIStream(agent), incoming)

    assert types_of(frames) == ["RUN_STARTED", "RUN_ERROR"]


async def test_a_failed_run_ends_its_stream_after_run_error_and_logs_the_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.ERROR, logger="ag2.ag_ui"):
        frames = await dispatch_run(AGUIStream(exploding_agent()), run_input(UserMessage(id="m1", content="go")))

    assert types_of(frames)[-1] == "RUN_ERROR"
    assert "downstream is down" in frames[-1]["message"]
    [record] = [r for r in caplog.records if r.name.startswith("ag2.ag_ui")]
    assert record.exc_info is not None
    assert "downstream is down" in str(record.exc_info[1])
