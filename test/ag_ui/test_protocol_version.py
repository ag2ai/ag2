# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Each side of a run declares the AG-UI version it speaks.

ag2 declares its own on every `RUN_STARTED`, and reads the client's: another
major is refused before any run starts, anything else is served.
"""

import logging

import pytest
from ag_ui.core import PROTOCOL_VERSION, RunAgentInput, UserMessage
from dirty_equals import IsPartialDict

from ag2 import Agent
from ag2.ag_ui import UNSUPPORTED_PROTOCOL_VERSION, AGUIStream
from ag2.testing import TestConfig, TrackingConfig
from test.ag_ui.harness import dispatch_run, only, outcome_of, run_input, types_of

pytestmark = pytest.mark.asyncio


def _client_on(version: str | None) -> RunAgentInput:
    return run_input(UserMessage(id="m1", content="hi"), protocol_version=version)


async def test_every_run_declares_the_version_ag2_speaks() -> None:
    events = await dispatch_run(
        AGUIStream(Agent("test_agent", config=TestConfig("hello"))), _client_on(PROTOCOL_VERSION)
    )

    assert only(events, "RUN_STARTED") == IsPartialDict({"protocolVersion": PROTOCOL_VERSION})


async def test_it_declares_its_own_version_not_the_client_s() -> None:
    events = await dispatch_run(AGUIStream(Agent("test_agent", config=TestConfig("hello"))), _client_on("1.9"))

    assert only(events, "RUN_STARTED") == IsPartialDict({"protocolVersion": PROTOCOL_VERSION})


@pytest.mark.parametrize("version", ["2.0", "0.9"])
async def test_a_client_on_another_major_is_refused_before_any_run_starts(version: str) -> None:
    tracking = TrackingConfig(TestConfig("hello"))

    events = await dispatch_run(AGUIStream(Agent("test_agent", config=tracking)), _client_on(version))

    assert types_of(events) == ["RUN_ERROR"]
    assert only(events, "RUN_ERROR") == IsPartialDict({"code": UNSUPPORTED_PROTOCOL_VERSION})
    assert tracking.mock.call_args_list == []


@pytest.mark.parametrize("version", ["1.9", "one point oh", "1.0.1"])
async def test_a_newer_minor_or_an_unreadable_version_is_served_with_a_warning(
    version: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
        events = await dispatch_run(AGUIStream(Agent("test_agent", config=TestConfig("hello"))), _client_on(version))

    assert outcome_of(events) == {"type": "success"}
    [warning] = caplog.records
    assert version in warning.getMessage()


@pytest.mark.parametrize("version", [None, PROTOCOL_VERSION])
async def test_a_client_on_this_version_or_predating_versions_is_served_quietly(
    version: str | None, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
        events = await dispatch_run(AGUIStream(Agent("test_agent", config=TestConfig("hello"))), _client_on(version))

    assert outcome_of(events) == {"type": "success"}
    assert caplog.records == []
