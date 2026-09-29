# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""The AG-UI failure path: what a client sees when a run dies partway through.

``dispatch`` ends on ``RUN_ERROR`` and then returns: the run has already answered, so
the failure is reported on the wire and logged on the server, never raised into the
response body.
"""

import logging

import pytest
from ag_ui.core import EventType, RunErrorEvent, RunFinishedEvent, RunStartedEvent, UserMessage

from ag2 import Agent
from ag2.ag_ui import AGUIStream
from ag2.testing import TestConfig
from test.ag_ui.harness import dispatch_run, exploding_agent, frames_of_failing_run, run_input

pytestmark = pytest.mark.asyncio


class TestRunError:
    async def test_run_error_reports_the_failure(self) -> None:
        """The event names what went wrong and is stamped."""
        incoming = run_input(UserMessage(id="msg_1", content="go"))

        frames = await frames_of_failing_run(exploding_agent(), incoming)

        error = RunErrorEvent.model_validate(frames[-1])
        assert "downstream is down" in error.message
        assert error.timestamp is not None

    async def test_the_run_is_identified_by_run_started_not_by_run_error(self) -> None:
        """``RUN_ERROR`` carries no correlation ids, by protocol design.

        ``RunErrorEvent`` declares only ``message``, ``code`` and ``usage`` — unlike
        ``RunStartedEvent`` and ``RunFinishedEvent``, it has no ``thread_id`` /
        ``run_id``, in 0.1.20 and in the 0.1.21 pre-release alike. Setting them anyway
        would land in the model's extras and serialise as snake_case keys the protocol
        does not define, so ag2 does not. A client identifies the run from
        ``RUN_STARTED`` on the same event stream, which is per-run.

        Asserted on the raw frames rather than the parsed models: extras only exist on
        the wire, so parsing is exactly what would hide the failure this pins.
        """
        incoming = run_input(UserMessage(id="msg_1", content="go"))

        frames = await frames_of_failing_run(exploding_agent(), incoming)

        started = RunStartedEvent.model_validate(frames[0])
        assert (started.thread_id, started.run_id) == (incoming.thread_id, incoming.run_id)

        run_error = frames[-1]
        assert "threadId" not in run_error
        assert "runId" not in run_error
        assert "thread_id" not in run_error
        assert "run_id" not in run_error

    async def test_original_exception_reaches_the_server_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """The run's real cause is kept, traceback and all, where the operator looks."""
        incoming = run_input(UserMessage(id="msg_1", content="go"))

        with caplog.at_level(logging.ERROR, logger="ag2.ag_ui"):
            await dispatch_run(AGUIStream(exploding_agent()), incoming)

        [record] = caplog.records
        assert record.exc_info is not None
        assert (type(record.exc_info[1]), str(record.exc_info[1])) == (RuntimeError, "downstream is down")

    async def test_events_emitted_before_the_failure_are_observable(self) -> None:
        """Everything sent before the failure still reaches the client."""
        incoming = run_input(UserMessage(id="msg_1", content="go"))

        frames = await frames_of_failing_run(exploding_agent(), incoming)

        RunStartedEvent.model_validate(frames[0])
        RunErrorEvent.model_validate(frames[-1])
        assert EventType.TOOL_CALL_RESULT in [frame["type"] for frame in frames]

    async def test_a_successful_run_still_finishes_cleanly(self) -> None:
        """The failure path must not disturb the success path."""
        agent = Agent("test_agent", config=TestConfig("all good"))
        incoming = run_input(UserMessage(id="msg_1", content="go"))

        frames = await dispatch_run(AGUIStream(agent), incoming)

        finished = RunFinishedEvent.model_validate(frames[-1])
        assert (finished.thread_id, finished.run_id) == (incoming.thread_id, incoming.run_id)

        started = RunStartedEvent.model_validate(frames[0])
        assert (started.thread_id, started.run_id) == (incoming.thread_id, incoming.run_id)
