# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""The AG-UI failure path: what a client sees when a run dies partway through.

``dispatch`` re-raises after emitting ``RUN_ERROR``, and anyio surfaces that out of
its task group wrapped in an exception group — so a test must expect the group and
unwrap it. The event still arrives first: the memory object stream carrying events to
the encoder is unbuffered, so the send blocks until the consumer takes it and only
then does the re-raise run.
"""

import pytest
from ag_ui.core import EventType, RunErrorEvent, RunFinishedEvent, RunStartedEvent, UserMessage

from ag2 import Agent
from ag2.ag_ui import AGUIStream
from ag2.testing import TestConfig

from .utils import collect_events, create_run_input, exploding_agent, frames_of_failing_run, leaf_exceptions

pytestmark = pytest.mark.asyncio


class TestRunError:
    async def test_run_error_reports_the_failure(self) -> None:
        """The event names what went wrong and is stamped."""
        run_input = create_run_input(UserMessage(id="msg_1", content="go"))

        frames = await frames_of_failing_run(exploding_agent(), run_input)

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
        run_input = create_run_input(UserMessage(id="msg_1", content="go"))

        frames = await frames_of_failing_run(exploding_agent(), run_input)

        started = RunStartedEvent.model_validate(frames[0])
        assert (started.thread_id, started.run_id) == (run_input.thread_id, run_input.run_id)

        run_error = frames[-1]
        assert "threadId" not in run_error
        assert "runId" not in run_error
        assert "thread_id" not in run_error
        assert "run_id" not in run_error

    async def test_original_exception_reaches_the_caller(self) -> None:
        """The run's real cause must not be swallowed or replaced by the error event."""
        run_input = create_run_input(UserMessage(id="msg_1", content="go"))

        with pytest.raises(Exception) as exc_info:
            await collect_events(AGUIStream(exploding_agent()), run_input)

        leaves = leaf_exceptions(exc_info.value)
        assert [type(e) for e in leaves] == [RuntimeError]
        assert str(leaves[0]) == "downstream is down"

    async def test_events_emitted_before_the_failure_are_observable(self) -> None:
        """Everything sent before the re-raise is still available to assert on."""
        run_input = create_run_input(UserMessage(id="msg_1", content="go"))

        frames = await frames_of_failing_run(exploding_agent(), run_input)

        RunStartedEvent.model_validate(frames[0])
        RunErrorEvent.model_validate(frames[-1])
        assert EventType.TOOL_CALL_RESULT in [frame["type"] for frame in frames]

    async def test_a_successful_run_still_finishes_cleanly(self) -> None:
        """The failure path must not disturb the success path."""
        agent = Agent("test_agent", config=TestConfig("all good"))
        run_input = create_run_input(UserMessage(id="msg_1", content="go"))

        frames = await collect_events(AGUIStream(agent), run_input)

        finished = RunFinishedEvent.model_validate(frames[-1])
        assert (finished.thread_id, finished.run_id) == (run_input.thread_id, run_input.run_id)

        started = RunStartedEvent.model_validate(frames[0])
        assert (started.thread_id, started.run_id) == (run_input.thread_id, run_input.run_id)
