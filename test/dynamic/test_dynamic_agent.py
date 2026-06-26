# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from ag2 import Agent, MemoryStream, tool
from ag2.events import (
    TaskCompleted,
    TaskFailed,
    TaskStarted,
    ToolCallEvent,
    ToolResultEvent,
)
from ag2.testing import TestConfig
from ag2.tools.dynamic import dynamic_agent


@tool
def calc(expression: str) -> str:
    """A trivial calculator."""
    return f"calc({expression})"


@tool
def web_search(query: str) -> str:
    """Toy web search."""
    return f"results for {query}"


def _args(payload: dict) -> str:
    return json.dumps(payload)


def _spec(name: str, *, prompt: list[str] | None = None, tool_names: list[str] | None = None) -> dict:
    return {
        "name": name,
        "prompt": prompt or [],
        "tool_names": tool_names or [],
    }


def _text(event: ToolResultEvent) -> str:
    """Pull the text body out of a ToolResultEvent."""
    return event.result.parts[0].content


@pytest.mark.asyncio
async def test_happy_path_returns_child_reply() -> None:
    child_cfg = TestConfig("Compound interest is 4025.")
    parent_cfg = TestConfig(
        ToolCallEvent(
            name="create_and_run_agent",
            arguments=_args({
                "spec": _spec("math_helper", prompt=["Be precise."], tool_names=["calc"]),
                "objective": "Compute compound interest.",
            }),
        ),
        "Final: 4025.",
    )

    parent = Agent(
        "orchestrator",
        config=parent_cfg,
        tools=[
            dynamic_agent(
                available_tools=[calc, web_search],
                config=child_cfg,
            ),
        ],
    )

    reply = await parent.ask("Help me.")
    assert reply.body == "Final: 4025."


@pytest.mark.asyncio
class TestErrorPaths:
    async def test_unknown_tool_name_returns_error_string(self) -> None:
        child_cfg = TestConfig("ignored")
        parent_stream = MemoryStream()
        parent_cfg = TestConfig(
            ToolCallEvent(
                name="create_and_run_agent",
                arguments=_args({
                    "spec": _spec("broken", tool_names=["nonexistent"]),
                    "objective": "x",
                }),
            ),
            "Recovered.",
        )

        parent = Agent(
            "orchestrator",
            config=parent_cfg,
            tools=[dynamic_agent(available_tools=[calc, web_search], config=child_cfg)],
        )

        reply = await parent.ask("Go", stream=parent_stream)

        assert reply.body == "Recovered."
        events = list(await parent_stream.history.get_events())
        results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(results) == 1
        assert "Error: unknown tools" in _text(results[0])
        assert "nonexistent" in _text(results[0])
        # no run_task ever fired, so no Task events
        assert not [e for e in events if isinstance(e, TaskStarted)]
        assert not [e for e in events if isinstance(e, TaskCompleted)]

    async def test_child_failure_yields_task_failed(self) -> None:
        """Child agent raises (no responses queued) → handler returns 'Error:'."""
        child_cfg = TestConfig()  # empty → StopIteration when invoked
        parent_stream = MemoryStream()
        parent_cfg = TestConfig(
            ToolCallEvent(
                name="create_and_run_agent",
                arguments=_args({
                    "spec": _spec("doomed", tool_names=[]),
                    "objective": "do impossible",
                }),
            ),
            "Recovered.",
        )

        parent = Agent(
            "orchestrator",
            config=parent_cfg,
            tools=[dynamic_agent(available_tools=[calc], config=child_cfg)],
        )

        reply = await parent.ask("Go", stream=parent_stream)

        assert reply.body == "Recovered."
        events = list(await parent_stream.history.get_events())
        results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(results) == 1
        assert _text(results[0]).startswith("Error:")
        assert len([e for e in events if isinstance(e, TaskStarted)]) == 1
        assert len([e for e in events if isinstance(e, TaskFailed)]) == 1


@pytest.mark.asyncio
async def test_response_schema_in_spec_is_forwarded_to_child() -> None:
    """``response_schema`` in spec is accepted and forwarded to the child agent.

    Reusing :class:`AgentSpec` directly as the tool input means the
    parent LLM may pass a ``response_schema``; the framework reconstructs
    it on the child via :meth:`AgentSpec.to_agent`. The call must
    complete without a validation error.
    """
    child_cfg = TestConfig('{"answer": 42}')
    parent_stream = MemoryStream()
    parent_cfg = TestConfig(
        ToolCallEvent(
            name="create_and_run_agent",
            arguments=_args({
                "spec": {
                    "name": "structured_helper",
                    "prompt": [],
                    "tool_names": [],
                    "response_schema": {
                        "name": "Answer",
                        "description": "Structured answer.",
                        "json_schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "integer"}},
                            "required": ["answer"],
                            "additionalProperties": False,
                        },
                    },
                },
                "objective": "x",
            }),
        ),
        "Done.",
    )

    parent = Agent(
        "orchestrator",
        config=parent_cfg,
        tools=[dynamic_agent(available_tools=[calc], config=child_cfg)],
    )

    reply = await parent.ask("Go", stream=parent_stream)

    assert reply.body == "Done."
    events = list(await parent_stream.history.get_events())
    completed = [e for e in events if isinstance(e, TaskCompleted)]
    assert len(completed) == 1
    assert completed[0].agent_name == "structured_helper"
    assert completed[0].result == '{"answer": 42}'


@pytest.mark.asyncio
async def test_child_cannot_recursively_spawn_dynamic_agent() -> None:
    """A spawned dynamic agent must not expose ``create_and_run_agent`` itself.

    We probe this through public behaviour: instruct the child to call
    ``create_and_run_agent``. If the tool were exposed, the call would
    succeed. Because it is not, the framework emits a ``ToolNotFoundEvent``
    on the child stream, ``TestClient`` re-raises it on the next turn,
    and ``run_task`` reports the failure as ``TaskFailed`` on the parent
    stream — naming the missing tool.
    """
    child_cfg = TestConfig(
        ToolCallEvent(
            name="create_and_run_agent",
            arguments=_args({
                "spec": _spec("nested", tool_names=[]),
                "objective": "recurse",
            }),
        ),
        "ignored",
    )
    parent_cfg = TestConfig(
        ToolCallEvent(
            name="create_and_run_agent",
            arguments=_args({
                "spec": _spec("child", tool_names=["calc"]),
                "objective": "try to recurse",
            }),
        ),
        "Final.",
    )

    parent_stream = MemoryStream()
    parent = Agent(
        "orchestrator",
        config=parent_cfg,
        tools=[dynamic_agent(available_tools=[calc], config=child_cfg)],
    )

    reply = await parent.ask("Go", stream=parent_stream)
    assert reply.body == "Final."

    events = list(await parent_stream.history.get_events())
    failed = [e for e in events if isinstance(e, TaskFailed)]
    assert len(failed) == 1
    assert failed[0].agent_name == "child"
    assert "create_and_run_agent" in str(failed[0].error)


@pytest.mark.asyncio
async def test_task_started_and_completed_on_parent_stream() -> None:
    child_cfg = TestConfig("Child reply.")
    parent_cfg = TestConfig(
        ToolCallEvent(
            name="create_and_run_agent",
            arguments=_args({
                "spec": _spec("math_helper", tool_names=[]),
                "objective": "Compute 2+2",
            }),
        ),
        "Final.",
    )

    parent_stream = MemoryStream()
    parent = Agent(
        "orchestrator",
        config=parent_cfg,
        tools=[dynamic_agent(available_tools=[calc], config=child_cfg)],
    )

    await parent.ask("Go", stream=parent_stream)
    events = list(await parent_stream.history.get_events())

    started = [e for e in events if isinstance(e, TaskStarted)]
    completed = [e for e in events if isinstance(e, TaskCompleted)]

    assert len(started) == 1
    assert started[0].agent_name == "math_helper"
    assert started[0].objective == "Compute 2+2"

    assert len(completed) == 1
    assert completed[0].agent_name == "math_helper"
    assert completed[0].result == "Child reply."
