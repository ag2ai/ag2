# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import faulthandler
import gc
import sys
import threading
import weakref
from collections.abc import Awaitable, Callable, Iterable
from typing import Annotated
from unittest.mock import MagicMock

import pytest

from ag2 import Agent, Context, MemoryStream, Variable, tool
from ag2.agent import TaskConfig
from ag2.context import StreamId
from ag2.events import (
    BaseEvent,
    HumanInputRequest,
    HumanMessage,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TaskCompleted,
    TaskFailed,
    TaskStarted,
    ToolCallEvent,
    ToolCallsEvent,
    Usage,
    UsageEvent,
)
from ag2.history import MemoryStorage, Storage
from ag2.middleware import BaseMiddleware
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools import Toolkit
from ag2.tools.subagents import background_agent_tool, subagent_tool
from ag2.tools.subagents.run_task import run_task
from ag2.usage import UsageReport


@tool
def noop() -> str:
    """A tool that does nothing."""
    return "ok"


def _tool_names(tools: list) -> set[str]:
    """Collect tool names, expanding any ``Toolkit`` into its member tools."""
    names: set[str] = set()
    for t in tools:
        if isinstance(t, Toolkit):
            names |= {inner.schema.function.name for inner in t.tools}
        else:
            names.add(t.schema.function.name)
    return names


class _BreakableStorage(Storage):
    """A storage backend whose reads start failing once ``break_reads()`` is called.

    Models a backend that goes down mid-run: writes still land, history can no
    longer be read back.
    """

    def __init__(self) -> None:
        self._inner = MemoryStorage()
        self._broken = False

    def break_reads(self) -> None:
        self._broken = True

    async def save_event(self, event: BaseEvent, context: Context) -> None:
        await self._inner.save_event(event, context)

    async def get_history(self, stream_id: StreamId) -> Iterable[BaseEvent]:
        if self._broken:
            raise ConnectionError("storage backend is down")
        return await self._inner.get_history(stream_id)

    async def set_history(self, stream_id: StreamId, events: Iterable[BaseEvent]) -> None:
        await self._inner.set_history(stream_id, events)

    async def drop_history(self, stream_id: StreamId) -> None:
        await self._inner.drop_history(stream_id)


@tool
def flaky() -> str:
    """A tool that fails the way a downstream API does."""
    raise RuntimeError("downstream API is down")


def _make_parent_context(
    *,
    dependencies: dict | None = None,
    variables: dict | None = None,
) -> Context:
    """Helper to build a minimal parent Context for run_task tests."""
    return Context(
        stream=MemoryStream(),
        dependencies=dependencies or {},
        variables=variables or {},
    )


class TestRunTask:
    @pytest.mark.asyncio
    async def test_basic(self):
        config = TestConfig(ModelResponse(ModelMessage("Task done!")))
        agent = Agent("worker", config=config)

        result = await run_task(agent, "Do something", parent_context=_make_parent_context())

        assert result.completed is True
        assert result.result == "Task done!"
        assert result.objective == "Do something"

    @pytest.mark.asyncio
    async def test_with_context(self):
        """Context string is appended to the objective in the prompt."""
        config = TestConfig(ModelResponse(ModelMessage("Analyzed.")))
        agent = Agent("analyst", config=config)

        result = await run_task(
            agent, "Analyze data", parent_context=_make_parent_context(), context="Here is some data"
        )

        assert result.completed is True
        events = list(await result.stream.history.get_events())
        request = [e for e in events if isinstance(e, ModelRequest)][0]
        assert "## Context" in request.parts[0].content
        assert "Here is some data" in request.parts[0].content

    @pytest.mark.asyncio
    async def test_failure(self):
        """Agent that raises an exception produces completed=False."""
        config = TestConfig()  # no responses -> StopIteration
        agent = Agent("broken", config=config)

        result = await run_task(agent, "This will fail", parent_context=_make_parent_context())

        assert result.completed is False
        assert result.error is not None
        assert result.result is None

    @pytest.mark.asyncio
    async def test_with_custom_stream(self):
        """run_task uses the provided stream instead of creating a MemoryStream."""
        config = TestConfig(ModelResponse(ModelMessage("Done.")))
        agent = Agent("worker", config=config)

        custom_stream = MemoryStream()
        result = await run_task(agent, "Do it", parent_context=_make_parent_context(), stream=custom_stream)

        assert result.completed is True
        assert result.stream is custom_stream
        events = list(await custom_stream.history.get_events())
        assert len(events) > 0
        assert any(isinstance(e, ModelRequest) for e in events)

    @pytest.mark.asyncio
    async def test_with_dependencies(self):
        """Dependencies are passed through to the agent."""

        @tool
        def get_db_name(ctx: Context) -> str:
            """Get the database name from dependencies."""
            return ctx.dependencies.get("db_name", "unknown")

        config = TestConfig(
            ToolCallEvent(name="get_db_name", arguments="{}"),
            ModelResponse(ModelMessage("Got it.")),
        )
        agent = Agent("worker", config=config, tools=[get_db_name])

        parent_ctx = _make_parent_context(dependencies={"db_name": "prod_db"})
        result = await run_task(agent, "Check DB", parent_context=parent_ctx)

        assert result.completed is True

    @pytest.mark.asyncio
    async def test_default_stream(self):
        """Without a stream argument, run_task creates a MemoryStream."""
        config = TestConfig(ModelResponse(ModelMessage("OK")))
        agent = Agent("worker", config=config)

        result = await run_task(agent, "Test", parent_context=_make_parent_context())

        assert result.completed is True
        assert result.stream is not None
        events = list(await result.stream.history.get_events())
        assert len(events) > 0


class TestSpecialistDelegation:
    @pytest.mark.asyncio
    async def test_via_as_tool(self):
        """Coordinator delegates to a specialist via as_tool."""
        researcher_config = TestConfig(ModelResponse(ModelMessage("Research findings: X is true.")))
        researcher = Agent("researcher", config=researcher_config)

        coordinator_config = TestConfig(
            ToolCallEvent(name="task_researcher", arguments='{"objective": "Find info about X"}'),
            ModelResponse(ModelMessage("Based on research, X is true.")),
        )
        coordinator = Agent(
            "coordinator",
            config=coordinator_config,
            tools=[researcher.as_tool(description="Delegate research tasks to the researcher agent")],
        )

        reply = await coordinator.ask("Tell me about X")

        assert reply.body == "Based on research, X is true."

    @pytest.mark.asyncio
    async def test_via_subagent_tool(self):
        """Coordinator delegates to a specialist via subagent_tool."""
        researcher_config = TestConfig(ModelResponse(ModelMessage("Research findings: X is true.")))
        researcher = Agent("researcher", config=researcher_config)

        coordinator_config = TestConfig(
            ToolCallEvent(name="task_researcher", arguments='{"objective": "Find info about X"}'),
            ModelResponse(ModelMessage("Based on research, X is true.")),
        )
        coordinator = Agent(
            "coordinator",
            config=coordinator_config,
            tools=[subagent_tool(researcher, description="Delegate research tasks to the researcher agent")],
        )

        reply = await coordinator.ask("Tell me about X")

        assert reply.body == "Based on research, X is true."

    @pytest.mark.asyncio
    async def test_with_context_param(self):
        """The LLM can pass context to the sub-task via the context tool parameter."""
        researcher_config = TestConfig(ModelResponse(ModelMessage("Found data.")))
        researcher = Agent("researcher", config=researcher_config)

        coordinator_config = TestConfig(
            ToolCallEvent(
                name="task_researcher",
                arguments='{"objective": "Find X", "context": "Focus on recent papers"}',
            ),
            ModelResponse(ModelMessage("Done.")),
        )
        coordinator = Agent(
            "coordinator",
            config=coordinator_config,
            tools=[researcher.as_tool(description="Research")],
        )

        parent_stream = MemoryStream()
        await coordinator.ask("Research X", stream=parent_stream)

        events = list(await parent_stream.history.get_events())
        completed = [e for e in events if isinstance(e, TaskCompleted)][0]
        sub_events = list(await parent_stream.history.storage.get_history(completed.task_stream))
        request = [e for e in sub_events if isinstance(e, ModelRequest)][0]
        assert "Focus on recent papers" in request.parts[0].content

    @pytest.mark.asyncio
    async def test_with_tools(self):
        """Sub-task agent uses its own tools during execution."""

        @tool
        def lookup(term: str) -> str:
            """Look up a term."""
            return f"Definition of {term}: something important"

        researcher_config = TestConfig(
            ToolCallEvent(name="lookup", arguments='{"term": "quantum"}'),
            ModelResponse(ModelMessage("Quantum means something important.")),
        )
        researcher = Agent("researcher", config=researcher_config, tools=[lookup])

        coordinator_config = TestConfig(
            ToolCallEvent(name="task_researcher", arguments='{"objective": "Define quantum"}'),
            ModelResponse(ModelMessage("Quantum is important.")),
        )
        coordinator = Agent(
            "coordinator",
            config=coordinator_config,
            tools=[researcher.as_tool(description="Research with lookup")],
        )

        parent_stream = MemoryStream()
        reply = await coordinator.ask("What is quantum?", stream=parent_stream)

        assert reply.body == "Quantum is important."

        events = list(await parent_stream.history.get_events())
        completed = [e for e in events if isinstance(e, TaskCompleted)][0]
        sub_events = list(await parent_stream.history.storage.get_history(completed.task_stream))
        tool_calls = [e for e in sub_events if isinstance(e, ToolCallEvent)]
        assert any(tc.name == "lookup" for tc in tool_calls)

    @pytest.mark.asyncio
    async def test_multiple_specialists(self):
        """Coordinator delegates to multiple specialists sequentially."""
        researcher_config = TestConfig(ModelResponse(ModelMessage("Research done.")))
        researcher = Agent("researcher", config=researcher_config)

        writer_config = TestConfig(ModelResponse(ModelMessage("Article written.")))
        writer = Agent("writer", config=writer_config)

        coordinator_config = TestConfig(
            ToolCallEvent(name="task_researcher", arguments='{"objective": "Research topic"}'),
            ToolCallEvent(name="task_writer", arguments='{"objective": "Write article", "context": "Research done."}'),
            ModelResponse(ModelMessage("All done.")),
        )
        coordinator = Agent(
            "coordinator",
            config=coordinator_config,
            tools=[
                researcher.as_tool(description="Research"),
                writer.as_tool(description="Write"),
            ],
        )

        parent_stream = MemoryStream()
        reply = await coordinator.ask("Write a report", stream=parent_stream)

        assert reply.body == "All done."

        events = list(await parent_stream.history.get_events())
        started = [e for e in events if isinstance(e, TaskStarted)]
        completed = [e for e in events if isinstance(e, TaskCompleted)]
        assert len(started) == 2
        assert len(completed) == 2
        assert started[0].agent_name == "researcher"
        assert started[1].agent_name == "writer"


@pytest.mark.asyncio
async def test_self_delegation():
    """Agent delegates to a copy of itself via as_tool."""
    inner_config = TestConfig(
        ModelResponse(ModelMessage("Sub-task A done.")),
    )
    inner_agent = Agent("analyst", config=inner_config)

    outer_config = TestConfig(
        ToolCallEvent(name="self_delegate", arguments='{"objective": "Sub-task A"}'),
        ModelResponse(ModelMessage("All sub-tasks complete.")),
    )
    agent = Agent("analyst", config=outer_config)
    agent.add_tool(inner_agent.as_tool(description="Break work into sub-tasks", name="self_delegate"))

    reply = await agent.ask("Do complex analysis")

    assert reply.body == "All sub-tasks complete."


class TestLifecycleEvents:
    @pytest.mark.asyncio
    async def test_on_parent_stream(self):
        """TaskStarted and TaskCompleted appear on the parent stream."""
        researcher_config = TestConfig(ModelResponse(ModelMessage("Found it.")))
        researcher = Agent("researcher", config=researcher_config)

        coordinator_config = TestConfig(
            ToolCallEvent(name="task_researcher", arguments='{"objective": "Search for Y"}'),
            ModelResponse(ModelMessage("Done.")),
        )

        parent_stream = MemoryStream()
        coordinator = Agent(
            "coordinator",
            config=coordinator_config,
            tools=[researcher.as_tool(description="Research agent")],
        )

        await coordinator.ask("Find Y", stream=parent_stream)

        events = list(await parent_stream.history.get_events())

        task_started = [e for e in events if isinstance(e, TaskStarted)]
        task_completed = [e for e in events if isinstance(e, TaskCompleted)]

        assert len(task_started) == 1
        assert task_started[0].agent_name == "researcher"
        assert task_started[0].objective == "Search for Y"

        assert len(task_completed) == 1
        assert task_completed[0].agent_name == "researcher"
        assert task_completed[0].result == "Found it."

    @pytest.mark.asyncio
    async def test_completed_has_stream_reference(self):
        """TaskCompleted.task_stream points to the sub-task's stream."""
        worker_config = TestConfig(ModelResponse(ModelMessage("Done.")))
        worker = Agent("worker", config=worker_config)

        coordinator_config = TestConfig(
            ToolCallEvent(name="task_worker", arguments='{"objective": "Do work"}'),
            ModelResponse(ModelMessage("OK.")),
        )

        parent_stream = MemoryStream()
        coordinator = Agent(
            "coordinator",
            config=coordinator_config,
            tools=[worker.as_tool(description="Worker")],
        )

        await coordinator.ask("Go", stream=parent_stream)

        events = list(await parent_stream.history.get_events())
        completed = [e for e in events if isinstance(e, TaskCompleted)][0]

        assert completed.task_stream is not None
        sub_events = list(await parent_stream.history.storage.get_history(completed.task_stream))
        assert len(sub_events) > 0
        assert any(isinstance(e, ModelRequest) for e in sub_events)
        assert any(isinstance(e, ModelResponse) for e in sub_events)

    @pytest.mark.asyncio
    async def test_failure_event(self):
        """Agent that crashes produces TaskFailed on parent stream."""
        broken_config = TestConfig()
        broken = Agent("broken", config=broken_config)

        coordinator_config = TestConfig(
            ToolCallEvent(name="task_broken", arguments='{"objective": "Do impossible thing"}'),
            ModelResponse(ModelMessage("It failed.")),
        )

        parent_stream = MemoryStream()
        coordinator = Agent(
            "coordinator",
            config=coordinator_config,
            tools=[broken.as_tool(description="Broken agent")],
        )

        await coordinator.ask("Try impossible", stream=parent_stream)

        events = list(await parent_stream.history.get_events())
        task_failed = [e for e in events if isinstance(e, TaskFailed)]

        assert len(task_failed) == 1
        assert task_failed[0].agent_name == "broken"
        assert task_failed[0].objective == "Do impossible thing"
        assert isinstance(task_failed[0].error, Exception)
        assert task_failed[0].content  # traceback string is non-empty


class TestStreamFactory:
    @pytest.mark.asyncio
    async def test_creates_fresh_stream(self):
        """Stream factory creates a fresh stream for each sub-task."""
        streams_created: list[MemoryStream] = []

        def make_stream(agent, ctx):
            s = MemoryStream()
            streams_created.append(s)
            return s

        worker_config = TestConfig(ModelResponse(ModelMessage("Done.")))
        worker = Agent("worker", config=worker_config)

        coordinator_config = TestConfig(
            ToolCallEvent(name="task_worker", arguments='{"objective": "Task 1"}'),
            ModelResponse(ModelMessage("OK.")),
        )
        coordinator = Agent(
            "coordinator",
            config=coordinator_config,
            tools=[worker.as_tool(description="Worker", stream=make_stream)],
        )

        await coordinator.ask("Go", stream=MemoryStream())

        assert len(streams_created) == 1
        events = list(await streams_created[0].history.get_events())
        assert any(isinstance(e, ModelRequest) for e in events)

    @pytest.mark.asyncio
    async def test_multiple_calls(self):
        """Each sub-task invocation gets its own stream from the factory."""
        streams_created: list[MemoryStream] = []

        def make_stream(agent, ctx):
            s = MemoryStream()
            streams_created.append(s)
            return s

        worker_config = TestConfig(
            ModelResponse(ModelMessage("A done.")),
            ModelResponse(ModelMessage("B done.")),
        )
        worker = Agent("worker", config=worker_config)

        coordinator_config = TestConfig(
            ToolCallEvent(name="task_worker", arguments='{"objective": "Task A"}'),
            ToolCallEvent(name="task_worker", arguments='{"objective": "Task B"}'),
            ModelResponse(ModelMessage("Both done.")),
        )
        coordinator = Agent(
            "coordinator",
            config=coordinator_config,
            tools=[worker.as_tool(description="Worker", stream=make_stream)],
        )

        await coordinator.ask("Do A and B", stream=MemoryStream())

        assert len(streams_created) == 2
        events_a = list(await streams_created[0].history.get_events())
        events_b = list(await streams_created[1].history.get_events())
        requests_a = [e for e in events_a if isinstance(e, ModelRequest)]
        requests_b = [e for e in events_b if isinstance(e, ModelRequest)]
        assert "Task A" in requests_a[0].parts[0].content
        assert "Task B" in requests_b[0].parts[0].content

    @pytest.mark.asyncio
    async def test_defaults_to_memory_stream(self):
        """Without stream factory, sub-tasks use MemoryStream."""
        worker_config = TestConfig(ModelResponse(ModelMessage("Done.")))
        worker = Agent("worker", config=worker_config)

        coordinator_config = TestConfig(
            ToolCallEvent(name="task_worker", arguments='{"objective": "Do it"}'),
            ModelResponse(ModelMessage("OK.")),
        )

        parent_stream = MemoryStream()
        coordinator = Agent(
            "coordinator",
            config=coordinator_config,
            tools=[worker.as_tool(description="Worker")],
        )

        await coordinator.ask("Go", stream=parent_stream)

        events = list(await parent_stream.history.get_events())
        completed = [e for e in events if isinstance(e, TaskCompleted)][0]
        assert completed.task_stream is not None
        sub_events = list(await parent_stream.history.storage.get_history(completed.task_stream))
        assert len(sub_events) > 0


class TestVariablesPropagation:
    @pytest.mark.asyncio
    async def test_propagates_variables(self, mock: MagicMock) -> None:
        @tool
        def read_var(secret: Annotated[str, Variable("secret")], ctx: Context) -> str:
            """Read a variable from context."""
            mock(secret)
            return ctx.variables["secret"]

        worker_config = TestConfig(
            ToolCallEvent(name="read_var", arguments="{}"),
            ModelResponse(ModelMessage("Got the secret.")),
        )
        worker = Agent("worker", config=worker_config, tools=[read_var])

        coordinator_config = TestConfig(
            ToolCallEvent(name="task_worker", arguments='{"objective": "Read the secret"}'),
            ModelResponse(ModelMessage("Secret retrieved.")),
        )
        coordinator = Agent(
            "coordinator",
            config=coordinator_config,
            tools=[worker.as_tool(description="Worker with variable access")],
        )

        await coordinator.ask("Get secret", variables={"secret": "42"})

        mock.assert_called_once_with("42")

    @pytest.mark.asyncio
    async def test_child_mutations_do_not_leak_to_parent(self) -> None:
        """Child variable mutations are isolated from the parent context.

        ``run_task`` deliberately does not sync child variable mutations back
        to the parent: with concurrent sibling subtasks via ``asyncio.gather``,
        last-writer-wins would silently clobber values. Each subtask runs on
        a copy of the parent's variables and any mutations stay scoped to it.
        """

        @tool
        def mutate_var(ctx: Context) -> str:
            """Add a new variable and update an existing one."""
            ctx.variables["new_key"] = "new_value"
            ctx.variables["counter"] = ctx.variables["counter"] + 1
            return "mutated"

        worker_config = TestConfig(
            ToolCallEvent(name="mutate_var", arguments="{}"),
            ModelResponse(ModelMessage("Done.")),
        )
        worker = Agent("worker", config=worker_config, tools=[mutate_var])

        parent_ctx = _make_parent_context(variables={"counter": 10, "existing": "yes"})

        result = await run_task(worker, "Mutate", parent_context=parent_ctx)

        assert result.completed is True
        # Parent's variables are unchanged — the child mutated its own copy.
        assert parent_ctx.variables == {"counter": 10, "existing": "yes"}


@pytest.mark.asyncio
class TestSubtaskOptOut:
    async def test_default_actor_has_no_subtask_tools(self) -> None:
        """A bare Agent has no subtask tools by default (``tasks=False``)."""
        agent = Agent("default")

        # Auto-injected tools live in ``_additional_tools``, not ``agent.tools``.
        assert _tool_names(agent._additional_tools) == set()

    async def test_explicit_disabled_actor_has_no_subtask_tools(self) -> None:
        """Explicit ``tasks=False`` also suppresses subtask tools."""
        agent = Agent("disabled", tasks=False)

        assert _tool_names(agent._additional_tools) == set()

    async def test_taskconfig_actor_has_subtask_tools(self) -> None:
        """Passing ``tasks=TaskConfig(...)`` opts in to the auto-injected subtask tools."""
        agent = Agent("enabled", tasks=TaskConfig())

        assert _tool_names(agent._additional_tools) == {"run_subtask", "run_subtasks"}

    async def test_taskconfig_rejects_non_positive_max_concurrency(self) -> None:
        with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
            TaskConfig(max_concurrency=0)


@pytest.mark.asyncio
class TestSubtaskInheritance:
    async def test_subtask_inherits_parent_tools(self) -> None:
        """A subtask spawned via ``run_subtask`` sees the parent's tools."""

        @tool
        def lookup(term: str) -> str:
            """Look up a term."""
            return f"definition of {term}"

        # Subtask's first response calls lookup; second wraps up.
        subtask_inner_config = TestConfig(
            ToolCallEvent(name="lookup", arguments='{"term": "quark"}'),
            ModelResponse(ModelMessage("a quark is definition of quark")),
        )
        # Parent: dispatch run_subtask, then summarize.
        parent_config = TestConfig(
            ToolCallEvent(name="run_subtask", arguments='{"task": "Define quark"}'),
            ModelResponse(ModelMessage("Got it.")),
        )
        # The subtask Agent uses the parent's task config — but we can't share
        # config instances because TestConfig consumes responses in order. So
        # we use a TaskConfig with a fresh config we control.
        parent = Agent(
            "parent",
            config=parent_config,
            tools=[lookup],
            tasks=TaskConfig(config=subtask_inner_config),
        )

        parent_stream = MemoryStream()
        reply = await parent.ask("Define quark", stream=parent_stream)

        assert reply.body == "Got it."
        events = list(await parent_stream.history.get_events())
        completed = [e for e in events if isinstance(e, TaskCompleted)][0]
        # Subtask actually called lookup — proving it inherited the tool.
        sub_events = list(await parent_stream.history.storage.get_history(completed.task_stream))
        assert any(isinstance(e, ToolCallEvent) and e.name == "lookup" for e in sub_events)

    async def test_subtask_excludes_via_exclude_tools(self) -> None:
        tracking = TrackingConfig(TestConfig("Done without the excluded tool."))
        parent = Agent(
            "parent",
            config=TestConfig(ToolCallEvent(name="run_subtask", arguments='{"task":"work"}'), "done"),
            tools=[noop, flaky],
            tasks=TaskConfig(config=tracking, exclude_tools=["flaky"]),
        )
        await parent.ask("start")
        assert {schema.function.name for schema in tracking.calls[0].tools} == {"noop"}

    async def test_subtask_include_tools_allowlist(self) -> None:
        """``include_tools=[...]`` keeps only the named tools."""

        @tool
        def t_a() -> str:
            return "a"

        @tool
        def t_b() -> str:
            return "b"

        @tool
        def t_c() -> str:
            return "c"

        sub_config = TestConfig(ModelResponse(ModelMessage("Done.")))
        parent = Agent(
            "parent",
            config=TestConfig(),
            tools=[t_a, t_b, t_c],
            tasks=TaskConfig(config=sub_config, include_tools=["t_a", "t_c"]),
        )

        # Drive _spawn_subtask directly.
        ctx = _make_parent_context()
        result = await parent._spawn_subtask("any", ctx)
        assert result == "Done."

    async def test_subtask_extra_tools_added(self) -> None:
        """``extra_tools=[...]`` adds tools beyond what the parent has."""

        @tool
        def parent_only() -> str:
            return "parent"

        @tool
        def subtask_only() -> str:
            return "subtask"

        # Subtask must see BOTH the inherited and the extra tool.
        sub_config = TestConfig(
            ToolCallEvent(name="subtask_only", arguments="{}"),
            ModelResponse(ModelMessage("Used the extra tool.")),
        )
        parent_config = TestConfig(
            ToolCallEvent(name="run_subtask", arguments='{"task": "Use subtask_only"}'),
            ModelResponse(ModelMessage("OK.")),
        )
        parent = Agent(
            "parent",
            config=parent_config,
            tools=[parent_only],
            tasks=TaskConfig(config=sub_config, extra_tools=[subtask_only]),
        )

        parent_stream = MemoryStream()
        await parent.ask("go", stream=parent_stream)

        events = list(await parent_stream.history.get_events())
        completed = [e for e in events if isinstance(e, TaskCompleted)][0]
        sub_events = list(await parent_stream.history.storage.get_history(completed.task_stream))
        # The extra tool was actually invoked in the subtask.
        assert any(isinstance(e, ToolCallEvent) and e.name == "subtask_only" for e in sub_events)


@pytest.mark.asyncio
class TestSubtaskNoRecursion:
    async def test_child_has_no_subtask_tools(self) -> None:
        tracking = TrackingConfig(TestConfig("done"))
        parent = Agent(
            "parent",
            config=TestConfig(ToolCallEvent(name="run_subtask", arguments='{"task":"work"}'), "done"),
            tasks=TaskConfig(config=tracking),
        )
        await parent.ask("start")
        assert tracking.calls[0].tools == ()


@pytest.mark.asyncio
class TestParallelSubtasks:
    async def test_max_concurrency_bounds_parallel_run_subtask_calls(self) -> None:
        parent = Agent(
            "parent",
            config=TestConfig(
                [
                    ToolCallEvent(name="run_subtask", arguments='{"task": "do A"}'),
                    ToolCallEvent(name="run_subtask", arguments='{"task": "do B"}'),
                    ToolCallEvent(name="run_subtask", arguments='{"task": "do C"}'),
                ],
                "done",
            ),
            tasks=TaskConfig(
                config=TestConfig(ModelResponse(ModelMessage("subtask done."))),
                max_concurrency=2,
            ),
        )
        stream = MemoryStream()
        active = 0
        peak = 0
        two_started = asyncio.Event()
        release = asyncio.Event()

        @stream.where(TaskStarted).subscribe
        async def hold_started(_: TaskStarted) -> None:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 2:
                two_started.set()
            try:
                await release.wait()
            finally:
                active -= 1

        turn = asyncio.create_task(parent.ask("go", stream=stream))
        await asyncio.wait_for(two_started.wait(), timeout=1)

        assert active == 2
        assert not turn.done()

        release.set()
        reply = await asyncio.wait_for(turn, timeout=1)

        assert reply.body == "done"
        assert peak == 2

    async def test_max_concurrency_bounds_run_subtasks_fan_out(self) -> None:
        parent = Agent(
            "parent",
            config=TestConfig(
                ToolCallEvent(
                    name="run_subtasks",
                    arguments='{"tasks": ["task X", "task Y"], "parallel": true}',
                ),
                "done",
            ),
            tasks=TaskConfig(
                config=TestConfig(ModelResponse(ModelMessage("subtask done."))),
                max_concurrency=1,
            ),
        )
        stream = MemoryStream()
        active = 0
        peak = 0
        started = asyncio.Event()
        release = asyncio.Event()

        @stream.where(TaskStarted).subscribe
        async def hold_started(_: TaskStarted) -> None:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            started.set()
            try:
                await release.wait()
            finally:
                active -= 1

        turn = asyncio.create_task(parent.ask("go", stream=stream))
        await asyncio.wait_for(started.wait(), timeout=1)

        assert active == 1
        assert not turn.done()

        release.set()
        reply = await asyncio.wait_for(turn, timeout=1)
        events = list(await stream.history.get_events())

        assert reply.body == "done"
        assert peak == 1
        assert len([event for event in events if isinstance(event, TaskCompleted)]) == 2

    async def test_parallel_run_subtask_calls_succeed(self) -> None:
        """The LLM can emit multiple ``run_subtask`` tool_use blocks in one
        assistant message; the executor dispatches them concurrently and
        the parent sees one ``TaskStarted`` / ``TaskCompleted`` per call.

        This is the parallel-tool-use pattern Anthropic recommends — N
        independent subtasks bundled into one assistant turn become N
        parallel children, each with its own stream.
        """
        parent_config = TestConfig(
            ModelResponse(
                tool_calls=ToolCallsEvent(
                    calls=[
                        ToolCallEvent(name="run_subtask", arguments='{"task": "do A"}'),
                        ToolCallEvent(name="run_subtask", arguments='{"task": "do B"}'),
                        ToolCallEvent(name="run_subtask", arguments='{"task": "do C"}'),
                    ]
                )
            ),
            ModelResponse(ModelMessage("All three done.")),
        )

        parent = Agent(
            "parent",
            config=parent_config,
            # Each spawn rebuilds the TestClient from this config, so every
            # child runs against an independent queue with this single response.
            tasks=TaskConfig(config=TestConfig(ModelResponse(ModelMessage("subtask done.")))),
        )

        parent_stream = MemoryStream()
        reply = await parent.ask("go", stream=parent_stream)

        assert reply.body == "All three done."
        events = list(await parent_stream.history.get_events())
        starts = [e for e in events if isinstance(e, TaskStarted)]
        completions = [e for e in events if isinstance(e, TaskCompleted)]
        # All three subtasks dispatched and completed concurrently.
        assert len(starts) == 3
        assert len(completions) == 3

    async def test_run_subtasks_bundles_parallel_dispatch(self) -> None:
        """``run_subtasks(parallel=True)`` is one tool call, N subtasks.

        Each spawned subtask Agent creates its own TestClient from the shared
        ``TaskConfig.config``, so all children produce the same first response;
        we assert on dispatch shape (2 TaskCompleted events for 2 task strings)
        rather than on per-call content.
        """
        parent_config = TestConfig(
            ToolCallEvent(
                name="run_subtasks",
                arguments='{"tasks": ["task X", "task Y"], "parallel": true}',
            ),
            ModelResponse(ModelMessage("Wrapped up.")),
        )

        parent = Agent(
            "parent",
            config=parent_config,
            tasks=TaskConfig(config=TestConfig(ModelResponse(ModelMessage("subtask done.")))),
        )

        parent_stream = MemoryStream()
        await parent.ask("go", stream=parent_stream)

        events = list(await parent_stream.history.get_events())
        completions = [e for e in events if isinstance(e, TaskCompleted)]
        assert len(completions) == 2


def test_run_subtask_description_advertises_parallel_invocation() -> None:
    """The ``run_subtask`` tool description must inform the LLM that it can
    be invoked multiple times in parallel within a single response.

    This is what teaches Claude/GPT/Gemini to emit multiple ``tool_use``
    blocks in one assistant message rather than serializing them.
    """
    agent = Agent("any", tasks=TaskConfig())
    [toolkit] = agent._additional_tools
    [run_subtask, run_subtasks] = toolkit.tools

    desc = run_subtask.schema.function.description
    assert "parallel" in desc.lower()
    # And ``run_subtasks`` advertises the bundled fan-out.
    assert "parallel" in run_subtasks.schema.function.description.lower()


def test_max_concurrency_agent_is_reusable_across_event_loops() -> None:
    """A capped Agent survives being driven from more than one event loop.

    ``asyncio.Semaphore`` binds to the first loop that awaits it *while
    contended*, so a semaphore cached on the Agent for its whole lifetime would
    raise ``RuntimeError`` on the second loop. This is deliberately sync (one
    ``asyncio.run`` per turn) and uses a cap smaller than the fan-out so the
    semaphore is guaranteed to block.
    """
    parent = Agent(
        "parent",
        config=TestConfig(
            [
                ToolCallEvent(name="run_subtask", arguments='{"task": "do A"}'),
                ToolCallEvent(name="run_subtask", arguments='{"task": "do B"}'),
                ToolCallEvent(name="run_subtask", arguments='{"task": "do C"}'),
            ],
            "done",
        ),
        tasks=TaskConfig(
            config=TestConfig(ModelResponse(ModelMessage("subtask done."))),
            max_concurrency=1,
        ),
    )

    async def turn() -> tuple[str | None, int]:
        stream = MemoryStream()
        active = 0
        peak = 0

        @stream.where(TaskStarted).subscribe
        async def track(_: TaskStarted) -> None:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1

        reply = await parent.ask("go", stream=stream)
        return reply.body, peak

    for _ in range(3):
        body, peak = asyncio.run(turn())
        assert body == "done"
        assert peak == 1


def test_max_concurrency_holds_per_loop_under_concurrent_threads() -> None:
    """Two OS threads driving one capped Agent each see the cap on their own loop.

    A fresh Agent per round, because the cap can only go missing while its
    semaphore is first provisioned, and a tightened switch interval so CPython
    interleaves the two threads inside that window. The interval is left alone
    under a tracer (coverage): on 3.11 a thread switch landing inside a new
    thread's ``settrace`` kills that thread, so AnyIO's worker never answers.
    """
    old_interval = sys.getswitchinterval()
    if sys.gettrace() is None:
        sys.setswitchinterval(0.00001)

    def make_parent() -> Agent:
        return Agent(
            "parent",
            config=TestConfig(
                [
                    ToolCallEvent(name="run_subtask", arguments='{"task": "do A"}'),
                    ToolCallEvent(name="run_subtask", arguments='{"task": "do B"}'),
                    ToolCallEvent(name="run_subtask", arguments='{"task": "do C"}'),
                ],
                "done",
            ),
            tasks=TaskConfig(
                config=TestConfig(ModelResponse(ModelMessage("subtask done."))),
                max_concurrency=1,
            ),
        )

    async def turn(parent: Agent) -> int:
        stream = MemoryStream()
        active = 0
        peak = 0

        @stream.where(TaskStarted).subscribe
        async def track(_: TaskStarted) -> None:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1

        await parent.ask("go", stream=stream)
        return peak

    peaks: list[int | BaseException] = [0, 0]

    def worker(parent: Agent, index: int) -> None:
        try:
            peaks[index] = asyncio.run(turn(parent))
        except BaseException as exc:  # surfaced by the assertions below
            peaks[index] = exc

    try:
        for _round in range(10):
            parent = make_parent()
            # Daemon threads: a worker that fails to finish is the very thing the
            # assertion below reports, and a non-daemon one would then keep
            # `threading._shutdown` — and so the whole pytest process — waiting
            # on it forever after the session ends.
            threads = [threading.Thread(target=worker, args=(parent, i), daemon=True) for i in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            hung = [thread for thread in threads if thread.is_alive()]
            if hung:
                # Seen on Windows CI only. Dump every thread's stack so the next
                # occurrence says *where* a worker is stuck, not merely that it is.
                faulthandler.dump_traceback()
            assert not hung, "a worker thread hung"

            assert peaks == [1, 1], f"each thread's own loop should have capped at 1: {peaks!r}"
    finally:
        sys.setswitchinterval(old_interval)


def test_max_concurrency_does_not_retain_finished_loops() -> None:
    """Finished loops are not pinned by the semaphores provisioned for them.

    A contended ``asyncio.Semaphore`` caches its loop, so weak keys alone never
    reclaim these entries.
    """
    parent = Agent(
        "parent",
        config=TestConfig(
            [
                ToolCallEvent(name="run_subtask", arguments='{"task": "do A"}'),
                ToolCallEvent(name="run_subtask", arguments='{"task": "do B"}'),
            ],
            "done",
        ),
        tasks=TaskConfig(
            config=TestConfig(ModelResponse(ModelMessage("subtask done."))),
            max_concurrency=1,
        ),
    )

    loops: list[weakref.ref[asyncio.AbstractEventLoop]] = []

    async def turn() -> None:
        loops.append(weakref.ref(asyncio.get_running_loop()))
        await parent.ask("go")

    for _ in range(5):
        asyncio.run(turn())
    gc.collect()

    alive = [loop for loop in loops[:-1] if loop() is not None]
    assert not alive, f"{len(alive)} finished event loop(s) still retained by the Agent"


@pytest.mark.asyncio
class TestSubtaskSourceUsage:
    async def test_each_source_call_keeps_its_provider_and_model(self) -> None:
        worker = Agent(
            "worker",
            config=TestConfig(
                ModelResponse(
                    tool_calls=ToolCallsEvent([ToolCallEvent(name="noop", arguments="{}")]),
                    usage=Usage(prompt_tokens=10, completion_tokens=5),
                    model="small",
                    provider="first",
                ),
                ModelResponse(
                    ModelMessage("done"),
                    usage=Usage(prompt_tokens=20, completion_tokens=8),
                    model="large",
                    provider="second",
                ),
            ),
            tools=[noop],
        )
        ctx = _make_parent_context()
        result = await run_task(worker, "work", parent_context=ctx)
        report = UsageReport.from_events(await ctx.stream.history.get_events())
        assert result.usage == Usage(prompt_tokens=30, completion_tokens=13)
        assert report.total == result.usage
        assert report.by_model == {
            "small": Usage(prompt_tokens=10, completion_tokens=5),
            "large": Usage(prompt_tokens=20, completion_tokens=8),
        }
        assert [record.kind for record in report.records] == ["model_call", "model_call"]

    async def test_failure_keeps_previously_published_spend(self) -> None:
        spent = Usage(prompt_tokens=250, completion_tokens=50, total_tokens=300)
        worker = Agent(
            "worker",
            config=TestConfig(
                ModelResponse(tool_calls=ToolCallsEvent([ToolCallEvent(name="flaky", arguments="{}")]), usage=spent),
            ),
            tools=[flaky],
        )
        ctx = _make_parent_context()
        result = await run_task(worker, "work", parent_context=ctx)
        assert not result.completed
        assert isinstance(result.error, RuntimeError)
        assert result.usage == spent
        assert UsageReport.from_events(await ctx.stream.history.get_events()).total == spent
        events = list(await ctx.stream.history.get_events())
        assert [type(event) for event in events if isinstance(event, (UsageEvent, TaskFailed))] == [
            UsageEvent,
            TaskFailed,
        ]

    async def test_lifecycle_suppression_does_not_suppress_accounting(self) -> None:
        spent = Usage(total_tokens=30)
        worker = Agent("worker", config=TestConfig(ModelResponse(ModelMessage("done"), usage=spent)))
        ctx = _make_parent_context()
        result = await run_task(worker, "work", parent_context=ctx, emit_events=False)
        events = list(await ctx.stream.history.get_events())
        assert result.completed
        assert UsageReport.from_events(events).total == spent
        assert not any(isinstance(event.source_event, (TaskStarted, TaskCompleted, TaskFailed)) for event in events)

    async def test_failure_before_any_call_spends_nothing(self) -> None:
        ctx = _make_parent_context()
        result = await run_task(Agent("broken", config=TestConfig()), "work", parent_context=ctx)
        assert not result.completed
        assert result.usage == Usage()
        assert UsageReport.from_events(await ctx.stream.history.get_events()).total == Usage()

    async def test_storage_outage_does_not_hide_spend_or_failure(self) -> None:
        storage = _BreakableStorage()
        spent = Usage(total_tokens=30)
        worker = Agent(
            "worker",
            config=TestConfig(
                ModelResponse(
                    tool_calls=ToolCallsEvent([ToolCallEvent(name="break_storage", arguments="{}")]), usage=spent
                ),
            ),
            tools=[break_storage],
        )
        ctx = _make_parent_context(dependencies={"breakable_storage": storage})
        result = await run_task(worker, "work", parent_context=ctx, stream=MemoryStream(storage=storage))
        assert not result.completed
        assert isinstance(result.error, ConnectionError)
        assert result.usage == spent
        assert UsageReport.from_events(await ctx.stream.history.get_events()).total == spent


async def break_storage(ctx: Context) -> str:
    """Model an external storage failure through a public tool."""
    ctx.dependencies["breakable_storage"].break_reads()
    raise RuntimeError("downstream API is down")


class TestHitlPropagation:
    @pytest.mark.asyncio
    async def test_reuses_parent_hitl_hook(self, mock: MagicMock) -> None:
        """Subagent should use the parent agent's HITL hook when it calls ctx.input()."""

        worker_config = TestConfig(
            ToolCallEvent(name="ask_human", arguments="{}"),
            ModelResponse(ModelMessage("Human approved.")),
        )
        worker = Agent("worker", config=worker_config)

        @worker.tool
        async def ask_human(ctx: Context) -> str:
            """Tool that asks for human input."""
            answer = await ctx.input("Need approval", timeout=1.0)
            mock.tool_got(answer)
            return f"Human said: {answer}"

        coordinator_config = TestConfig(
            ToolCallEvent(name="task_worker", arguments='{"objective": "Get approval"}'),
            ModelResponse(ModelMessage("Approval obtained.")),
        )
        coordinator = Agent(
            "coordinator",
            config=coordinator_config,
            tools=[worker.as_tool(description="Worker that needs human input")],
        )

        @coordinator.hitl_hook
        def hitl_hook(event: HumanInputRequest) -> HumanMessage:
            mock.hitl_called(event.content)
            return HumanMessage("approved")

        await coordinator.ask("Get approval from human")

        mock.hitl_called.assert_called_once_with("Need approval")
        mock.tool_got.assert_called_once_with("approved")

    @pytest.mark.asyncio
    async def test_own_hitl_hook_takes_priority(self, mock: MagicMock) -> None:
        """When the subagent has its own HITL hook, it should be used instead of the parent's."""

        worker_config = TestConfig(
            ToolCallEvent(name="ask_human", arguments="{}"),
            ModelResponse(ModelMessage("Done.")),
        )
        worker = Agent("worker", config=worker_config)

        @worker.tool
        async def ask_human(ctx: Context) -> str:
            """Tool that asks for human input."""
            answer = await ctx.input("Need approval", timeout=1.0)
            mock.tool_got(answer)
            return f"Human said: {answer}"

        @worker.hitl_hook
        def worker_hitl(event: HumanInputRequest) -> HumanMessage:
            mock.worker_hitl(event.content)
            return HumanMessage("worker answer")

        coordinator_config = TestConfig(
            ToolCallEvent(name="task_worker", arguments='{"objective": "Get approval"}'),
            ModelResponse(ModelMessage("OK.")),
        )
        coordinator = Agent(
            "coordinator",
            config=coordinator_config,
            tools=[worker.as_tool(description="Worker")],
        )

        @coordinator.hitl_hook
        def parent_hitl(event: HumanInputRequest) -> HumanMessage:
            mock.parent_hitl(event.content)
            return HumanMessage("parent answer")

        await coordinator.ask("Go")

        mock.worker_hitl.assert_called_once_with("Need approval")
        mock.tool_got.assert_called_once_with("worker answer")
        mock.parent_hitl.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_parent_hears_which_delegation_asks(self) -> None:
        worker = Agent("worker", config=TestConfig(ToolCallEvent(name="ask_human", arguments="{}"), "Done."))

        @worker.tool
        async def ask_human(ctx: Context) -> str:
            """Ask the human."""
            return await ctx.input("Need approval", timeout=1.0)

        coordinator = Agent(
            "coordinator",
            config=TestConfig(ToolCallEvent(name="task_worker", arguments='{"objective": "Go"}'), "OK."),
            tools=[worker.as_tool(description="Worker")],
        )
        heard: list[HumanInputRequest] = []
        started: list[TaskStarted] = []

        @coordinator.hitl_hook
        def parent_hitl(event: HumanInputRequest) -> HumanMessage:
            heard.append(event)
            return HumanMessage("yes")

        stream = MemoryStream()
        stream.where(TaskStarted).subscribe(_append_to(started))
        await coordinator.ask("Go", stream=stream)

        [request] = heard
        [task] = started
        assert request.task_id == task.task_id


def _append_to(into: list[BaseEvent]) -> Callable[[BaseEvent], Awaitable[None]]:
    async def _append(event: BaseEvent) -> None:
        into.append(event)

    return _append


class TestAsToolStreamArgument:
    """`as_tool(stream=...)` accepts a Stream instance, a StreamFactory, or
    None, and raises ``TypeError`` for anything else.
    """

    @pytest.mark.asyncio
    async def test_stream_instance_captures_subagent_events(self):
        """A ``Stream`` instance captures the sub-agent's events."""
        researcher_config = TestConfig(ModelResponse(ModelMessage("Found X.")))
        researcher = Agent("researcher", config=researcher_config)

        coordinator_config = TestConfig(
            ToolCallEvent(name="task_researcher", arguments='{"objective": "Find X"}'),
            ModelResponse(ModelMessage("OK.")),
        )
        sub_stream = MemoryStream()
        coordinator = Agent(
            "coordinator",
            config=coordinator_config,
            tools=[researcher.as_tool(description="Research", stream=sub_stream)],
        )

        await coordinator.ask("Find X")

        sub_events = list(await sub_stream.history.get_events())
        # TaskStarted / TaskCompleted land on the parent stream, not this one.
        assert any(isinstance(e, ModelRequest) for e in sub_events)
        assert any(isinstance(e, ModelResponse) for e in sub_events)

    @pytest.mark.asyncio
    async def test_stream_factory_still_works(self):
        """The existing ``StreamFactory`` (callable) contract is unchanged."""
        researcher_config = TestConfig(ModelResponse(ModelMessage("Found X.")))
        researcher = Agent("researcher", config=researcher_config)

        captured: list[MemoryStream] = []

        def factory(_agent, _ctx) -> MemoryStream:
            s = MemoryStream()
            captured.append(s)
            return s

        coordinator_config = TestConfig(
            ToolCallEvent(name="task_researcher", arguments='{"objective": "Find X"}'),
            ModelResponse(ModelMessage("OK.")),
        )
        coordinator = Agent(
            "coordinator",
            config=coordinator_config,
            tools=[researcher.as_tool(description="Research", stream=factory)],
        )

        await coordinator.ask("Find X")

        assert len(captured) == 1
        events = list(await captured[0].history.get_events())
        # Factory-produced stream is the sub-agent's stream — same routing as
        # the instance case: ModelRequest / ModelResponse land here.
        assert any(isinstance(e, ModelRequest) for e in events)
        assert any(isinstance(e, ModelResponse) for e in events)

    def test_invalid_stream_type_raises_typeerror(self):
        """Passing a non-callable / non-Stream / non-None must raise eagerly
        (no more silent empty-stream surprise)."""
        researcher_config = TestConfig(ModelResponse(ModelMessage("...")))
        researcher = Agent("researcher", config=researcher_config)

        with pytest.raises(TypeError, match="must be a Stream instance"):
            researcher.as_tool(description="Research", stream=42)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="must be a Stream instance"):
            researcher.as_tool(description="Research", stream="not a stream")  # type: ignore[arg-type]

    def test_stream_none_keeps_default_behaviour(self):
        """``stream=None`` (the default) leaves run_task to allocate a fresh
        per-call stream, just like before."""
        researcher_config = TestConfig(ModelResponse(ModelMessage("...")))
        researcher = Agent("researcher", config=researcher_config)

        # Should not raise and should produce a usable FunctionTool.
        delegate = researcher.as_tool(description="Research", stream=None)
        assert delegate is not None

    def test_stream_class_instead_of_instance_raises_typeerror(self):
        """A Stream class is rejected; only an instance is accepted."""
        researcher = Agent("researcher", config=TestConfig(ModelResponse(ModelMessage("..."))))

        with pytest.raises(TypeError, match="not an instance"):
            researcher.as_tool(description="Research", stream=MemoryStream)  # type: ignore[arg-type]

    def test_non_stream_factory_class_is_still_accepted(self):
        """Only *Stream* classes are rejected; other callables stay valid factories."""
        researcher = Agent("researcher", config=TestConfig(ModelResponse(ModelMessage("..."))))

        class MakeStream:
            def __init__(self, agent, ctx) -> None:
                self.stream = MemoryStream()

        assert researcher.as_tool(description="Research", stream=MakeStream) is not None  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_stream_instance_makes_subagent_stateful(self):
        """A reused stream carries history forward: the second delegation sees
        the first one's turns."""
        seen: list[list[str]] = []

        class CaptureLLMCalls(BaseMiddleware):
            async def on_llm_call(self, call_next, events, context):
                seen.append([type(e).__name__ for e in events])
                return await call_next(events, context)

        researcher_config = TestConfig(
            ModelResponse(ModelMessage("A1.")),
            ModelResponse(ModelMessage("A2.")),
        )
        researcher = Agent("researcher", config=researcher_config, middleware=[CaptureLLMCalls])

        coordinator_config = TestConfig(
            ToolCallEvent(name="task_researcher", arguments='{"objective": "FIRST"}'),
            ToolCallEvent(name="task_researcher", arguments='{"objective": "SECOND"}'),
            ModelResponse(ModelMessage("Done.")),
        )
        sub_stream = MemoryStream()
        coordinator = Agent(
            "coordinator",
            config=coordinator_config,
            tools=[researcher.as_tool(description="Research", stream=sub_stream)],
        )

        await coordinator.ask("go")

        # The second delegation replays the first one's turns before its own.
        assert seen == [
            ["ModelRequest"],
            ["ModelRequest", "ModelResponse", "ModelRequest"],
        ]

    @pytest.mark.asyncio
    async def test_stream_none_keeps_delegations_isolated(self):
        """The default gets a fresh stream per delegation, so no history leaks."""
        seen: list[list[str]] = []

        class CaptureLLMCalls(BaseMiddleware):
            async def on_llm_call(self, call_next, events, context):
                seen.append([type(e).__name__ for e in events])
                return await call_next(events, context)

        researcher_config = TestConfig(
            ModelResponse(ModelMessage("A1.")),
            ModelResponse(ModelMessage("A2.")),
        )
        researcher = Agent("researcher", config=researcher_config, middleware=[CaptureLLMCalls])

        coordinator_config = TestConfig(
            ToolCallEvent(name="task_researcher", arguments='{"objective": "FIRST"}'),
            ToolCallEvent(name="task_researcher", arguments='{"objective": "SECOND"}'),
            ModelResponse(ModelMessage("Done.")),
        )
        coordinator = Agent(
            "coordinator",
            config=coordinator_config,
            tools=[researcher.as_tool(description="Research")],
        )

        await coordinator.ask("go")

        assert seen == [["ModelRequest"], ["ModelRequest"]]


class TestBackgroundToolStreamArgument:
    """``background_agent_tool(stream=...)`` accepts a Stream instance, a
    StreamFactory, or None, and raises ``TypeError`` for anything else.
    """

    @pytest.mark.asyncio
    async def test_stream_instance_captures_subagent_events(self):
        researcher = Agent("researcher", config=TestConfig(ModelResponse(ModelMessage("bg done."))))
        sub_stream = MemoryStream()

        coordinator = Agent(
            "coordinator",
            config=TestConfig(
                ToolCallEvent(name="background_task_researcher", arguments='{"objective": "Q"}'),
                ModelResponse(ModelMessage("ack")),
                ModelResponse(ModelMessage("final")),
            ),
            tools=[background_agent_tool(researcher, description="bg", stream=sub_stream)],
        )

        await coordinator.ask("go")

        # ``background_agent_tool`` runs the subagent fire-and-forget via
        # ``spawn_background``, so ``coordinator.ask`` can return before the
        # subagent's ModelResponse lands in the stream. Poll until it arrives.
        async def _events_with_response() -> list:
            for _ in range(100):
                events = list(await sub_stream.history.get_events())
                if any(isinstance(e, ModelResponse) for e in events):
                    return events
                await asyncio.sleep(0.01)
            return list(await sub_stream.history.get_events())

        events = await _events_with_response()
        assert any(isinstance(e, ModelRequest) for e in events)
        assert any(isinstance(e, ModelResponse) for e in events)

    def test_invalid_stream_type_raises_typeerror(self):
        researcher = Agent("researcher", config=TestConfig(ModelResponse(ModelMessage("..."))))

        with pytest.raises(TypeError, match="must be a Stream instance"):
            background_agent_tool(researcher, description="bg", stream=42)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="not an instance"):
            background_agent_tool(researcher, description="bg", stream=MemoryStream)  # type: ignore[arg-type]

    def test_stream_factory_and_none_still_accepted(self):
        researcher = Agent("researcher", config=TestConfig(ModelResponse(ModelMessage("..."))))

        assert background_agent_tool(researcher, description="bg", stream=None) is not None
        assert background_agent_tool(researcher, description="bg", stream=lambda a, c: MemoryStream()) is not None
