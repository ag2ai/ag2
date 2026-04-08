# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated
from unittest.mock import MagicMock

import pytest

from autogen.beta import Agent, MemoryStream, tool
from autogen.beta.annotations import Context as Ctx
from autogen.beta.annotations import Variable
from autogen.beta.context import Context
from autogen.beta.events import (
    HumanInputRequest,
    HumanMessage,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TaskCompleted,
    TaskStarted,
)
from autogen.beta.events.task_events import TaskFailed
from autogen.beta.events.tool_events import ToolCallEvent, ToolCallsEvent
from autogen.beta.testing import TestConfig, TrackingConfig
from autogen.beta.tools.subagents import depth_limiter, subagent_tool
from autogen.beta.tools.subagents.run_task import run_task


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


# ---------------------------------------------------------------------------
# run_task() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_task_basic():
    config = TestConfig(ModelResponse(message=ModelMessage(content="Task done!")))
    agent = Agent("worker", config=config)

    result = await run_task(agent, "Do something", parent_context=_make_parent_context())

    assert result.completed is True
    assert result.result == "Task done!"
    assert result.objective == "Do something"


@pytest.mark.asyncio
async def test_run_task_with_context():
    """Context string is appended to the objective in the prompt."""
    config = TestConfig(ModelResponse(message=ModelMessage(content="Analyzed.")))
    agent = Agent("analyst", config=config)

    result = await run_task(agent, "Analyze data", parent_context=_make_parent_context(), context="Here is some data")

    assert result.completed is True
    # Verify the prompt included the context
    events = list(await result.stream.history.get_events())
    request = [e for e in events if isinstance(e, ModelRequest)][0]
    assert "## Context" in request.content
    assert "Here is some data" in request.content


@pytest.mark.asyncio
async def test_run_task_failure():
    """Agent that raises an exception produces completed=False."""
    config = TestConfig()  # no responses → StopIteration
    agent = Agent("broken", config=config)

    result = await run_task(agent, "This will fail", parent_context=_make_parent_context())

    assert result.completed is False
    assert result.error is not None
    assert result.result is None


@pytest.mark.asyncio
async def test_run_task_with_custom_stream():
    """run_task uses the provided stream instead of creating a MemoryStream."""
    config = TestConfig(ModelResponse(message=ModelMessage(content="Done.")))
    agent = Agent("worker", config=config)

    custom_stream = MemoryStream()
    result = await run_task(agent, "Do it", parent_context=_make_parent_context(), stream=custom_stream)

    assert result.completed is True
    assert result.stream is custom_stream
    # Events should be on the custom stream
    events = list(await custom_stream.history.get_events())
    assert len(events) > 0
    assert any(isinstance(e, ModelRequest) for e in events)


@pytest.mark.asyncio
async def test_run_task_with_dependencies():
    """Dependencies are passed through to the agent."""

    @tool
    def get_db_name(ctx: Ctx) -> str:
        """Get the database name from dependencies."""
        return ctx.dependencies.get("db_name", "unknown")

    config = TestConfig(
        ToolCallEvent(name="get_db_name", arguments="{}"),
        ModelResponse(message=ModelMessage(content="Got it.")),
    )
    agent = Agent("worker", config=config, tools=[get_db_name])

    parent_ctx = _make_parent_context(dependencies={"db_name": "prod_db"})
    result = await run_task(agent, "Check DB", parent_context=parent_ctx)

    assert result.completed is True


@pytest.mark.asyncio
async def test_run_task_default_stream():
    """Without a stream argument, run_task creates a MemoryStream."""
    config = TestConfig(ModelResponse(message=ModelMessage(content="OK")))
    agent = Agent("worker", config=config)

    result = await run_task(agent, "Test", parent_context=_make_parent_context())

    assert result.completed is True
    assert result.stream is not None
    events = list(await result.stream.history.get_events())
    assert len(events) > 0


# ---------------------------------------------------------------------------
# subagent_tool / as_tool — specialist delegation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_specialist_delegation():
    """Coordinator delegates to a specialist via as_tool."""
    researcher_config = TestConfig(ModelResponse(message=ModelMessage(content="Research findings: X is true.")))
    researcher = Agent("researcher", config=researcher_config)

    coordinator_config = TestConfig(
        ToolCallEvent(name="task_researcher", arguments='{"objective": "Find info about X"}'),
        ModelResponse(message=ModelMessage(content="Based on research, X is true.")),
    )
    coordinator = Agent(
        "coordinator",
        config=coordinator_config,
        tools=[researcher.as_tool(description="Delegate research tasks to the researcher agent")],
    )

    reply = await coordinator.ask("Tell me about X")

    assert reply.body == "Based on research, X is true."


@pytest.mark.asyncio
async def test_specialist_delegation_with_subagent_tool():
    """Coordinator delegates to a specialist via subagent_tool."""
    researcher_config = TestConfig(ModelResponse(message=ModelMessage(content="Research findings: X is true.")))
    researcher = Agent("researcher", config=researcher_config)

    coordinator_config = TestConfig(
        ToolCallEvent(name="task_researcher", arguments='{"objective": "Find info about X"}'),
        ModelResponse(message=ModelMessage(content="Based on research, X is true.")),
    )
    coordinator = Agent(
        "coordinator",
        config=coordinator_config,
        tools=[subagent_tool(researcher, description="Delegate research tasks to the researcher agent")],
    )

    reply = await coordinator.ask("Tell me about X")

    assert reply.body == "Based on research, X is true."


@pytest.mark.asyncio
async def test_specialist_delegation_with_context_param():
    """The LLM can pass context to the sub-task via the context tool parameter."""
    researcher_config = TestConfig(ModelResponse(message=ModelMessage(content="Found data.")))
    researcher = Agent("researcher", config=researcher_config)

    coordinator_config = TestConfig(
        ToolCallEvent(
            name="task_researcher",
            arguments='{"objective": "Find X", "context": "Focus on recent papers"}',
        ),
        ModelResponse(message=ModelMessage(content="Done.")),
    )
    coordinator = Agent(
        "coordinator",
        config=coordinator_config,
        tools=[researcher.as_tool(description="Research")],
    )

    parent_stream = MemoryStream()
    await coordinator.ask("Research X", stream=parent_stream)

    # Verify the sub-task's stream has the context in the request
    events = list(await parent_stream.history.get_events())
    completed = [e for e in events if isinstance(e, TaskCompleted)][0]
    sub_events = list(await parent_stream.history.storage.get_history(completed.task_stream))
    request = [e for e in sub_events if isinstance(e, ModelRequest)][0]
    assert "Focus on recent papers" in request.content


@pytest.mark.asyncio
async def test_specialist_with_tools():
    """Sub-task agent uses its own tools during execution."""

    @tool
    def lookup(term: str) -> str:
        """Look up a term."""
        return f"Definition of {term}: something important"

    # Researcher uses lookup tool, then responds
    researcher_config = TestConfig(
        ToolCallEvent(name="lookup", arguments='{"term": "quantum"}'),
        ModelResponse(message=ModelMessage(content="Quantum means something important.")),
    )
    researcher = Agent("researcher", config=researcher_config, tools=[lookup])

    coordinator_config = TestConfig(
        ToolCallEvent(name="task_researcher", arguments='{"objective": "Define quantum"}'),
        ModelResponse(message=ModelMessage(content="Quantum is important.")),
    )
    coordinator = Agent(
        "coordinator",
        config=coordinator_config,
        tools=[researcher.as_tool(description="Research with lookup")],
    )

    parent_stream = MemoryStream()
    reply = await coordinator.ask("What is quantum?", stream=parent_stream)

    assert reply.body == "Quantum is important."

    # Verify sub-task used the lookup tool
    events = list(await parent_stream.history.get_events())
    completed = [e for e in events if isinstance(e, TaskCompleted)][0]
    sub_events = list(await parent_stream.history.storage.get_history(completed.task_stream))
    tool_calls = [e for e in sub_events if isinstance(e, ToolCallEvent)]
    assert any(tc.name == "lookup" for tc in tool_calls)


@pytest.mark.asyncio
async def test_multiple_specialists():
    """Coordinator delegates to multiple specialists sequentially."""
    researcher_config = TestConfig(ModelResponse(message=ModelMessage(content="Research done.")))
    researcher = Agent("researcher", config=researcher_config)

    writer_config = TestConfig(ModelResponse(message=ModelMessage(content="Article written.")))
    writer = Agent("writer", config=writer_config)

    coordinator_config = TestConfig(
        ToolCallEvent(name="task_researcher", arguments='{"objective": "Research topic"}'),
        ToolCallEvent(name="task_writer", arguments='{"objective": "Write article", "context": "Research done."}'),
        ModelResponse(message=ModelMessage(content="All done.")),
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


# ---------------------------------------------------------------------------
# as_tool — self-delegation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_self_delegation():
    """Agent delegates to a copy of itself via as_tool."""
    inner_config = TestConfig(
        ModelResponse(message=ModelMessage(content="Sub-task A done.")),
    )
    inner_agent = Agent("analyst", config=inner_config)

    outer_config = TestConfig(
        ToolCallEvent(name="self_delegate", arguments='{"objective": "Sub-task A"}'),
        ModelResponse(message=ModelMessage(content="All sub-tasks complete.")),
    )
    agent = Agent("analyst", config=outer_config)
    agent.add_tool(inner_agent.as_tool(description="Break work into sub-tasks", name="self_delegate"))

    reply = await agent.ask("Do complex analysis")

    assert reply.body == "All sub-tasks complete."


# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lifecycle_events_on_parent_stream():
    """TaskStarted and TaskCompleted appear on the parent stream."""
    researcher_config = TestConfig(ModelResponse(message=ModelMessage(content="Found it.")))
    researcher = Agent("researcher", config=researcher_config)

    coordinator_config = TestConfig(
        ToolCallEvent(name="task_researcher", arguments='{"objective": "Search for Y"}'),
        ModelResponse(message=ModelMessage(content="Done.")),
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
async def test_task_completed_has_stream_reference():
    """TaskCompleted.task_stream points to the sub-task's stream."""
    worker_config = TestConfig(ModelResponse(message=ModelMessage(content="Done.")))
    worker = Agent("worker", config=worker_config)

    coordinator_config = TestConfig(
        ToolCallEvent(name="task_worker", arguments='{"objective": "Do work"}'),
        ModelResponse(message=ModelMessage(content="OK.")),
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

    # task_stream is a StreamId pointing to the sub-task's history
    assert completed.task_stream is not None
    sub_events = list(await parent_stream.history.storage.get_history(completed.task_stream))
    assert len(sub_events) > 0
    assert any(isinstance(e, ModelRequest) for e in sub_events)
    assert any(isinstance(e, ModelResponse) for e in sub_events)


@pytest.mark.asyncio
async def test_task_failure_event():
    """Agent that crashes produces TaskFailed on parent stream."""
    broken_config = TestConfig()
    broken = Agent("broken", config=broken_config)

    coordinator_config = TestConfig(
        ToolCallEvent(name="task_broken", arguments='{"objective": "Do impossible thing"}'),
        ModelResponse(message=ModelMessage(content="It failed.")),
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


# ---------------------------------------------------------------------------
# Stream factory on as_tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_as_tool_stream_factory():
    """Stream factory creates a fresh stream for each sub-task."""
    streams_created: list[MemoryStream] = []

    def make_stream(agent, ctx):
        s = MemoryStream()
        streams_created.append(s)
        return s

    worker_config = TestConfig(ModelResponse(message=ModelMessage(content="Done.")))
    worker = Agent("worker", config=worker_config)

    coordinator_config = TestConfig(
        ToolCallEvent(name="task_worker", arguments='{"objective": "Task 1"}'),
        ModelResponse(message=ModelMessage(content="OK.")),
    )
    coordinator = Agent(
        "coordinator",
        config=coordinator_config,
        tools=[worker.as_tool(description="Worker", stream=make_stream)],
    )

    await coordinator.ask("Go", stream=MemoryStream())

    assert len(streams_created) == 1
    # The factory-created stream should have events
    events = list(await streams_created[0].history.get_events())
    assert any(isinstance(e, ModelRequest) for e in events)


@pytest.mark.asyncio
async def test_as_tool_stream_factory_multiple_calls():
    """Each sub-task invocation gets its own stream from the factory."""
    streams_created: list[MemoryStream] = []

    def make_stream(agent, ctx):
        s = MemoryStream()
        streams_created.append(s)
        return s

    worker_config = TestConfig(
        ModelResponse(message=ModelMessage(content="A done.")),
        ModelResponse(message=ModelMessage(content="B done.")),
    )
    worker = Agent("worker", config=worker_config)

    # Coordinator calls worker twice sequentially
    coordinator_config = TestConfig(
        ToolCallEvent(name="task_worker", arguments='{"objective": "Task A"}'),
        ToolCallEvent(name="task_worker", arguments='{"objective": "Task B"}'),
        ModelResponse(message=ModelMessage(content="Both done.")),
    )
    coordinator = Agent(
        "coordinator",
        config=coordinator_config,
        tools=[worker.as_tool(description="Worker", stream=make_stream)],
    )

    await coordinator.ask("Do A and B", stream=MemoryStream())

    assert len(streams_created) == 2
    # Each stream should have independent events
    events_a = list(await streams_created[0].history.get_events())
    events_b = list(await streams_created[1].history.get_events())
    requests_a = [e for e in events_a if isinstance(e, ModelRequest)]
    requests_b = [e for e in events_b if isinstance(e, ModelRequest)]
    assert "Task A" in requests_a[0].content
    assert "Task B" in requests_b[0].content


@pytest.mark.asyncio
async def test_as_tool_no_stream_factory_defaults_to_memory():
    """Without stream factory, sub-tasks use MemoryStream."""
    worker_config = TestConfig(ModelResponse(message=ModelMessage(content="Done.")))
    worker = Agent("worker", config=worker_config)

    coordinator_config = TestConfig(
        ToolCallEvent(name="task_worker", arguments='{"objective": "Do it"}'),
        ModelResponse(message=ModelMessage(content="OK.")),
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
    # task_stream is a StreamId; events are in the shared storage
    assert completed.task_stream is not None
    sub_events = list(await parent_stream.history.storage.get_history(completed.task_stream))
    assert len(sub_events) > 0


# ---------------------------------------------------------------------------
# Variables propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delegation_propagates_variables(mock: MagicMock) -> None:
    @tool
    def read_var(secret: Annotated[str, Variable("secret")], ctx: Ctx) -> str:
        """Read a variable from context."""
        mock(secret)
        return ctx.variables["secret"]

    worker_config = TestConfig(
        ToolCallEvent(name="read_var", arguments="{}"),
        ModelResponse(message=ModelMessage(content="Got the secret.")),
    )
    worker = Agent("worker", config=worker_config, tools=[read_var])

    coordinator_config = TestConfig(
        ToolCallEvent(name="task_worker", arguments='{"objective": "Read the secret"}'),
        ModelResponse(message=ModelMessage(content="Secret retrieved.")),
    )
    coordinator = Agent(
        "coordinator",
        config=coordinator_config,
        tools=[worker.as_tool(description="Worker with variable access")],
    )

    # act
    await coordinator.ask("Get secret", variables={"secret": "42"})

    # assert
    mock.assert_called_once_with("42")


@pytest.mark.asyncio
async def test_subagent_variable_mutation_affects_parent() -> None:
    """Mutating variables in a sub-task is visible in the parent context
    because run_task syncs variables back after completion."""

    @tool
    def mutate_var(ctx: Ctx) -> str:
        """Add a new variable and update an existing one."""
        ctx.variables["new_key"] = "new_value"
        ctx.variables["counter"] = ctx.variables["counter"] + 1
        return "mutated"

    worker_config = TestConfig(
        ToolCallEvent(name="mutate_var", arguments="{}"),
        ModelResponse(message=ModelMessage(content="Done.")),
    )
    worker = Agent("worker", config=worker_config, tools=[mutate_var])

    parent_ctx = _make_parent_context(variables={"counter": 10, "existing": "yes"})

    result = await run_task(worker, "Mutate", parent_context=parent_ctx)

    assert result.completed is True
    # Mutations from sub-task are visible on the parent context
    assert parent_ctx.variables["counter"] == 11
    assert parent_ctx.variables["new_key"] == "new_value"
    # Pre-existing variable is untouched
    assert parent_ctx.variables["existing"] == "yes"


# ---------------------------------------------------------------------------
# Depth limiter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_depth_limiter_rejects() -> None:
    """Real nested delegation: outer → L1 → L2 → L3 (rejected at depth 3 with max_depth=2).

    outer (depth 0) delegates to L1 (depth 1) which delegates to L2 (depth 2).
    L2 tries to delegate to L3 but the limiter sees depth=2 >= max_depth=2 and blocks.
    """
    limiter = depth_limiter(max_depth=2)

    # L3 — should never actually run
    l3 = Agent(
        "l3",
        config=TestConfig(ModelResponse(message=ModelMessage(content="Should not reach."))),
    )

    # L2 — tries to delegate to L3, gets blocked
    l2_config = TestConfig(
        ToolCallEvent(name="task_l3", arguments='{"objective": "Go even deeper"}'),
        ModelResponse(message=ModelMessage(content="L3 was blocked.")),
    )
    l2_tracking = TrackingConfig(l2_config)
    l2 = Agent(
        "l2",
        config=l2_tracking,
        tools=[l3.as_tool(description="L3 agent", middleware=[limiter])],
    )

    # L1 — delegates to L2 (depth 1, allowed)
    l1 = Agent(
        "l1",
        config=TestConfig(
            ToolCallEvent(name="task_l2", arguments='{"objective": "Go deeper"}'),
            ModelResponse(message=ModelMessage(content="L2 done.")),
        ),
        tools=[l2.as_tool(description="L2 agent", middleware=[limiter])],
    )

    # outer — delegates to L1 (depth 0, allowed)
    outer = Agent(
        "outer",
        config=TestConfig(
            ToolCallEvent(name="task_l1", arguments='{"objective": "Start"}'),
            ModelResponse(message=ModelMessage(content="Done.")),
        ),
        tools=[l1.as_tool(description="L1 agent", middleware=[limiter])],
    )

    await outer.ask("Go")

    # L2's LLM saw the rejection error when it tried to call L3
    tool_results = l2_tracking.mock.call_args_list[1].args[0]
    assert "maximum task depth" in tool_results.results[0].content


@pytest.mark.asyncio
async def test_depth_limiter_passes() -> None:
    """Below max_depth the tool executes and produces a TaskCompleted event."""
    worker_config = TestConfig(ModelResponse(message=ModelMessage(content="Done.")))
    worker = Agent("worker", config=worker_config)

    inner_config = TestConfig(
        ToolCallEvent(name="task_worker", arguments='{"objective": "Do work"}'),
        ModelResponse(message=ModelMessage(content="OK.")),
    )
    tracking_config = TrackingConfig(inner_config)

    coordinator = Agent(
        "coordinator",
        config=tracking_config,
        tools=[worker.as_tool(description="Worker", middleware=[depth_limiter(max_depth=2)])],
    )

    parent_stream = MemoryStream()
    await coordinator.ask("Go", stream=parent_stream)

    # TaskCompleted appears on the parent stream
    events = list(await parent_stream.history.get_events())
    assert any(isinstance(e, TaskCompleted) for e in events)


@pytest.mark.asyncio
async def test_depth_limiter_concurrent_subagents() -> None:
    """Concurrent subagent calls get independent depth counters.

    coordinator (depth 0) dispatches worker_a and worker_b in parallel.
    worker_a itself delegates to sub_worker (depth 2), proving it can go
    deeper without affecting worker_b's depth counter.  If depth leaked
    between siblings, worker_b would see depth=2 and be incorrectly blocked.
    """
    limiter = depth_limiter(max_depth=3)

    # sub_worker — called by worker_a at depth 2
    sub_worker = Agent(
        "sub_worker",
        config=TestConfig(ModelResponse(message=ModelMessage(content="Sub done."))),
    )

    # worker_a — delegates to sub_worker (depth 1 → 2, allowed)
    worker_a = Agent(
        "worker_a",
        config=TestConfig(
            ToolCallEvent(name="task_sub_worker", arguments='{"objective": "Sub-task"}'),
            ModelResponse(message=ModelMessage(content="A done.")),
        ),
        tools=[sub_worker.as_tool(description="Sub-worker", middleware=[limiter])],
    )

    # worker_b — simple leaf, no further delegation
    worker_b = Agent(
        "worker_b",
        config=TestConfig(ModelResponse(message=ModelMessage(content="B done."))),
    )

    # coordinator — dispatches both workers concurrently via a single ToolCallsEvent
    inner_config = TestConfig(
        ModelResponse(
            tool_calls=ToolCallsEvent(
                calls=[
                    ToolCallEvent(name="task_worker_a", arguments='{"objective": "Task A"}'),
                    ToolCallEvent(name="task_worker_b", arguments='{"objective": "Task B"}'),
                ]
            )
        ),
        ModelResponse(message=ModelMessage(content="Both done.")),
    )
    tracking_config = TrackingConfig(inner_config)

    coordinator = Agent(
        "coordinator",
        config=tracking_config,
        tools=[
            worker_a.as_tool(description="Worker A", middleware=[limiter]),
            worker_b.as_tool(description="Worker B", middleware=[limiter]),
        ],
    )

    parent_stream = MemoryStream()
    await coordinator.ask("Go", stream=parent_stream)

    # Both tool results should be successful (not rejected by depth limiter)
    tool_results = tracking_config.mock.call_args_list[1].args[0]
    assert "A done." in tool_results.results[0].content
    assert "B done." in tool_results.results[1].content

    # Both concurrent calls plus sub_worker produce TaskCompleted events
    events = list(await parent_stream.history.get_events())
    completed = [e for e in events if isinstance(e, TaskCompleted)]
    assert len(completed) == 2


# ---------------------------------------------------------------------------
# HITL hook propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subagent_reuses_parent_hitl_hook(mock: MagicMock) -> None:
    """Subagent should use the parent agent's HITL hook when it calls ctx.input().

    The worker agent has a tool that requests human input via ctx.input().
    The HITL hook is set only on the coordinator (parent). The worker (subagent)
    should inherit and use the parent's HITL hook to answer the input request.
    """

    worker_config = TestConfig(
        ToolCallEvent(name="ask_human", arguments="{}"),
        ModelResponse(message=ModelMessage(content="Human approved.")),
    )
    worker = Agent("worker", config=worker_config)

    @worker.tool
    async def ask_human(ctx: Ctx) -> str:
        """Tool that asks for human input."""
        answer = await ctx.input("Need approval", timeout=1.0)
        mock.tool_got(answer)
        return f"Human said: {answer}"

    coordinator_config = TestConfig(
        ToolCallEvent(name="task_worker", arguments='{"objective": "Get approval"}'),
        ModelResponse(message=ModelMessage(content="Approval obtained.")),
    )
    coordinator = Agent(
        "coordinator",
        config=coordinator_config,
        tools=[worker.as_tool(description="Worker that needs human input")],
    )

    @coordinator.hitl_hook
    def hitl_hook(event: HumanInputRequest) -> HumanMessage:
        mock.hitl_called(event.content)
        return HumanMessage(content="approved")

    # act
    await coordinator.ask("Get approval from human")

    # assert
    mock.hitl_called.assert_called_once_with("Need approval")
    mock.tool_got.assert_called_once_with("approved")


@pytest.mark.asyncio
async def test_subagent_own_hitl_hook_takes_priority(mock: MagicMock) -> None:
    """When the subagent has its own HITL hook, it should be used instead of the parent's."""

    worker_config = TestConfig(
        ToolCallEvent(name="ask_human", arguments="{}"),
        ModelResponse(message=ModelMessage(content="Done.")),
    )
    worker = Agent("worker", config=worker_config)

    @worker.tool
    async def ask_human(ctx: Ctx) -> str:
        """Tool that asks for human input."""
        answer = await ctx.input("Need approval", timeout=1.0)
        mock.tool_got(answer)
        return f"Human said: {answer}"

    @worker.hitl_hook
    def worker_hitl(event: HumanInputRequest) -> HumanMessage:
        mock.worker_hitl(event.content)
        return HumanMessage(content="worker answer")

    coordinator_config = TestConfig(
        ToolCallEvent(name="task_worker", arguments='{"objective": "Get approval"}'),
        ModelResponse(message=ModelMessage(content="OK.")),
    )
    coordinator = Agent(
        "coordinator",
        config=coordinator_config,
        tools=[worker.as_tool(description="Worker")],
    )

    @coordinator.hitl_hook
    def parent_hitl(event: HumanInputRequest) -> HumanMessage:
        mock.parent_hitl(event.content)
        return HumanMessage(content="parent answer")

    # act
    await coordinator.ask("Go")

    # assert
    mock.worker_hitl.assert_called_once_with("Need approval")
    mock.tool_got.assert_called_once_with("worker answer")
    mock.parent_hitl.assert_not_called()
