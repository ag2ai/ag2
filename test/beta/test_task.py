# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta import Agent, MemoryStream, tool
from autogen.beta.events import ModelMessage, ModelRequest, ModelResponse, TaskCompleted, TaskStarted
from autogen.beta.events.task_events import TaskFailed
from autogen.beta.events.tool_events import ToolCallEvent
from autogen.beta.task import DEFAULT_MAX_TASK_DEPTH, _run_task, _task_depth
from autogen.beta.testing import TestConfig

# ---------------------------------------------------------------------------
# run_task() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_task_basic():
    config = TestConfig(ModelResponse(message=ModelMessage(content="Task done!")))
    agent = Agent("worker", config=config)

    result = await _run_task(agent, "Do something")

    assert result.completed is True
    assert result.result == "Task done!"
    assert result.objective == "Do something"


@pytest.mark.asyncio
async def test_run_task_with_context():
    """Context string is appended to the objective in the prompt."""
    config = TestConfig(ModelResponse(message=ModelMessage(content="Analyzed.")))
    agent = Agent("analyst", config=config)

    result = await _run_task(agent, "Analyze data", context="Here is some data")

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

    result = await _run_task(agent, "This will fail")

    assert result.completed is False
    assert result.result is not None


@pytest.mark.asyncio
async def test_run_task_with_custom_stream():
    """run_task uses the provided stream instead of creating a MemoryStream."""
    config = TestConfig(ModelResponse(message=ModelMessage(content="Done.")))
    agent = Agent("worker", config=config)

    custom_stream = MemoryStream()
    result = await _run_task(agent, "Do it", stream=custom_stream)

    assert result.completed is True
    assert result.stream is custom_stream
    # Events should be on the custom stream
    events = list(await custom_stream.history.get_events())
    assert len(events) > 0
    assert any(isinstance(e, ModelRequest) for e in events)


@pytest.mark.asyncio
async def test_run_task_with_dependencies():
    """Dependencies are passed through to the agent."""
    from autogen.beta.annotations import Context as Ctx

    @tool
    def get_db_name(ctx: Ctx) -> str:
        """Get the database name from dependencies."""
        return ctx.dependencies.get("db_name", "unknown")

    config = TestConfig(
        ToolCallEvent(name="get_db_name", arguments="{}"),
        ModelResponse(message=ModelMessage(content="Got it.")),
    )
    agent = Agent("worker", config=config, tools=[get_db_name])

    result = await _run_task(agent, "Check DB", dependencies={"db_name": "prod_db"})

    assert result.completed is True


@pytest.mark.asyncio
async def test_run_task_default_stream():
    """Without a stream argument, run_task creates a MemoryStream."""
    config = TestConfig(ModelResponse(message=ModelMessage(content="OK")))
    agent = Agent("worker", config=config)

    result = await _run_task(agent, "Test")

    assert result.completed is True
    assert result.stream is not None
    events = list(await result.stream.history.get_events())
    assert len(events) > 0


# ---------------------------------------------------------------------------
# as_tool() — specialist delegation
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
    sub_events = list(await completed.task_stream.history.get_events())
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
    sub_events = list(await completed.task_stream.history.get_events())
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
# as_tool() — self-delegation
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
    agent.tools.append(inner_agent.as_tool(description="Break work into sub-tasks", name="self_delegate"))

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

    # task_stream should be a real stream with history
    assert completed.task_stream is not None
    sub_events = list(await completed.task_stream.history.get_events())
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


# ---------------------------------------------------------------------------
# Stream factory on as_tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_as_tool_stream_factory():
    """Stream factory creates a fresh stream for each sub-task."""
    streams_created: list[MemoryStream] = []

    def make_stream():
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

    def make_stream():
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
    # task_stream exists and is a MemoryStream
    assert completed.task_stream is not None
    sub_events = list(await completed.task_stream.history.get_events())
    assert len(sub_events) > 0


# ---------------------------------------------------------------------------
# Depth guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_max_depth():
    """DEFAULT_MAX_TASK_DEPTH is 1."""
    assert DEFAULT_MAX_TASK_DEPTH == 1


@pytest.mark.asyncio
async def test_depth_guard_safety_net():
    """If depth >= max_depth, delegate returns an error string (safety net)."""
    # Manually set depth to simulate being at max
    from autogen.beta.task import _make_task_tool

    worker_config = TestConfig(ModelResponse(message=ModelMessage(content="Should not reach.")))
    worker = Agent("worker", config=worker_config)
    task_tool = _make_task_tool(worker, description="Test", max_depth=1)

    # Simulate being at depth 1 (>= max_depth=1)
    token = _task_depth.set(1)
    try:
        # The inner FunctionTool's delegate function should return an error
        # We can't easily call it directly, but we can verify schemas are hidden
        schemas = list(await task_tool.schemas(None))
        assert len(schemas) == 0  # tool hidden at this depth
    finally:
        _task_depth.reset(token)


@pytest.mark.asyncio
async def test_depth_guard_tool_visible_at_lower_depth():
    """Task tool is visible when depth < max_depth."""
    worker_config = TestConfig(ModelResponse(message=ModelMessage(content="OK")))
    worker = Agent("worker", config=worker_config)
    task_tool = worker.as_tool(description="Test", max_depth=2)

    # At depth 0, tool should be visible
    schemas = list(await task_tool.schemas(None))
    assert len(schemas) == 1

    # At depth 1, still visible (< 2)
    token = _task_depth.set(1)
    try:
        schemas = list(await task_tool.schemas(None))
        assert len(schemas) == 1
    finally:
        _task_depth.reset(token)

    # At depth 2, hidden (>= 2)
    token = _task_depth.set(2)
    try:
        schemas = list(await task_tool.schemas(None))
        assert len(schemas) == 0
    finally:
        _task_depth.reset(token)


@pytest.mark.asyncio
async def test_custom_max_depth():
    """as_tool respects custom max_depth."""
    worker_config = TestConfig(ModelResponse(message=ModelMessage(content="OK")))
    worker = Agent("worker", config=worker_config)
    task_tool = worker.as_tool(description="Test", max_depth=5)

    # At depth 4, still visible
    token = _task_depth.set(4)
    try:
        schemas = list(await task_tool.schemas(None))
        assert len(schemas) == 1
    finally:
        _task_depth.reset(token)

    # At depth 5, hidden
    token = _task_depth.set(5)
    try:
        schemas = list(await task_tool.schemas(None))
        assert len(schemas) == 0
    finally:
        _task_depth.reset(token)
