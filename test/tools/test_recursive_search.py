# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for recursive_search_tool: WebSwarm-style recursive delegation.

The delegation tree is observed through the shared stream storage: every
``run_task`` child runs on its own stream, and ``TaskCompleted.task_stream``
links a parent stream to its child's, so walking completed tasks level by
level reconstructs the whole tree.
"""

import asyncio
from collections.abc import Sequence
from unittest.mock import MagicMock

import pytest

from ag2 import Agent, Context, MemoryStream, tool
from ag2.events import (
    BaseEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TaskCompleted,
    TaskFailed,
    TaskStarted,
    TextInput,
    ToolCallEvent,
    ToolResultEvent,
    ToolResultsEvent,
)
from ag2.stream import Stream
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools.subagents import (
    SearchMode,
    SubtaskSpec,
    recursive_search_agent,
    recursive_search_tool,
)


def _delegate_script(*objectives: str, mode: str = "wide") -> ToolCallEvent:
    subtasks = ", ".join(f'{{"objective": "{o}", "mode": "{mode}"}}' for o in objectives)
    return ToolCallEvent(name="solve_subtasks", arguments=f'{{"subtasks": [{subtasks}]}}')


async def _collect_tree(stream: Stream, stream_id) -> list[BaseEvent]:
    """All task lifecycle events under ``stream_id``, recursively."""
    collected: list[BaseEvent] = []
    for event in await stream.history.storage.get_history(stream_id):
        if isinstance(event, (TaskStarted, TaskFailed)):
            collected.append(event)
        elif isinstance(event, TaskCompleted):
            collected.append(event)
            collected.extend(await _collect_tree(stream, event.task_stream))
    return collected


async def _run_search(parent: Agent, stream: MemoryStream) -> tuple[str, list[BaseEvent]]:
    reply = await parent.ask("research this", stream=stream)
    return reply.body or "", await _collect_tree(stream, stream.id)


def _starts(events: Sequence[BaseEvent]) -> list[TaskStarted]:
    return [e for e in events if isinstance(e, TaskStarted)]


def _completions(events: Sequence[BaseEvent]) -> list[TaskCompleted]:
    return [e for e in events if isinstance(e, TaskCompleted)]


def _failures(events: Sequence[BaseEvent]) -> list[TaskFailed]:
    return [e for e in events if isinstance(e, TaskFailed)]


def _tool_results_sent_to_llm(mock: MagicMock) -> str:
    """Text of every tool result a TrackingConfig recorded going to the LLM."""
    parts: list[str] = []
    for call in mock.call_args_list:
        message = call.args[0]
        if not isinstance(message, ToolResultsEvent):
            continue
        for result in message.results:
            if isinstance(result, ToolResultEvent):
                parts.extend(p.content for p in result.result.parts if isinstance(p, TextInput))
    return "\n".join(parts)


def _prompts_sent_to_llm(mock: MagicMock) -> str:
    """Text of every model request prompt a TrackingConfig recorded."""
    texts: list[str] = []
    for call in mock.call_args_list:
        message = call.args[0]
        if isinstance(message, ModelRequest):
            texts.extend(p.content for p in message.parts if isinstance(p, TextInput))
    return "\n".join(texts)


@pytest.mark.asyncio
class TestSearchModeContract:
    async def test_modes_are_the_papers_atom_and_wide_verbs(self):
        # This module ports the paper's `atom` and `wide` verbs; `deep` and
        # `entity_collect` are intentionally out of scope for follow-up modules.
        assert {m.value for m in SearchMode} == {"atom", "wide"}

    async def test_subtask_defaults_to_atom(self):
        spec = SubtaskSpec(objective="look up a fact")
        assert spec.mode is SearchMode.ATOM
        assert spec.context == ""

    async def test_delegation_schema_constrains_mode_to_enum(self):
        """The mode parameter reaches the LLM as an enum-constrained schema."""
        node_config = TestConfig(ModelResponse(ModelMessage("done")))
        search = recursive_search_tool(config=node_config)
        root_params = search.schema.function.parameters
        assert set(root_params["properties"]) == {"query", "context"}
        assert root_params["required"] == ["query"]

    async def test_config_is_required(self):
        """No speculative config fallback: nodes need an explicit ModelConfig."""
        with pytest.raises(TypeError):
            recursive_search_tool()  # type: ignore[call-arg]


@pytest.mark.asyncio
class TestDepthBudget:
    @pytest.mark.parametrize("max_depth", [0, 1, 2, 3])
    async def test_tree_depth_matches_budget(self, max_depth: int):
        """Every node carries solve_subtasks and spawns one wide child per level.

        The shared script makes each node delegate once, then reply — so the
        recursion only stops when the depth budget runs out. The tree must
        contain exactly ``max_depth`` levels of children below the root.
        """
        node_config = TestConfig(
            _delegate_script("drill down"),
            ModelResponse(ModelMessage("level findings")),
        )
        parent_config = TestConfig(
            ToolCallEvent(name="recursive_search", arguments='{"query": "research X"}'),
            ModelResponse(ModelMessage("final synthesis")),
        )
        stream = MemoryStream()
        parent = Agent(
            "parent",
            config=parent_config,
            tools=[recursive_search_tool(config=node_config, max_depth=max_depth)],
        )

        body, events = await _run_search(parent, stream)

        assert body == "final synthesis"
        # root + one child per budgeted depth level, and nothing failed.
        assert len(_starts(events)) == max_depth + 1
        assert len(_completions(events)) == max_depth + 1
        assert _failures(events) == []
        assert all(c.result == "level findings" for c in _completions(events))

    async def test_depth_cap_returns_downgrade_sentinel(self):
        """At the cap, solve_subtasks spawns nothing and tells the node to
        solve the objectives itself (downgrade to atom)."""
        node_config = TrackingConfig(
            TestConfig(
                _delegate_script("keep digging"),
                ModelResponse(ModelMessage("solved it myself")),
            )
        )
        parent_config = TestConfig(
            ToolCallEvent(name="recursive_search", arguments='{"query": "research X"}'),
            ModelResponse(ModelMessage("final synthesis")),
        )
        stream = MemoryStream()
        parent = Agent(
            "parent",
            config=parent_config,
            tools=[recursive_search_tool(config=node_config, max_depth=0)],
        )

        body, events = await _run_search(parent, stream)

        assert body == "final synthesis"
        assert len(_starts(events)) == 1  # only the root — no children spawned
        # The sentinel was fed back to the root as the tool result.
        tool_results = _tool_results_sent_to_llm(node_config.mock)
        assert "DELEGATION_BUDGET_EXHAUSTED" in tool_results
        assert "keep digging" in tool_results


@pytest.mark.asyncio
class TestFanOutCap:
    async def test_wide_fan_out_is_truncated_to_max_children(self):
        """Requesting more children than max_children spawns only max_children
        and reports the dropped subtasks back to the delegating node."""
        node_config = TrackingConfig(
            TestConfig(
                _delegate_script("aspect A", "aspect B", "aspect C"),
                ModelResponse(ModelMessage("level findings")),
            )
        )
        parent_config = TestConfig(
            ToolCallEvent(name="recursive_search", arguments='{"query": "research X"}'),
            ModelResponse(ModelMessage("final synthesis")),
        )
        stream = MemoryStream()
        parent = Agent(
            "parent",
            config=parent_config,
            tools=[recursive_search_tool(config=node_config, max_depth=1, max_children=2)],
        )

        body, events = await _run_search(parent, stream)

        assert body == "final synthesis"
        starts = _starts(events)
        # root + exactly 2 children despite 3 requested subtasks.
        assert len(starts) == 3
        assert len(_completions(events)) == 3
        assert _failures(events) == []
        # The truncation notice reached the root with the dropped objective.
        tool_results = _tool_results_sent_to_llm(node_config.mock)
        assert "Fan-out cap reached" in tool_results
        assert "aspect C" in tool_results


@pytest.mark.asyncio
class TestModeGating:
    async def test_wide_child_recurses(self):
        """A wide child carries solve_subtasks and delegates further."""
        node_config = TestConfig(
            _delegate_script("drill down", mode="wide"),
            ModelResponse(ModelMessage("level findings")),
        )
        parent_config = TestConfig(
            ToolCallEvent(name="recursive_search", arguments='{"query": "research X"}'),
            ModelResponse(ModelMessage("final synthesis")),
        )
        stream = MemoryStream()
        parent = Agent(
            "parent",
            config=parent_config,
            tools=[recursive_search_tool(config=node_config, max_depth=2)],
        )

        _, events = await _run_search(parent, stream)

        # root -> wide child -> wide grandchild (whose delegation is capped).
        assert len(_starts(events)) == 3
        assert _failures(events) == []

    async def test_atom_child_cannot_delegate(self):
        """An atom child is built without solve_subtasks: when it tries to
        delegate anyway, the call fails as an unknown tool and no grandchild
        is spawned."""
        node_config = TestConfig(
            _delegate_script("drill down", mode="atom"),
            ModelResponse(ModelMessage("level findings")),
        )
        parent_config = TestConfig(
            ToolCallEvent(name="recursive_search", arguments='{"query": "research X"}'),
            ModelResponse(ModelMessage("final synthesis")),
        )
        stream = MemoryStream()
        parent = Agent(
            "parent",
            config=parent_config,
            tools=[recursive_search_tool(config=node_config, max_depth=2)],
        )

        body, events = await _run_search(parent, stream)

        assert body == "final synthesis"
        # root + atom child only; the child's delegation attempt failed.
        assert len(_starts(events)) == 2
        assert len(_failures(events)) == 1
        assert _failures(events)[0].agent_name == "node_atom_0"


@pytest.mark.asyncio
class TestSearchToolsReachNodes:
    async def test_caller_supplied_search_tools_run_inside_children(self):
        """The web search substitutes (Serper/Jina in the paper) are caller
        tools: every node — root and delegated children — can invoke them."""
        searched: list[str] = []

        @tool
        def web_search(query: str) -> str:
            """Search the web (test double)."""
            searched.append(query)
            return f"results for {query}: FINDING"

        node_config = TestConfig(
            _delegate_script("drill down"),
            ToolCallEvent(name="web_search", arguments='{"query": "X"}'),
            ModelResponse(ModelMessage("synthesis with FINDING")),
        )
        parent_config = TestConfig(
            ToolCallEvent(name="recursive_search", arguments='{"query": "research X"}'),
            ModelResponse(ModelMessage("final synthesis")),
        )
        stream = MemoryStream()
        parent = Agent(
            "parent",
            config=parent_config,
            tools=[recursive_search_tool(config=node_config, tools=[web_search], max_depth=1)],
        )

        body, events = await _run_search(parent, stream)

        assert body == "final synthesis"
        assert len(_completions(events)) == 2  # root and its child both finished
        assert _failures(events) == []
        # Both the root and its child invoked the search tool.
        assert searched == ["X", "X"]


@pytest.mark.asyncio
class TestFailurePath:
    async def test_root_failure_returns_error_message(self):
        """When the root search node fails, recursive_search_tool reports the
        error instead of pretending research succeeded."""

        @tool
        def broken_search(query: str) -> str:
            """Search backend that always fails."""
            raise RuntimeError("search backend down")

        node_config = TestConfig(
            ToolCallEvent(name="broken_search", arguments='{"query": "X"}'),
        )
        parent_config = TrackingConfig(
            TestConfig(
                ToolCallEvent(name="recursive_search", arguments='{"query": "research X"}'),
                ModelResponse(ModelMessage("final synthesis")),
            )
        )
        stream = MemoryStream()
        parent = Agent(
            "parent",
            config=parent_config,
            tools=[recursive_search_tool(config=node_config, tools=[broken_search], max_depth=1)],
        )

        body, events = await _run_search(parent, stream)

        assert body == "final synthesis"
        # The root node failed and the tool surfaced that to the parent.
        assert len(_failures(events)) == 1
        tool_results = _tool_results_sent_to_llm(parent_config.mock)
        assert "Recursive search failed" in tool_results
        assert "search backend down" in tool_results


@pytest.mark.asyncio
class TestContextPropagation:
    async def test_subtask_context_reaches_child(self):
        """SubtaskSpec.context is passed to the child node as run_task context
        and shows up in the child's first model request."""
        node_config = TrackingConfig(
            TestConfig(
                ToolCallEvent(
                    name="solve_subtasks",
                    arguments=('{"subtasks": [{"objective": "drill down", "mode": "wide", "context": "CHILD-CTX-7"}]}'),
                ),
                ModelResponse(ModelMessage("level findings")),
            )
        )
        parent_config = TestConfig(
            ToolCallEvent(name="recursive_search", arguments='{"query": "research X"}'),
            ModelResponse(ModelMessage("final synthesis")),
        )
        stream = MemoryStream()
        parent = Agent(
            "parent",
            config=parent_config,
            tools=[recursive_search_tool(config=node_config, max_depth=1)],
        )

        body, events = await _run_search(parent, stream)

        assert body == "final synthesis"
        assert _failures(events) == []
        # The child's model request carried the subtask's context.
        prompts = _prompts_sent_to_llm(node_config.mock)
        assert "CHILD-CTX-7" in prompts


@pytest.mark.asyncio
class TestParamValidation:
    async def test_rejects_negative_depth(self):
        config = TestConfig(ModelResponse(ModelMessage("done")))
        with pytest.raises(ValueError, match="max_depth"):
            recursive_search_tool(config=config, max_depth=-1)

    async def test_rejects_zero_fan_out(self):
        config = TestConfig(ModelResponse(ModelMessage("done")))
        with pytest.raises(ValueError, match="max_children"):
            recursive_search_tool(config=config, max_children=0)

    async def test_rejects_non_positive_timeout(self):
        config = TestConfig(ModelResponse(ModelMessage("done")))
        with pytest.raises(ValueError, match="timeout"):
            recursive_search_tool(config=config, timeout=0)

    async def test_rejects_zero_node_budget(self):
        config = TestConfig(ModelResponse(ModelMessage("done")))
        with pytest.raises(ValueError, match="max_nodes"):
            recursive_search_tool(config=config, max_nodes=0)

    async def test_rejects_zero_evidence_chars(self):
        config = TestConfig(ModelResponse(ModelMessage("done")))
        with pytest.raises(ValueError, match="max_evidence_chars"):
            recursive_search_tool(config=config, max_evidence_chars=0)

    async def test_rejects_timeout_with_custom_stream(self):
        config = TestConfig(ModelResponse(ModelMessage("done")))
        with pytest.raises(ValueError, match="cannot be combined"):
            recursive_search_tool(
                config=config,
                timeout=30,
                stream=lambda _agent, _ctx: MemoryStream(),
            )


@pytest.mark.asyncio
class TestEmptySubtasks:
    async def test_empty_subtasks_returns_guidance_message(self):
        """solve_subtasks with an empty list returns guidance instead of
        spawning nothing silently — the LLM should be told to try again
        with at least one subtask."""
        node_config = TrackingConfig(
            TestConfig(
                ToolCallEvent(name="solve_subtasks", arguments='{"subtasks": []}'),
                ModelResponse(ModelMessage("ok, I will answer directly")),
            )
        )
        parent_config = TestConfig(
            ToolCallEvent(name="recursive_search", arguments='{"query": "research X"}'),
            ModelResponse(ModelMessage("final synthesis")),
        )
        stream = MemoryStream()
        parent = Agent(
            "parent",
            config=parent_config,
            tools=[recursive_search_tool(config=node_config, max_depth=2)],
        )

        body, events = await _run_search(parent, stream)

        assert body == "final synthesis"
        # No children spawned — only the root ran.
        assert len(_starts(events)) == 1
        assert _failures(events) == []
        # The guidance was fed back to the root as a tool result.
        tool_results = _tool_results_sent_to_llm(node_config.mock)
        assert "No subtasks provided" in tool_results


@pytest.mark.asyncio
class TestSubtaskFailure:
    async def test_failed_subtask_is_marked_in_evidence(self):
        """When a delegated child fails, its slot in the parent's evidence
        carries a FAILED marker with the error.

        Each child runs an independent TestClient over the same event list,
        so every atom child consumes its first event identically. We make
        the first event a delegation attempt: atom children carry no
        ``solve_subtasks`` tool, so the call fails as an unknown tool —
        exactly the failure path ``_format_results`` renders as
        ``FAILED: <error>``.
        """
        node_config = TrackingConfig(
            TestConfig(
                _delegate_script("aspect A", mode="atom"),
                ModelResponse(ModelMessage("root fallback")),
            )
        )
        parent_config = TestConfig(
            ToolCallEvent(name="recursive_search", arguments='{"query": "research X"}'),
            ModelResponse(ModelMessage("final synthesis")),
        )
        stream = MemoryStream()
        parent = Agent(
            "parent",
            config=parent_config,
            tools=[recursive_search_tool(config=node_config, max_depth=1)],
        )

        body, events = await _run_search(parent, stream)

        assert body == "final synthesis"
        # root + 1 atom child. The atom child's first move is to try
        # solve_subtasks (from the shared event stream), which it does not
        # carry, so it fails.
        assert len(_starts(events)) == 2
        assert len(_failures(events)) == 1
        assert _failures(events)[0].agent_name == "node_atom_0"
        # The FAILED marker reached the root as part of the evidence.
        tool_results = _tool_results_sent_to_llm(node_config.mock)
        assert "FAILED" in tool_results
        assert "aspect A" in tool_results


@pytest.mark.asyncio
class TestNodeBudgetMidFlight:
    async def test_node_budget_can_be_spent_by_a_child(self):
        """max_nodes=2 with depth>0: the root (1) + one child (2), then the
        child tries to delegate further and gets NODE_BUDGET_EXHAUSTED — the
        budget sentinel is reachable by a child, not just the root."""
        node_config = TrackingConfig(
            TestConfig(
                _delegate_script("drill down"),
                ModelResponse(ModelMessage("solved it myself")),
            )
        )
        parent_config = TestConfig(
            ToolCallEvent(name="recursive_search", arguments='{"query": "research X"}'),
            ModelResponse(ModelMessage("final synthesis")),
        )
        stream = MemoryStream()
        parent = Agent(
            "parent",
            config=parent_config,
            tools=[
                recursive_search_tool(
                    config=node_config,
                    max_depth=3,
                    max_children=3,
                    max_nodes=2,
                )
            ],
        )

        body, events = await _run_search(parent, stream)

        assert body == "final synthesis"
        # root + exactly one child: budget stopped the tree at 2 nodes.
        assert len(_starts(events)) == 2
        assert len(_completions(events)) == 2
        assert _failures(events) == []
        # The child — not the root — received the budget sentinel.
        tool_results = _tool_results_sent_to_llm(node_config.mock)
        assert "NODE_BUDGET_EXHAUSTED" in tool_results
        assert "drill down" in tool_results

    async def test_partial_node_budget_within_one_call(self):
        """A single solve_subtasks call where the budget runs out halfway
        through the accepted list spawns the first children and reports
        the rest as 'Node budget reached' (not 'Fan-out cap reached').

        Setup: max_nodes=3 (root=1, then budget=2). Root delegates three
        subtasks; only two budget units remain, so the third is dropped
        mid-call and listed under the budget-reached notice.
        """
        node_config = TrackingConfig(
            TestConfig(
                _delegate_script("first", "second", "third"),
                ModelResponse(ModelMessage("level findings")),
            )
        )
        parent_config = TestConfig(
            ToolCallEvent(name="recursive_search", arguments='{"query": "research X"}'),
            ModelResponse(ModelMessage("final synthesis")),
        )
        stream = MemoryStream()
        parent = Agent(
            "parent",
            config=parent_config,
            tools=[
                recursive_search_tool(
                    config=node_config,
                    max_depth=1,
                    max_children=3,
                    max_nodes=3,
                )
            ],
        )

        body, events = await _run_search(parent, stream)

        assert body == "final synthesis"
        # root + 2 children: "third" was dropped by the budget inside the
        # single solve_subtasks call, not by the fan-out cap (cap=3).
        assert len(_starts(events)) == 3
        assert len(_completions(events)) == 3
        assert _failures(events) == []
        tool_results = _tool_results_sent_to_llm(node_config.mock)
        assert "Node budget reached" in tool_results
        assert "third" in tool_results
        # The fan-out cap message must not appear — the cap was 3.
        assert "Fan-out cap reached" not in tool_results


@pytest.mark.asyncio
class TestEvidenceTruncation:
    async def test_long_child_results_are_truncated(self):
        """max_evidence_chars caps how much of a child result flows up the
        tree, so deep swarms cannot blow up the parent's context."""
        node_config = TrackingConfig(
            TestConfig(
                _delegate_script("drill down"),
                ModelResponse(ModelMessage("A" * 5000)),
            )
        )
        parent_config = TestConfig(
            ToolCallEvent(name="recursive_search", arguments='{"query": "research X"}'),
            ModelResponse(ModelMessage("final synthesis")),
        )
        stream = MemoryStream()
        parent = Agent(
            "parent",
            config=parent_config,
            tools=[recursive_search_tool(config=node_config, max_depth=1, max_evidence_chars=100)],
        )

        body, events = await _run_search(parent, stream)

        assert body == "final synthesis"
        tool_results = _tool_results_sent_to_llm(node_config.mock)
        assert "[...truncated" in tool_results
        # The 5000-char child result was cut down to the 100-char budget.
        assert "A" * 200 not in tool_results


@pytest.mark.asyncio
class TestNodeBudget:
    async def test_node_budget_bounds_total_tree(self):
        """max_nodes caps the total tree size regardless of depth: once the
        budget is spent, further delegation returns a sentinel and nodes
        solve objectives themselves."""
        node_config = TrackingConfig(
            TestConfig(
                _delegate_script("drill down"),
                ModelResponse(ModelMessage("level findings")),
            )
        )
        parent_config = TestConfig(
            ToolCallEvent(name="recursive_search", arguments='{"query": "research X"}'),
            ModelResponse(ModelMessage("final synthesis")),
        )
        stream = MemoryStream()
        parent = Agent(
            "parent",
            config=parent_config,
            tools=[recursive_search_tool(config=node_config, max_depth=3, max_children=3, max_nodes=2)],
        )

        body, events = await _run_search(parent, stream)

        assert body == "final synthesis"
        # root + exactly one child: the budget stopped the tree at 2 nodes
        # even though depth would have allowed a deeper tree. Both finished
        # gracefully — the budget sentinel is degradation, not failure.
        assert len(_starts(events)) == 2
        assert len(_completions(events)) == 2
        assert _failures(events) == []
        # The budget-exhausted child was told to solve the objective itself.
        tool_results = _tool_results_sent_to_llm(node_config.mock)
        assert "NODE_BUDGET_EXHAUSTED" in tool_results


@pytest.mark.asyncio
class TestIdleTimeout:
    async def test_slow_but_progressing_search_survives(self):
        """The idle deadline kills only hung searches: a node that keeps
        producing events runs as long as it needs, even well past the
        deadline that would cancel a silent one."""

        @tool
        async def slow_search(ctx: Context, query: str) -> str:
            """Search that keeps emitting progress and finishes slowly."""
            for i in range(10):
                await ctx.send(TextInput(f"progress {i}"))
                await asyncio.sleep(0.05)
            return "slow findings"

        node_config = TestConfig(
            ToolCallEvent(name="slow_search", arguments='{"query": "X"}'),
            ModelResponse(ModelMessage("level findings")),
        )
        parent_config = TrackingConfig(
            TestConfig(
                ToolCallEvent(name="recursive_search", arguments='{"query": "research X"}'),
                ModelResponse(ModelMessage("final synthesis")),
            )
        )
        stream = MemoryStream()
        parent = Agent(
            "parent",
            config=parent_config,
            tools=[
                recursive_search_tool(
                    config=node_config,
                    tools=[slow_search],
                    max_depth=1,
                    timeout=0.15,  # far shorter than the search's 0.5s runtime
                )
            ],
        )

        body, events = await _run_search(parent, stream)

        assert body == "final synthesis"
        assert _failures(events) == []
        # The search finished normally and delivered its findings instead of
        # being killed for running longer than the idle deadline.
        tool_results = _tool_results_sent_to_llm(parent_config.mock)
        assert "level findings" in tool_results
        assert "timed out" not in tool_results.lower()


@pytest.mark.asyncio
class TestTimeout:
    async def test_root_deadline_reports_instead_of_hanging(self):
        """A root node that never returns is cancelled after the timeout; the
        caller gets a deadline report instead of waiting forever."""
        started = asyncio.Event()

        @tool
        async def hanging_search(query: str) -> str:
            """Search that blocks forever until cancelled."""
            started.set()
            await asyncio.Event().wait()
            return "late findings"

        node_config = TestConfig(
            ToolCallEvent(name="hanging_search", arguments='{"query": "X"}'),
        )
        parent_config = TrackingConfig(
            TestConfig(
                ToolCallEvent(name="recursive_search", arguments='{"query": "research X"}'),
                ModelResponse(ModelMessage("final synthesis")),
            )
        )
        stream = MemoryStream()
        parent = Agent(
            "parent",
            config=parent_config,
            tools=[recursive_search_tool(config=node_config, tools=[hanging_search], max_depth=1, timeout=0.2)],
        )

        body, events = await _run_search(parent, stream)

        assert body == "final synthesis"
        tool_results = _tool_results_sent_to_llm(parent_config.mock)
        assert "timed out" in tool_results.lower()
        # The hung node was cancelled, never completed or failed afterwards.
        assert len(_starts(events)) == 1
        assert _completions(events) == []
        assert _failures(events) == []

    async def test_hung_child_triggers_whole_search_deadline(self):
        """A child node that never returns cannot hang the search: when the
        overall deadline expires, the in-flight swarm is cancelled and a
        deadline report is returned to the caller."""
        started = asyncio.Event()

        @tool
        async def hanging_search(query: str) -> str:
            """Search that blocks forever until cancelled."""
            started.set()
            await asyncio.Event().wait()
            return "late findings"

        node_config = TestConfig(
            _delegate_script("drill down"),
            ToolCallEvent(name="hanging_search", arguments='{"query": "X"}'),
            ModelResponse(ModelMessage("level findings")),
        )
        parent_config = TrackingConfig(
            TestConfig(
                ToolCallEvent(name="recursive_search", arguments='{"query": "research X"}'),
                ModelResponse(ModelMessage("final synthesis")),
            )
        )
        stream = MemoryStream()
        parent = Agent(
            "parent",
            config=parent_config,
            tools=[
                recursive_search_tool(
                    config=node_config,
                    tools=[hanging_search],
                    max_depth=1,
                    timeout=0.2,
                )
            ],
        )

        body, events = await _run_search(parent, stream)

        assert body == "final synthesis"
        # The caller received the deadline report, not an eternal wait.
        tool_results = _tool_results_sent_to_llm(parent_config.mock)
        assert "timed out" in tool_results.lower()
        # The swarm was cancelled mid-flight: nothing completed or failed.
        assert _completions(events) == []
        assert _failures(events) == []


@pytest.mark.asyncio
class TestCancellation:
    async def test_parent_abort_cancels_in_flight_children(self):
        """Cancelling the parent unwinds the whole delegation chain: an abort
        mid-tool-call stops the running search node instead of leaving it
        running to completion in the background."""
        started = asyncio.Event()

        @tool
        async def hanging_search(query: str) -> str:
            """Search that blocks forever until the parent is cancelled."""
            started.set()
            await asyncio.Event().wait()
            return "late findings"

        node_config = TestConfig(
            ToolCallEvent(name="hanging_search", arguments='{"query": "X"}'),
        )
        parent_config = TestConfig(
            ToolCallEvent(name="recursive_search", arguments='{"query": "research X"}'),
            ModelResponse(ModelMessage("final synthesis")),
        )
        stream = MemoryStream()
        parent = Agent(
            "parent",
            config=parent_config,
            tools=[recursive_search_tool(config=node_config, tools=[hanging_search], max_depth=1)],
        )

        task = asyncio.create_task(parent.ask("research this", stream=stream))
        # Wait until the root node is actually inside the hanging search tool.
        await asyncio.wait_for(started.wait(), timeout=5)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The root node was aborted mid-tool-call: its run_task started but
        # never completed or failed, and nothing kept running afterwards.
        events = await _collect_tree(stream, stream.id)
        assert len(_starts(events)) == 1
        assert _completions(events) == []
        assert _failures(events) == []


@pytest.mark.asyncio
class TestRecursiveSearchAgent:
    async def test_factory_builds_agent_with_search_tool(self):
        config = TestConfig(ModelResponse(ModelMessage("done")))
        agent = recursive_search_agent(config=config)

        assert isinstance(agent, Agent)
        assert agent.name == "recursive_researcher"
        assert [t.schema.function.name for t in agent.tools] == ["recursive_search"]

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"max_depth": -1}, "max_depth"),
            ({"max_children": 0}, "max_children"),
            ({"timeout": 0}, "timeout"),
            (
                {"timeout": 30, "stream": lambda _agent, _ctx: MemoryStream()},
                "cannot be combined",
            ),
            ({"max_nodes": 0}, "max_nodes"),
            ({"max_evidence_chars": 0}, "max_evidence_chars"),
        ],
    )
    async def test_validates_arguments(self, kwargs, match):
        """recursive_search_agent mirrors recursive_search_tool's argument
        validation — every ValueError raised by the tool factory must also
        be raised by the agent factory before it builds an Agent."""
        config = TestConfig(ModelResponse(ModelMessage("done")))
        with pytest.raises(ValueError, match=match):
            recursive_search_agent(config=config, **kwargs)

    async def test_agent_end_to_end(self):
        node_config = TestConfig(
            _delegate_script("drill down"),
            ModelResponse(ModelMessage("level findings")),
        )
        agent_config = TestConfig(
            ToolCallEvent(name="recursive_search", arguments='{"query": "research X"}'),
            ModelResponse(ModelMessage("agent answer")),
        )
        # The outer agent and the swarm nodes use separate configs.
        agent = recursive_search_agent(config=node_config, max_depth=1)
        agent.config = agent_config

        stream = MemoryStream()
        reply = await agent.ask("research X", stream=stream)

        assert reply.body == "agent answer"
        events = await _collect_tree(stream, stream.id)
        assert len(_starts(events)) == 2  # root + one child level
        assert _failures(events) == []
