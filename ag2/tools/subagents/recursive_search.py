# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Recursive fan-out for parallel research decomposition.

Adapted from WebSwarm: Recursive Multi-Agent Orchestration for Deep-and-Wide
Web Search (https://arxiv.org/abs/2607.08662v1). WebSwarm builds a delegation
tree at inference time: each node couples a local objective with a search
*mode* and either solves the objective itself or delegates child nodes whose
results flow back up as evidence for further expansion, revision, or
aggregation.

This module ports the paper's ``atom`` and ``wide`` verbs — the recursive
fan-out primitive — onto AG2's subagent primitives:

- Every non-atom search node is an :class:`~ag2.agent.Agent` carrying
  the caller's search tools plus a self-referential ``solve_subtasks`` tool,
  so every node can spawn structurally identical children — true recursion,
  executed by :func:`~ag2.tools.subagents.run_task.run_task` with
  ``asyncio.gather`` fan-out (in place of the paper's ThreadPoolExecutor).
- Search modes (``atom`` / ``wide``) are enum values that gate behavior in
  code: ``atom`` nodes receive no delegation tool at all, and ``wide`` nodes
  may fan out up to ``max_children``.
- The depth budget is an int threaded through tool closures. When a node's
  budget is exhausted, its ``solve_subtasks`` returns a downgrade sentinel
  instructing the node to solve the objectives itself (atom behavior)
  instead of spawning children.

Substitutions vs. the paper: Serper/Jina are replaced by caller-supplied
search/fetch tools (e.g. ``DuckDuckSearchTool`` / ``WebFetchTool``).

Intentionally out of scope for this port (candidates for follow-up modules):

- The paper's ``deep`` verb (proposer/verifier architecture with structurally
  independent source-isolated verification and survived/weakened/refuted
  verdicts).
- The paper's ``entity_collect`` verb (schema inference, multi-strategy
  parallel sampling, split-verify-merge validation pipeline).
- Web-probing scout (pre-expansion topology finding) and cross-sibling
  experience reuse (scout subset → experience transfer for wide batches).
"""

import asyncio
import time
from collections.abc import Coroutine, Iterable
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ag2.annotations import Context
from ag2.events import BaseEvent
from ag2.history import Storage
from ag2.stream import MemoryStream
from ag2.tools.final import FunctionTool, tool
from ag2.tools.tool import Tool

from .run_task import TaskResult, run_task
from .subagent_tool import StreamFactory

if TYPE_CHECKING:
    from ag2.agent import Agent
    from ag2.config import ModelConfig
    from ag2.context import StreamId


_DEFAULT_RESEARCHER_PROMPT = (
    "You are a deep research agent. For complex queries, delegate "
    "sub-objectives with recursive_search, then synthesize the evidence "
    "returned by the swarm into a cited answer."
)


def _agent_cls() -> "type[Agent]":
    """Resolve the Agent class lazily.

    ``ag2.agent`` is mid-initialization when this module is first imported
    (``ag2/agent.py`` imports ``ag2.tools.subagents.run_task``, which runs
    this package's ``__init__``), so a module-level import would deadlock.
    Deferring the import to call time keeps the package import order safe.
    """
    from ag2.agent import Agent

    return Agent


class SearchMode(str, Enum):
    """How a search node organizes its search and collaboration.

    Mirrors WebSwarm's ``atom`` and ``wide`` verbs. The mode is enforced
    structurally, not just described in prompts: ``ATOM`` nodes are built
    without the ``solve_subtasks`` tool, and ``WIDE`` nodes may fan out up
    to ``max_children``.
    """

    ATOM = "atom"
    WIDE = "wide"


class SubtaskSpec(BaseModel):
    """A single child node to spawn: its local objective and search mode."""

    objective: str
    mode: SearchMode = SearchMode.ATOM
    context: str = ""


_DEPTH_DOWNGRADE_SENTINEL = "DELEGATION_BUDGET_EXHAUSTED"
_NODE_BUDGET_SENTINEL = "NODE_BUDGET_EXHAUSTED"

_MODE_GUIDANCE: dict[SearchMode, str] = {
    SearchMode.ATOM: (
        "Solve the objective directly with your search tools, ReAct-style: "
        "issue focused queries, read the evidence, and answer. You cannot "
        "delegate — reply with your findings and cite the key evidence."
    ),
    SearchMode.WIDE: (
        "Cover the objective broadly: identify 2-3 independent aspects and "
        "delegate them as homogeneous child nodes via solve_subtasks, then "
        "aggregate their evidence into one answer."
    ),
}

_READINESS_LOOP = """
Decide whether you are Ready to answer the objective with your own search
tools. If not Ready, decompose it and call solve_subtasks with self-contained
child objectives; you may call it multiple times. When child results come
back, expand, revise, or aggregate them — and only reply once you are Ready.
Your reply is returned to the parent node as evidence, so make it
self-contained and cite what you found."""


def _delegation_cap(mode: SearchMode, max_children: int) -> int:
    """Fan-out cap for a node's ``solve_subtasks``; 0 means no delegation."""
    if mode is SearchMode.ATOM:
        return 0
    return max_children


class _NodeBudget:
    """Shared cap on how many nodes a single search may spawn.

    The depth budget bounds the *depth* of the tree and the fan-out cap
    bounds one delegation *call*, but a node is allowed to call
    ``solve_subtasks`` repeatedly, so neither alone bounds the *total* tree
    size. This counter is threaded through the tool closures and every node
    spends one unit at creation, giving a hard ceiling on LLM calls and cost.
    """

    def __init__(self, remaining: int) -> None:
        self.remaining = remaining

    def spend(self) -> bool:
        """Spend one node from the budget; ``False`` when it is exhausted."""
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True


class _ActivityStorage:
    """Storage proxy that records when the last event was written.

    Every node stream in a search is created with the same shared storage
    (``MemoryStream(storage=...)`` chains down the delegation tree), so a
    single proxy installed at the root covers the whole swarm: any model
    call, tool result, or message bumps ``last_activity``. This is what lets
    the idle timeout distinguish a *hung* search (no events) from a *slow*
    one (events keep arriving).
    """

    __slots__ = ("_inner", "_last_activity")

    def __init__(self, inner: Storage, last_activity: list[float]) -> None:
        self._inner = inner
        self._last_activity = last_activity

    async def save_event(self, event: BaseEvent, context: Context) -> None:
        self._last_activity[0] = time.monotonic()
        await self._inner.save_event(event, context)

    async def get_history(self, stream_id: "StreamId") -> Iterable[BaseEvent]:
        return await self._inner.get_history(stream_id)

    async def set_history(self, stream_id: "StreamId", events: Iterable[BaseEvent]) -> None:
        await self._inner.set_history(stream_id, events)

    async def drop_history(self, stream_id: "StreamId") -> None:
        await self._inner.drop_history(stream_id)


async def _run_with_idle_timeout(
    coro: Coroutine[Any, Any, TaskResult],
    timeout: float,
    last_activity: list[float],
) -> TaskResult:
    """Run a search delegation with an idle deadline.

    The search is cancelled only when *no events* have been written to its
    shared stream storage for ``timeout`` seconds — a search that is slow
    but keeps making progress is never interrupted. On an idle timeout the
    delegation is cancelled and ``TimeoutError`` is raised; any other
    cancellation (an external abort) propagates untouched.
    """
    task: asyncio.Task[TaskResult] = asyncio.create_task(coro)
    timed_out = False
    try:
        while True:
            idle_for = time.monotonic() - last_activity[0]
            if idle_for >= timeout:
                timed_out = True
                task.cancel()
                break
            # ``asyncio.wait`` returns the moment the search finishes, and
            # polls finely near the deadline, so a fast search costs no
            # extra wait and a short test timeout fires promptly.
            done, _ = await asyncio.wait({task}, timeout=min(1.0, max(0.05, timeout - idle_for)))
            if task in done:
                break
    finally:
        if not task.done():
            task.cancel()
    try:
        return await task
    except asyncio.CancelledError:
        if timed_out:
            raise TimeoutError(f"no progress for {timeout}s") from None
        raise


def _node_prompt(mode: SearchMode, cap: int) -> str:
    lines = [
        f"You are a {mode.value} web search node in a recursive search swarm.",
        "",
        _MODE_GUIDANCE[mode].strip(),
    ]
    if cap > 0:
        lines.append(_READINESS_LOOP.strip())
    return "\n".join(lines)


def _make_search_node(
    name: str,
    *,
    mode: SearchMode,
    config: "ModelConfig",
    search_tools: tuple[Tool, ...],
    depth: int,
    max_children: int,
    stream: StreamFactory | None,
    budget: _NodeBudget | None,
    max_evidence_chars: int | None,
) -> "Agent":
    """Build one search node: an Agent with search tools and, unless the mode
    is atom, a self-referential ``solve_subtasks`` tool whose closure carries
    the remaining depth budget, this mode's fan-out cap, and the shared node
    budget."""
    cap = _delegation_cap(mode, max_children)
    tools: list[Tool] = list(search_tools)
    if cap > 0:
        tools.append(
            _make_solve_subtasks_tool(
                config=config,
                search_tools=search_tools,
                depth=depth,
                max_children=cap,
                stream=stream,
                budget=budget,
                max_evidence_chars=max_evidence_chars,
            )
        )
    return _agent_cls()(
        name,
        prompt=_node_prompt(mode, cap),
        config=config,
        tools=tools,
    )


def _format_results(
    specs: list[SubtaskSpec],
    results: list[TaskResult],
    dropped: list[SubtaskSpec],
    max_evidence_chars: int | None,
) -> str:
    lines = []
    for spec, result in zip(specs, results, strict=True):
        if result.completed:
            body = _truncate(result.result or "(no result)", max_evidence_chars)
            lines.append(f"## [{spec.mode.value}] {spec.objective}\n{body}")
        else:
            lines.append(f"## [{spec.mode.value}] {spec.objective}\nFAILED: {result.error}")
    evidence = "\n\n".join(lines)
    if dropped:
        skipped = "\n".join(f"- {s.objective}" for s in dropped)
        evidence += (
            f"\n\nFan-out cap reached: {len(dropped)} requested subtask(s) were not "
            f"delegated. Solve or re-delegate them yourself if still needed:\n{skipped}"
        )
    return evidence


def _truncate(text: str, limit: int | None) -> str:
    """Cap a child node's result so accumulated evidence stays bounded."""
    if limit is not None and len(text) > limit:
        return text[:limit] + f"\n[...truncated at {limit} chars]"
    return text


def _make_solve_subtasks_tool(
    *,
    config: "ModelConfig",
    search_tools: tuple[Tool, ...],
    depth: int,
    max_children: int,
    stream: StreamFactory | None,
    budget: _NodeBudget | None,
    max_evidence_chars: int | None,
) -> FunctionTool:
    """Create the self-referential delegation tool every non-atom node carries.

    Spawns one structurally identical child node per accepted subtask with a
    decremented depth budget. When the depth budget is exhausted the tool
    spawns nothing and returns a downgrade sentinel, so the node falls back
    to atom behavior (solve it yourself). When the shared node budget is
    exhausted the same fallback applies: no further children may be spawned."""

    @tool(
        name="solve_subtasks",
        description=(
            "Delegate self-contained sub-objectives to child search nodes. "
            "Each subtask names its own mode: atom (direct fact lookup) or "
            "wide (parallel aspect coverage). At most "
            f"{max_children} subtasks are accepted per call; extras are "
            "dropped. Child results return here as evidence you can expand, "
            "revise, or aggregate."
        ),
    )
    async def solve_subtasks(ctx: Context, subtasks: list[SubtaskSpec]) -> str:
        if not subtasks:
            return "No subtasks provided. Pass at least one subtask with an objective and a mode."

        if depth <= 0:
            objectives = "\n".join(f"- {s.objective}" for s in subtasks)
            return (
                f"{_DEPTH_DOWNGRADE_SENTINEL}: the delegation depth cap is "
                "reached, so no child nodes were spawned. Downgrade to atom "
                "mode and solve these objectives directly with your own "
                f"search tools:\n{objectives}"
            )

        accepted = subtasks[:max_children]
        dropped = subtasks[max_children:]
        children = []
        unspawned = []
        for index, spec in enumerate(accepted):
            if budget is not None and not budget.spend():
                unspawned.append(spec)
                continue
            children.append(
                _make_search_node(
                    f"node_{spec.mode.value}_{index}",
                    mode=spec.mode,
                    config=config,
                    search_tools=search_tools,
                    depth=depth - 1,
                    max_children=max_children,
                    stream=stream,
                    budget=budget,
                    max_evidence_chars=max_evidence_chars,
                )
            )

        if not children:
            objectives = "\n".join(f"- {s.objective}" for s in accepted)
            return (
                f"{_NODE_BUDGET_SENTINEL}: the node budget is exhausted, so no "
                "child nodes can be spawned. Solve these objectives directly "
                f"with your own search tools:\n{objectives}"
            )

        results: list[TaskResult] = await asyncio.gather(
            *(
                run_task(
                    child,
                    spec.objective,
                    parent_context=ctx,
                    context=spec.context,
                    stream=stream(child, ctx) if stream else None,
                )
                for child, spec in zip(children, accepted, strict=True)
            )
        )

        evidence = _format_results(accepted[: len(children)], list(results), dropped, max_evidence_chars)
        if unspawned:
            skipped = "\n".join(f"- {s.objective}" for s in unspawned)
            evidence += (
                f"\n\nNode budget reached: {len(unspawned)} subtask(s) were not "
                f"delegated. Solve or re-delegate them yourself if still needed:\n{skipped}"
            )
        return evidence

    return solve_subtasks


def recursive_search_tool(
    *,
    config: "ModelConfig",
    name: str = "recursive_search",
    search_mode: SearchMode = SearchMode.WIDE,
    tools: Iterable[Tool] = (),
    max_depth: int = 3,
    max_children: int = 3,
    stream: StreamFactory | None = None,
    timeout: float | None = 300,
    max_nodes: int | None = None,
    max_evidence_chars: int | None = 4000,
) -> FunctionTool:
    """Create a recursive deep-and-wide search tool.

    The returned tool runs a root search node for the given query. The root —
    and every non-atom node beneath it — carries a self-referential
    ``solve_subtasks`` tool, so the swarm grows its own delegation tree at
    inference time: nodes solve their objective or delegate child nodes whose
    results flow back up as evidence.

    Args:
        config: LLM config shared by every search node (required — nodes are
            plain agents and cannot inherit a model from the calling agent).
        name: Tool name (default: "recursive_search").
        search_mode: Mode of the root node; children pick their own mode per
            subtask. Default "wide" (parallel aspect coverage).
        tools: Search/fetch tools every node uses to actually search the web
            (e.g. ``DuckDuckSearchTool``, ``WebFetchTool`` — the paper's
            Serper/Jina equivalents). Pass mocks here in tests.
        max_depth: Delegation depth budget. A node whose budget is exhausted
            gets a downgrade sentinel from ``solve_subtasks`` and must solve
            objectives itself. Default 3.
        max_children: Fan-out cap per delegation call (2-3 recommended for
            wide mode per the paper; deep mode is capped at 2 regardless).
        stream: Optional stream factory for per-node persistent history
            (e.g. ``persistent_stream()``).
        timeout: Idle deadline in seconds. The search is cancelled only
            when *no events* are written for ``timeout`` seconds — a search
            that is slow but keeps making progress is never interrupted,
            only a hung one. Default 300; pass ``None`` to disable. Cannot
            be combined with ``stream=`` (idle tracking needs the
            shared-history stream).
        max_nodes: Hard ceiling on the total number of nodes the search may
            spawn (including the root). Nodes may call ``solve_subtasks``
            repeatedly, so depth and fan-out alone do not bound tree size;
            this budget does. When exhausted, further delegation returns a
            sentinel and the node solves the objectives itself. Default
            ``None`` (no cap).
        max_evidence_chars: Per-child cap on how much of a node's result is
            included in the evidence passed up the tree, so deep swarms do
            not blow up the parent's context. Default 4000; pass ``None``
            to disable.
    Returns:
        A FunctionTool that can be added to an agent's tools.

    Example:
        ```python
        from ag2 import Agent
        from ag2.tools.search import DuckDuckSearchTool
        from ag2.tools.subagents import recursive_search_tool

        agent = Agent(
            "researcher",
            config=config,
            tools=[recursive_search_tool(config=config, tools=[DuckDuckSearchTool()], timeout=120)],
        )
        ```
    """
    search_tools = tuple(tools)
    if max_depth < 0:
        raise ValueError(f"max_depth must be >= 0, got {max_depth}")
    if max_children < 1:
        raise ValueError(f"max_children must be >= 1, got {max_children}")
    if timeout is not None and timeout <= 0:
        raise ValueError(f"timeout must be > 0, got {timeout}")
    if timeout is not None and stream is not None:
        raise ValueError("timeout= and stream= cannot be combined: idle tracking needs the shared-history stream")
    if max_nodes is not None and max_nodes < 1:
        raise ValueError(f"max_nodes must be >= 1, got {max_nodes}")
    if max_evidence_chars is not None and max_evidence_chars < 1:
        raise ValueError(f"max_evidence_chars must be >= 1, got {max_evidence_chars}")
    budget = _NodeBudget(max_nodes) if max_nodes is not None else None

    @tool(
        name=name,
        description=(
            "Recursively research a complex query with a swarm of search "
            "nodes. The root node decomposes the query, delegates "
            "sub-objectives to child nodes (which may delegate further), and "
            "aggregates the evidence that flows back up into a final answer."
        ),
    )
    async def recursive_search(ctx: Context, query: str, context: str = "") -> str:
        """Research ``query`` with recursive deep-and-wide search and return
        the root node's synthesis of all evidence gathered by the swarm."""
        root = _make_search_node(
            f"{name}_root",
            mode=search_mode,
            config=config,
            search_tools=search_tools,
            depth=max_depth,
            max_children=max_children,
            stream=stream,
            budget=budget,
            max_evidence_chars=max_evidence_chars,
        )
        if budget is not None:
            budget.spend()  # the root counts against the node budget
        if timeout is None:
            result = await run_task(
                root,
                query,
                parent_context=ctx,
                context=context,
                stream=stream(root, ctx) if stream else None,
            )
        else:
            # Install an activity proxy on the shared stream storage: every
            # event any node in the swarm writes bumps the idle clock, so a
            # slow-but-progressing search is never interrupted. Only a search
            # with no events at all for ``timeout`` seconds is cancelled.
            last_activity = [time.monotonic()]
            # `ctx.stream` is typed as the Stream protocol, which does not
            # declare `history`; run_task() accesses the same storage the
            # same way (the shared history storage is what idle tracking
            # observes).
            task_stream = MemoryStream(
                storage=_ActivityStorage(ctx.stream.history.storage, last_activity)  # type: ignore[attr-defined]
            )
            try:
                result = await _run_with_idle_timeout(
                    run_task(root, query, parent_context=ctx, context=context, stream=task_stream),
                    timeout,
                    last_activity,
                )
            except TimeoutError:
                return f"Recursive search timed out: no progress for {timeout}s."
        if not result.completed:
            return f"Recursive search failed: {result.error}"
        return result.result or "No findings returned."

    return recursive_search


def recursive_search_agent(
    name: str = "recursive_researcher",
    *,
    config: "ModelConfig",
    search_mode: SearchMode = SearchMode.WIDE,
    tools: Iterable[Tool] = (),
    max_depth: int = 3,
    max_children: int = 3,
    stream: StreamFactory | None = None,
    prompt: str = _DEFAULT_RESEARCHER_PROMPT,
    timeout: float | None = 300,
    max_nodes: int | None = None,
    max_evidence_chars: int | None = 4000,
) -> "Agent":
    """Create an agent with recursive search capabilities pre-configured.

    Convenience factory that builds an Agent carrying ``recursive_search_tool``
    with the given configuration.

    Args:
        name: Agent name (default: "recursive_researcher").
        config: LLM config shared by every search node.
        search_mode: Mode of the root search node (default: wide).
        tools: Search/fetch tools every node uses to actually search.
        max_depth: Delegation depth budget (default: 3).
        max_children: Fan-out cap per delegation call (default: 3).
        stream: Optional stream factory for per-node persistent history.
        prompt: System prompt for the researcher agent.
        timeout: Idle deadline in seconds: the search is cancelled only when
            no events arrive for ``timeout`` seconds (default 300; pass
            ``None`` to disable). Cannot be combined with ``stream=``.
        max_nodes: Hard ceiling on total nodes the search may spawn
            (default: no cap).
        max_evidence_chars: Per-child cap on evidence size (default 4000;
            pass ``None`` to disable).

    Example:
        ```python
        from ag2.tools.subagents import recursive_search_agent

        agent = recursive_search_agent(config=config)
        answer = await agent.ask("What are the latest advances in quantum computing?")
        ```
    """
    if max_depth < 0:
        raise ValueError(f"max_depth must be >= 0, got {max_depth}")
    if max_children < 1:
        raise ValueError(f"max_children must be >= 1, got {max_children}")
    if timeout is not None and timeout <= 0:
        raise ValueError(f"timeout must be > 0, got {timeout}")
    if timeout is not None and stream is not None:
        raise ValueError("timeout= and stream= cannot be combined: idle tracking needs the shared-history stream")
    if max_nodes is not None and max_nodes < 1:
        raise ValueError(f"max_nodes must be >= 1, got {max_nodes}")
    if max_evidence_chars is not None and max_evidence_chars < 1:
        raise ValueError(f"max_evidence_chars must be >= 1, got {max_evidence_chars}")
    return _agent_cls()(
        name,
        prompt=prompt,
        config=config,
        tools=[
            recursive_search_tool(
                config=config,
                search_mode=search_mode,
                tools=tools,
                max_depth=max_depth,
                max_children=max_children,
                stream=stream,
                timeout=timeout,
                max_nodes=max_nodes,
                max_evidence_chars=max_evidence_chars,
            )
        ],
    )
