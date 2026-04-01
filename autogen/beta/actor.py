# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Actor — an autonomous agent with observers, signals, task spawning, and agent harness.

Actor extends Agent with:
1. Observer management — attach/detach observers that monitor the event stream
2. Signal injection — observer signals are delivered via SignalPolicy
3. Task spawning — spawn_task/spawn_tasks tools for delegating subtasks
4. Agent Harness — persistent knowledge, context assembly, compaction, aggregation

Works standalone (no Hub required). Optionally registers with a Hub
for cross-actor discovery and delegation.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable, Sequence
from contextlib import suppress
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from autogen.beta.agent import Agent, AgentReply, PromptType
from autogen.beta.annotations import Context
from autogen.beta.config import LLMClient, ModelConfig
from autogen.beta.events import BaseEvent, ModelMessageChunk, ModelResponse
from autogen.beta.hitl import HumanHook
from autogen.beta.middleware.base import BaseMiddleware, LLMCall, MiddlewareFactory
from autogen.beta.stream import MemoryStream
from autogen.beta.tools.final import tool
from autogen.beta.tools.tool import Tool
from autogen.beta.types import Omittable, omit

if TYPE_CHECKING:
    from autogen.beta.response import ResponseProto

from dataclasses import dataclass

from .aggregate import AggregateStrategy, AggregateTrigger
from .assembly import AssemblerMiddleware, AssemblyPolicy
from .compact import CompactStrategy, CompactTrigger
from .knowledge import DefaultBootstrap, EventLogWriter, KnowledgeStore, StoreBootstrap
from .network.events import (
    AggregationCompleted,
    CompactionCompleted,
    ObserverCompleted,
    ObserverStarted,
    TaskProgress,
    TaskRequest,
    TaskResult,
)
from .observer import Observer
from .policies.conversation import ConversationPolicy

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeConfig:
    """Groups knowledge-related Actor parameters."""

    store: KnowledgeStore
    compact: CompactStrategy | None = None
    compact_trigger: CompactTrigger | None = None
    aggregate: AggregateStrategy | None = None
    aggregate_trigger: AggregateTrigger | None = None
    bootstrap: StoreBootstrap | None = None


@dataclass
class TaskConfig:
    """Groups task-spawning Actor parameters."""

    config: ModelConfig | None = None
    prompt: str = "You are a task agent. Complete the assigned task thoroughly and concisely. Return only the result."


class Actor(Agent):
    """An autonomous agent with observers, signals, task spawning, and agent harness.

    Actor extends Agent. It delegates to ``super()._execute()`` rather than
    reimplementing the event loop. Actor concerns (observers, signals, tasks,
    knowledge, assembly, compaction, aggregation) are injected via additional
    tools and middleware.

    Parameters
    ----------
    name:
        Agent display name.
    prompt:
        System prompt(s).
    config:
        LLM configuration for this actor.
    observers:
        Observers that monitor the event stream.
    task_config:
        LLM config for spawned task sub-agents. Falls back to ``config``.
    task_prompt:
        System prompt for task sub-agents.
    knowledge_store:
        Persistent knowledge store for this actor.
    bootstrap:
        Store bootstrapping strategy. Default: DefaultBootstrap.
    assembly:
        Assembly policies controlling what the LLM sees. Default: [ConversationPolicy()].
    compact:
        Compaction strategy for reducing stream history.
    compact_trigger:
        When to trigger compaction.
    aggregate:
        Aggregation strategy for building long-term knowledge.
    aggregate_trigger:
        When to trigger aggregation.
    tools:
        Tools available to this actor.
    middleware:
        Middleware factories.
    """

    def __init__(
        self,
        name: str,
        prompt: PromptType | Iterable[PromptType] = (),
        *,
        config: ModelConfig | None = None,
        observers: Iterable[Observer] = (),
        knowledge: KnowledgeConfig | None = None,
        tasks: TaskConfig | None = None,
        assembly: Iterable[AssemblyPolicy] = (),
        hitl_hook: HumanHook | None = None,
        # Standard Agent params
        tools: Iterable[Callable[..., Any] | Tool] = (),
        middleware: Iterable[MiddlewareFactory] = (),
        dependencies: dict[Any, Any] | None = None,
        variables: dict[Any, Any] | None = None,
    ) -> None:
        super().__init__(
            name,
            prompt,
            config=config,
            hitl_hook=hitl_hook,
            tools=tools,
            middleware=middleware,
            dependencies=dependencies,
            variables=variables,
        )
        self._observers: list[Observer] = list(observers)

        # Task spawning
        tc = tasks or TaskConfig(config=config)
        self._task_config: ModelConfig | None = tc.config or config
        self._task_prompt = tc.prompt

        # Knowledge & memory
        kc = knowledge
        self._knowledge_store = kc.store if kc else None
        self._bootstrap = kc.bootstrap if kc else None
        self._policies: list[AssemblyPolicy] = list(assembly) if assembly else [ConversationPolicy()]
        self._compact_strategy = kc.compact if kc else None
        self._compact_trigger = (kc.compact_trigger if kc and kc.compact_trigger else CompactTrigger())
        self._aggregate_strategy = kc.aggregate if kc else None
        self._aggregate_trigger = (kc.aggregate_trigger if kc and kc.aggregate_trigger else AggregateTrigger())

        # Validate policy ordering
        warnings = AssemblerMiddleware.validate_order(self._policies)
        for w in warnings:
            logger.warning("Assembly policy ordering: %s", w)

    # ------------------------------------------------------------------
    # Observer management
    # ------------------------------------------------------------------

    def add_observer(self, observer: Observer) -> None:
        """Register an observer (before calling ask())."""
        self._observers.append(observer)

    # ------------------------------------------------------------------
    # Knowledge and memory tools
    # ------------------------------------------------------------------

    def _build_knowledge_tool(self) -> list[Tool]:
        """Build the knowledge tool (typed action group)."""
        store = self._knowledge_store

        @tool
        async def knowledge(action: str, path: str = "/", content: str = "") -> str:
            """Manage your knowledge store.

            Actions:
                read   - Read file at path.
                write  - Write content to path.
                list   - List entries at path.
                delete - Delete file at path.
            """
            if action == "read":
                result = await store.read(path)  # type: ignore[union-attr]
                return result if result is not None else f"Not found: {path}"

            elif action == "write":
                if not content:
                    return "Error: content is required for write action."
                await store.write(path, content)  # type: ignore[union-attr]
                return f"Written to {path}"

            elif action == "list":
                entries = await store.list(path)  # type: ignore[union-attr]
                if not entries:
                    return f"Empty: {path}"
                # Check for SKILL.md
                skill_path = f"{path.rstrip('/')}/SKILL.md"
                skill = await store.read(skill_path)  # type: ignore[union-attr]
                listing = "\n".join(entries)
                if skill:
                    listing = f"{skill}\n---\n{listing}"
                return listing

            elif action == "delete":
                await store.delete(path)  # type: ignore[union-attr]
                return f"Deleted: {path}"

            else:
                return f"Unknown action: {action}. Available: read, write, list, delete."

        return [knowledge]

    def _build_memory_tool(self) -> list[Tool]:
        """Build the memory management tool (typed action group)."""
        actor = self

        @tool
        async def memory(action: str, ctx: Context) -> str:
            """Manage your memory and context.

            Actions:
                compact   - Compact conversation history to free context space.
                summarize - Aggregate current context into knowledge store.
            """
            if action == "compact":
                if not actor._compact_strategy:
                    return "Compaction not configured."
                events = list(await ctx.stream.history.get_events())
                compacted = await actor._compact_strategy.compact(events, ctx, actor._knowledge_store)
                await ctx.stream.history.replace(compacted)
                return f"Compacted: {len(events)} events -> {len(compacted)} events."

            elif action == "summarize":
                if not actor._aggregate_strategy or not actor._knowledge_store:
                    return "Aggregation not configured."
                events = list(await ctx.stream.history.get_events())
                await actor._aggregate_strategy.aggregate(events, ctx, actor._knowledge_store)
                return "Knowledge store updated."

            else:
                return f"Unknown action: {action}. Available: compact, summarize."

        return [memory]

    # ------------------------------------------------------------------
    # Task spawning tools
    # ------------------------------------------------------------------

    def _build_subtask_tools(self) -> list[Tool]:
        """Create run_subtask and run_subtasks tools."""
        actor = self

        @tool
        async def run_subtask(task: str, ctx: Context) -> str:
            """Run a subtask agent to handle isolated compute work autonomously.

            Use this when a sub-problem can be researched or solved
            independently. The subtask agent runs with its own LLM context
            and returns the result.
            """
            return await actor._run_task(task, ctx)

        @tool
        async def run_subtasks(ctx: Context, tasks: list[str], parallel: bool = True) -> str:
            """Run multiple subtask agents at once.

            Args:
                ctx: The context to run the tasks in.
                tasks: List of subtask descriptions.
                parallel: Run concurrently (default True) or sequentially.
            """
            if parallel:
                results = await asyncio.gather(
                    *(actor._run_task(t, ctx) for t in tasks),
                    return_exceptions=True,
                )
                # Convert exceptions to error strings so partial results are preserved
                results = [r if not isinstance(r, BaseException) else f"Error: {r}" for r in results]
            else:
                results = []
                for t in tasks:
                    try:
                        results.append(await actor._run_task(t, ctx))
                    except Exception as e:
                        results.append(f"Error: {e}")

            parts = []
            for task_desc, result in zip(tasks, results):
                parts.append(f"## {task_desc}\n\n{result}")
            return "\n\n---\n\n".join(parts)

        return [run_subtask, run_subtasks]

    async def _run_task(self, task: str, ctx: Context) -> str:
        """Run a single task sub-agent and emit lifecycle events."""
        task_name = f"task-{uuid4().hex[:8]}"

        await ctx.send(TaskRequest(task=task, task_name=task_name))

        sat_stream = MemoryStream()
        parent_ctx = ctx

        async def _bridge_chunks(chunk: ModelMessageChunk) -> None:
            await parent_ctx.send(TaskProgress(task_name=task_name, content=chunk.content))

        chunk_sub = sat_stream.where(ModelMessageChunk).subscribe(_bridge_chunks)

        task_agent = Agent(
            task_name,
            prompt=self._task_prompt,
            config=self._task_config,
        )
        try:
            reply = await task_agent.ask(task, stream=sat_stream)
        finally:
            sat_stream.unsubscribe(chunk_sub)

        result_text = reply.body or ""
        usage = reply.response.usage if reply.response else {}

        await ctx.send(
            TaskResult(
                task=task,
                task_name=task_name,
                result=result_text,
                usage=usage,
            )
        )
        return result_text

    # ------------------------------------------------------------------
    # Core execution override
    # ------------------------------------------------------------------

    async def _execute(
        self,
        event: BaseEvent,
        *,
        context: Context,
        client: LLMClient,
        additional_tools: Iterable[Tool] = (),
        additional_middleware: Iterable[MiddlewareFactory] = (),
        response_schema: Omittable[ResponseProto[Any] | type | None] = omit,
    ) -> AgentReply:
        subtask_tools = self._build_subtask_tools()

        # Bootstrap knowledge store on first use (write sentinel first
        # to prevent duplicate bootstrap from concurrent _execute calls)
        if self._knowledge_store:
            if not await self._knowledge_store.exists("/.initialized"):
                await self._knowledge_store.write("/.initialized", self.name)
                bootstrap = self._bootstrap or DefaultBootstrap()
                await bootstrap.bootstrap(self._knowledge_store, self.name)

        # Build harness tools
        knowledge_tools = self._build_knowledge_tool() if self._knowledge_store else []
        memory_tools = (
            self._build_memory_tool() if (self._compact_strategy or self._aggregate_strategy) else []
        )

        # Make knowledge store available via DI
        if self._knowledge_store:
            context.dependencies[KnowledgeStore] = self._knowledge_store

        # Attach observers
        for obs in self._observers:
            obs.attach(context.stream, context)
            await context.send(ObserverStarted(name=obs.name))

        # Build middleware chain:
        # 1. AssemblerMiddleware (outermost) — assembles context
        #    (AlertPolicy in the assembly chain handles observer alerts)
        # 2. HaltCheckMiddleware — catches HaltEvent from AlertPolicy, short-circuits
        # 3. CompactionMiddleware — triggers compaction after turns
        # 4. AggregationMiddleware — triggers aggregation after turns
        # 5. User-provided middleware
        # 6. LLM client call (innermost)
        assembler_mw = _AssemblerMiddlewareFactory(self._policies)
        halt_mw = _HaltCheckMiddlewareFactory()

        harness_middleware: list[MiddlewareFactory] = [assembler_mw, halt_mw]

        # Compaction middleware
        if self._compact_strategy:
            harness_middleware.append(
                _CompactionMiddlewareFactory(
                    self.name,
                    self._compact_strategy,
                    self._knowledge_store,
                    self._compact_trigger,
                )
            )

        # Aggregation middleware (for every_n_turns / every_n_events triggers)
        if self._aggregate_strategy and self._knowledge_store:
            trigger = self._aggregate_trigger
            if trigger.every_n_turns > 0 or trigger.every_n_events > 0:
                harness_middleware.append(
                    _AggregationMiddlewareFactory(
                        self.name,
                        self._aggregate_strategy,
                        self._knowledge_store,
                        trigger,
                    )
                )

        try:
            return await super()._execute(
                event,
                context=context,
                client=client,
                additional_tools=(
                    list(additional_tools) + subtask_tools + knowledge_tools + memory_tools
                ),
                additional_middleware=harness_middleware + list(additional_middleware),
                response_schema=response_schema,
            )
        finally:
            for obs in self._observers:
                try:
                    obs.detach()
                except Exception:
                    logger.exception("Failed to detach observer %s", obs.name)
                finally:
                    with suppress(Exception):
                        await context.send(ObserverCompleted(name=obs.name))

            # on_end aggregation
            if (
                self._aggregate_strategy
                and self._knowledge_store
                and self._aggregate_trigger.on_end
            ):
                try:
                    events = list(await context.stream.history.get_events())
                    await self._aggregate_strategy.aggregate(events, context, self._knowledge_store)
                    usage = getattr(self._aggregate_strategy, "last_usage", {})
                    await context.send(
                        AggregationCompleted(
                            actor=self.name,
                            strategy=type(self._aggregate_strategy).__name__,
                            event_count=len(events),
                            llm_calls=1 if usage else 0,
                            usage=usage,
                        )
                    )
                except Exception:
                    logger.exception("Aggregation failed for %s", self.name)

            # Persist event log
            if self._knowledge_store:
                try:
                    events = list(await context.stream.history.get_events())
                    await EventLogWriter(self._knowledge_store).persist(context.stream.id, events)
                except Exception:
                    logger.exception("Event log persistence failed for %s", self.name)


# ------------------------------------------------------------------
# Internal middleware factories
# ------------------------------------------------------------------


class _HaltCheckMiddleware(BaseMiddleware):
    """Catches HaltEvent on the stream and short-circuits the LLM call.

    AlertPolicy emits HaltEvent when a FATAL alert is detected. This
    middleware runs after assembly and checks if a HaltEvent was emitted
    during the current execution. If so, it returns a synthetic "HALTED"
    response without calling the LLM.
    """

    def __init__(self, event: BaseEvent, context: Context) -> None:
        super().__init__(event, context)
        self._halted = False
        self._halt_reason = ""

        # Subscribe to HaltEvent on the stream
        from .events.alert import HaltEvent

        async def _on_halt(evt: HaltEvent) -> None:
            self._halted = True
            self._halt_reason = evt.reason

        self._sub = context.stream.where(HaltEvent).subscribe(_on_halt)

    async def on_llm_call(
        self,
        call_next: Callable[..., Any],
        events: Any,
        context: Context,
    ) -> ModelResponse:
        if self._halted:
            from autogen.beta.events import ModelMessage

            return ModelResponse(
                message=ModelMessage(content=f"HALTED: {self._halt_reason}"),
            )
        return await call_next(events, context)


class _HaltCheckMiddlewareFactory:
    """Factory for _HaltCheckMiddleware."""

    def __call__(self, event: BaseEvent, context: Context) -> _HaltCheckMiddleware:
        return _HaltCheckMiddleware(event, context)


class _AssemblerMiddlewareFactory:
    """Factory for AssemblerMiddleware."""

    def __init__(self, policies: list[AssemblyPolicy]) -> None:
        self._policies = policies

    def __call__(self, event: BaseEvent, context: Context) -> AssemblerMiddleware:
        return AssemblerMiddleware(event, context, policies=self._policies)


class _CompactionMiddleware(BaseMiddleware):
    """Triggers compaction after agent turns when thresholds are exceeded."""

    def __init__(
        self,
        event: BaseEvent,
        context: Context,
        *,
        actor_name: str,
        strategy: CompactStrategy,
        store: KnowledgeStore | None,
        trigger: CompactTrigger,
    ) -> None:
        super().__init__(event, context)
        self._actor_name = actor_name
        self._strategy = strategy
        self._store = store
        self._trigger = trigger
        self._last_compact_event_count = 0

    async def on_turn(
        self,
        call_next: Callable[..., Any],
        event: BaseEvent,
        context: Context,
    ) -> Any:
        result = await call_next(event, context)

        events = list(await context.stream.history.get_events())
        event_count = len(events)

        # Prevent double compaction — skip if count hasn't grown since last
        if event_count <= self._last_compact_event_count:
            return result

        should_compact = False
        if self._trigger.max_events > 0 and event_count > self._trigger.max_events:
            should_compact = True
        if self._trigger.max_tokens > 0:
            estimated = sum(len(str(e)) for e in events) // self._trigger.chars_per_token
            if estimated > self._trigger.max_tokens:
                should_compact = True

        if should_compact:
            compacted = await self._strategy.compact(events, context, self._store)
            await context.stream.history.replace(compacted)
            self._last_compact_event_count = len(compacted)

            usage = getattr(self._strategy, "last_usage", {})
            await context.send(
                CompactionCompleted(
                    actor=self._actor_name,
                    strategy=type(self._strategy).__name__,
                    events_before=event_count,
                    events_after=len(compacted),
                    llm_calls=1 if usage else 0,
                    usage=usage,
                )
            )

        return result


class _CompactionMiddlewareFactory:
    """Factory for _CompactionMiddleware."""

    def __init__(
        self,
        actor_name: str,
        strategy: CompactStrategy,
        store: KnowledgeStore | None,
        trigger: CompactTrigger,
    ) -> None:
        self._actor_name = actor_name
        self._strategy = strategy
        self._store = store
        self._trigger = trigger

    def __call__(self, event: BaseEvent, context: Context) -> _CompactionMiddleware:
        return _CompactionMiddleware(
            event,
            context,
            actor_name=self._actor_name,
            strategy=self._strategy,
            store=self._store,
            trigger=self._trigger,
        )


class _AggregationMiddleware(BaseMiddleware):
    """Triggers aggregation after agent turns when thresholds are exceeded."""

    def __init__(
        self,
        event: BaseEvent,
        context: Context,
        *,
        actor_name: str,
        strategy: AggregateStrategy,
        store: KnowledgeStore,
        trigger: AggregateTrigger,
    ) -> None:
        super().__init__(event, context)
        self._actor_name = actor_name
        self._strategy = strategy
        self._store = store
        self._trigger = trigger
        self._turn_count = 0
        self._last_aggregate_event_count = 0

    async def on_turn(
        self,
        call_next: Callable[..., Any],
        event: BaseEvent,
        context: Context,
    ) -> Any:
        result = await call_next(event, context)
        self._turn_count += 1

        events = list(await context.stream.history.get_events())

        should_aggregate = False
        if self._trigger.every_n_turns > 0 and self._turn_count % self._trigger.every_n_turns == 0:
            should_aggregate = True
        if self._trigger.every_n_events > 0:
            new_events = len(events) - self._last_aggregate_event_count
            if new_events >= self._trigger.every_n_events:
                should_aggregate = True

        if should_aggregate:
            await self._strategy.aggregate(events, context, self._store)
            self._last_aggregate_event_count = len(events)

            usage = getattr(self._strategy, "last_usage", {})
            await context.send(
                AggregationCompleted(
                    actor=self._actor_name,
                    strategy=type(self._strategy).__name__,
                    event_count=len(events),
                    llm_calls=1 if usage else 0,
                    usage=usage,
                )
            )

        return result


class _AggregationMiddlewareFactory:
    """Factory for _AggregationMiddleware."""

    def __init__(
        self,
        actor_name: str,
        strategy: AggregateStrategy,
        store: KnowledgeStore,
        trigger: AggregateTrigger,
    ) -> None:
        self._actor_name = actor_name
        self._strategy = strategy
        self._store = store
        self._trigger = trigger

    def __call__(self, event: BaseEvent, context: Context) -> _AggregationMiddleware:
        return _AggregationMiddleware(
            event,
            context,
            actor_name=self._actor_name,
            strategy=self._strategy,
            store=self._store,
            trigger=self._trigger,
        )
