# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Actor — an autonomous agent with observers, signals, and task spawning.

Actor extends Agent with three capabilities:
1. Observer management — attach/detach observers that monitor the event stream
2. Signal injection — observer signals are delivered via SignalPolicy
3. Task spawning — spawn_task/spawn_tasks tools for delegating subtasks

Works standalone (no Hub required). Optionally registers with a Hub
for cross-actor discovery and delegation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from contextlib import suppress
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from autogen.beta.agent import Agent, AgentReply, PromptType
from autogen.beta.annotations import Context
from autogen.beta.config import LLMClient, ModelConfig
from autogen.beta.events import BaseEvent, ModelMessage, ModelMessageChunk, ModelResponse
from autogen.beta.hitl import HumanHook
from autogen.beta.middleware.base import BaseMiddleware, MiddlewareFactory
from autogen.beta.stream import MemoryStream
from autogen.beta.tools.final import tool
from autogen.beta.tools.tool import Tool
from autogen.beta.types import Omittable, omit

if TYPE_CHECKING:
    from autogen.beta.response import ResponseProto

from .events import ObserverCompleted, ObserverStarted, TaskProgress, TaskRequest, TaskResult
from .observer import Observer
from .primitives.harness import ContextHarness, ConversationHarness, HarnessMiddleware
from .primitives.signal import InjectToPrompt, Severity, Signal, SignalPolicy


class Actor(Agent):
    """An autonomous agent with observers, signals, and task spawning.

    Actor extends Agent. It delegates to ``super()._execute()`` rather than
    reimplementing the event loop. Actor concerns (observers, signals, tasks)
    are injected via additional tools and middleware.

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
    harness:
        ContextHarness controlling what events the LLM sees.
        Default: ConversationHarness (preserves current Agent behavior).
    task_config:
        LLM config for spawned task sub-agents. Falls back to ``config``.
    task_prompt:
        System prompt for task sub-agents.
    signal_policy:
        How signals are delivered. Default: InjectToPrompt.
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
        harness: ContextHarness | None = None,
        task_config: ModelConfig | None = None,
        task_prompt: str = (
            "You are a task agent. Complete the assigned task " "thoroughly and concisely. Return only the result."
        ),
        signal_policy: SignalPolicy | None = None,
        hitl_hook: HumanHook | None = None,
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
        self._harness = harness or ConversationHarness()
        self._task_config: ModelConfig | None = task_config or config
        self._task_prompt = task_prompt
        self._signal_policy = signal_policy or InjectToPrompt()

    # ------------------------------------------------------------------
    # Observer management
    # ------------------------------------------------------------------

    def add_observer(self, observer: Observer) -> None:
        """Register an observer (before calling ask())."""
        self._observers.append(observer)

    # ------------------------------------------------------------------
    # Task spawning tools
    # ------------------------------------------------------------------

    def _build_spawn_tools(self) -> list[Tool]:
        """Create spawn_task and spawn_tasks tools."""
        actor = self

        @tool
        async def spawn_task(task: str, ctx: Context) -> str:
            """Spawn a task agent to handle a subtask autonomously.

            Use this when a sub-problem can be researched or solved
            independently. The task agent runs with its own LLM context
            and returns the result.
            """
            return await actor._run_task(task, ctx)

        @tool
        async def spawn_tasks(ctx: Context, tasks: list[str], parallel: bool = True) -> str:
            """Spawn multiple task agents at once.

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

        return [spawn_task, spawn_tasks]

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
        spawn_tools = self._build_spawn_tools()

        # Signal collection queue — observers write, middleware reads.
        # _delivered_ids tracks signals already delivered by the policy to
        # prevent re-collection when EmitToStream re-emits them on the stream.
        signal_queue: list[Signal] = []
        _delivered_ids: set[int] = set()

        async def _collect_signal(signal: Signal) -> None:
            if id(signal) not in _delivered_ids:
                signal_queue.append(signal)

        signal_sub = context.stream.where(Signal).subscribe(_collect_signal)

        # Attach observers
        for obs in self._observers:
            obs.attach(context.stream, context)
            await context.send(ObserverStarted(name=obs.name))

        # Build middleware chain:
        # 1. HarnessMiddleware (outermost) — filters what events reach the LLM
        # 2. SignalInjectionMiddleware — injects alerts into prompt
        # 3. User-provided middleware (via additional_middleware)
        harness_mw = _HarnessMiddlewareFactory(self._harness)
        signal_mw = _SignalInjectionFactory(signal_queue, self._signal_policy, _delivered_ids)

        try:
            return await super()._execute(
                event,
                context=context,
                client=client,
                additional_tools=list(additional_tools) + spawn_tools,
                additional_middleware=[harness_mw, signal_mw] + list(additional_middleware),
                response_schema=response_schema,
            )
        finally:
            for obs in self._observers:
                try:
                    obs.detach()
                except Exception:
                    import logging

                    logging.getLogger(__name__).exception("Failed to detach observer %s", obs.name)
                finally:
                    with suppress(Exception):
                        await context.send(ObserverCompleted(name=obs.name))
            context.stream.unsubscribe(signal_sub)


# ------------------------------------------------------------------
# Internal middleware factories
# ------------------------------------------------------------------


class _SignalInjectionMiddleware(BaseMiddleware):
    """Drains signal queue and delivers via SignalPolicy before each LLM call."""

    def __init__(
        self,
        event: BaseEvent,
        context: Context,
        signal_queue: list[Signal],
        policy: SignalPolicy,
        delivered_ids: set[int],
    ) -> None:
        super().__init__(event, context)
        self._signal_queue = signal_queue
        self._policy = policy
        self._delivered_ids = delivered_ids

    async def on_llm_call(
        self,
        call_next: Callable[..., Any],
        events: Any,
        context: Context,
    ) -> ModelResponse:
        if self._signal_queue:
            signals = self._signal_queue[:]
            self._signal_queue.clear()

            # Mark signals as delivered so the stream subscriber won't
            # re-collect them when EmitToStream re-emits on the stream.
            for s in signals:
                self._delivered_ids.add(id(s))

            # Check for FATAL — halt without calling the LLM
            has_fatal = any(s.severity == Severity.FATAL for s in signals)
            if has_fatal:
                fatal = next(s for s in signals if s.severity == Severity.FATAL)
                await self._policy.deliver(signals, context)
                # Return a synthetic response to halt the agent loop
                return ModelResponse(
                    message=ModelMessage(content=f"HALTED: {fatal.message}"),
                )

            # Non-fatal: deliver (policy may append to prompt), call LLM, clean up
            prompt_len_before = len(context.prompt) if context.prompt else 0
            await self._policy.deliver(signals, context)
            prompt_len_after = len(context.prompt) if context.prompt else 0
            num_added = prompt_len_after - prompt_len_before

            try:
                return await call_next(events, context)
            finally:
                # Remove the entries appended by the policy using index-based
                # slicing so we never accidentally remove a pre-existing entry
                # that happens to have the same value.
                if num_added > 0 and context.prompt:
                    del context.prompt[-num_added:]

        return await call_next(events, context)


class _SignalInjectionFactory:
    """Factory for _SignalInjectionMiddleware."""

    def __init__(self, signal_queue: list[Signal], policy: SignalPolicy, delivered_ids: set[int]) -> None:
        self._signal_queue = signal_queue
        self._policy = policy
        self._delivered_ids = delivered_ids

    def __call__(self, event: BaseEvent, context: Context) -> _SignalInjectionMiddleware:
        return _SignalInjectionMiddleware(event, context, self._signal_queue, self._policy, self._delivered_ids)


class _HarnessMiddlewareFactory:
    """Factory for HarnessMiddleware."""

    def __init__(self, harness: ContextHarness) -> None:
        self._harness = harness

    def __call__(self, event: BaseEvent, context: Context) -> HarnessMiddleware:
        return HarnessMiddleware(event, context, harness=self._harness)
