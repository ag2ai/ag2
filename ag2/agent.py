# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Agent — the agentic unit of ag2.

An ``Agent`` runs a model loop, invokes tools, honours middleware, surfaces
events through observers, and optionally runs the harness primitives
(assembly policies, compaction/aggregation, knowledge store, subtask
spawning).

A bare ``Agent(name, config=cfg)`` has zero harness middleware — it behaves
exactly like a plain LLM loop. Harness features are opt-in: pass ``assembly=``
for context policies, ``knowledge=KnowledgeConfig(...)`` for a knowledge store,
or ``tasks=TaskConfig(...)`` to enable subtask spawning (disabled by default).
"""

import asyncio
import json
import logging
import types
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Sequence
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, suppress
from dataclasses import dataclass
from functools import partial
from itertools import chain
from typing import Any, Generic, Literal, TypeVar, overload
from uuid import uuid4

from fast_depends import Provider
from pydantic import ValidationError
from typing_extensions import TypeVar as TypeVar313

from ag2.events import BinaryResult

from .aggregate import AggregateStrategy, AggregateTrigger
from .annotations import Context
from .assembly import AssemblerMiddleware, AssemblyPolicy
from .compact import CompactStrategy, CompactTrigger
from .config import LLMClient, ModelConfig
from .events import (
    BaseEvent,
    DrainedModelRequest,
    HumanInputRequest,
    Input,
    ModelRequest,
    ModelResponse,
    ToolCallsEvent,
    ToolResultsEvent,
    UsageEvent,
    estimated_tokens,
    is_conversational,
)
from .events.lifecycle import (
    AggregationCompleted,
    AggregationFailed,
    AggregationStarted,
    CompactionCompleted,
    CompactionFailed,
    CompactionStarted,
    EventLogFailed,
    ObserverCompleted,
    ObserverStarted,
)
from .exceptions import ConfigNotProvidedError
from .history import History
from .hitl import HumanHook, default_hitl_hook, wrap_hitl
from .knowledge import DefaultBootstrap, EventLogWriter, KnowledgeStore
from .knowledge.config import KnowledgeConfig
from .middleware.base import (
    AgentTurn,
    BaseMiddleware,
    LLMCall,
    MiddlewareFactory,
    ToolMiddleware,
)
from .observers import Observer
from .plugin import Plugin, PluginTarget, PromptType
from .response import ResponseProto, ResponseSchema
from .stream import MemoryStream, Stream
from .task import CheckpointStore, Task, TaskSpec
from .tools.builtin.tool_search import ToolSearchToolSchema
from .tools.final import FunctionTool, FunctionToolSchema, Toolkit, tool
from .tools.schemas import ToolSchema
from .tools.subagents.run_task import run_task as _run_task
from .tools.subagents.subagent_tool import StreamFactory, subagent_tool
from .tools.tool import Tool
from .types import Omittable, SendableMessage, omit
from .usage import UsageReport

logger = logging.getLogger(__name__)


TResult = TypeVar313("TResult", default=str)
TAgent = TypeVar313("TAgent", default=str)
T2 = TypeVar("T2")


@dataclass
class TaskConfig:
    """Groups task-spawning Agent parameters.

    By default a subtask Agent inherits **all** of the parent's user-supplied
    tools (everything passed via ``tools=``). Subtask Agents never receive the
    auto-injected ``run_subtask`` / ``run_subtasks`` tools, so recursive
    delegation is structurally impossible — no depth limiting required.

    Use ``include_tools`` (allowlist) and ``exclude_tools`` (blocklist) to
    narrow what the subtask sees, and ``extra_tools`` to add capabilities the
    parent doesn't have.
    """

    config: ModelConfig | None = None
    prompt: str = "You are a task agent. Complete the assigned task thoroughly and concisely. Return only the result."
    include_tools: Iterable[str] | None = None
    exclude_tools: Iterable[str] = ()
    extra_tools: Iterable[Callable[..., Any] | Tool] = ()


class AgentReply(Generic[TResult, TAgent]):
    def __init__(
        self,
        response: ModelResponse,
        *,
        context: Context,
        client: LLMClient,
        agent: "Agent[TAgent]",
        provider: Provider | None,
        response_schema: ResponseProto[TResult] | None,
    ) -> None:
        self.response = response
        self.context = context
        self.__client = client
        self.__agent = agent
        self.__provider = provider
        self.__schema = response_schema

    async def content(
        self,
        *,
        retries: int | float = 0,
    ) -> TResult | None:
        schema = self.__schema
        if schema is None:
            return self.body  # type: ignore[return-value]

        max_retries = max(retries, 0)

        current = self
        attempt = 0

        while True:
            if current.body is None:
                return None

            attempt += 1
            try:
                return await schema.validate(
                    current.body,
                    context=current.context,
                    provider=current.__provider,
                )
            except ValidationError as e:
                if attempt > max_retries:
                    raise e

                schema_section = (
                    f"\n\n== Schema ==\n{json.dumps(schema.json_schema)}." if schema.json_schema is not None else ""
                )
                current = await current.ask(
                    "Your previous response could not be validated by schema."
                    f"{schema_section}"
                    "\n\nPlease try again."
                    "\n\n== Validation Error ==\n"
                    f"{e.json()}",
                    response_schema=schema,
                )

    @property
    def body(self) -> str | None:
        """Text body of the model's response for this turn."""
        return self.response.content

    @property
    def files(self) -> list[BinaryResult]:
        """Images generated by the model in this turn (decoded bytes)."""
        return self.response.files

    @property
    def history(self) -> History:
        return self.context.stream.history

    async def usage(self) -> UsageReport:
        """Token usage aggregated over the whole run (all events on this stream)."""
        return UsageReport.from_events(await self.context.stream.history.get_events())

    @overload
    async def ask(
        self,
        *msg: SendableMessage | Input,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: type[T2],
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentReply[T2, TAgent]": ...

    @overload
    async def ask(
        self,
        *msg: SendableMessage | Input,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: ResponseProto[T2],
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentReply[T2, TAgent]": ...

    @overload
    async def ask(
        self,
        *msg: SendableMessage | Input,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: None,
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentReply[str, TAgent]": ...

    @overload
    async def ask(
        self,
        *msg: SendableMessage | Input,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentReply[TAgent, TAgent]": ...

    async def ask(
        self,
        *msg: SendableMessage | Input,
        dependencies: dict[Any, Any] | None = None,
        variables: dict[Any, Any] | None = None,
        prompt: Iterable[str] = (),
        config: ModelConfig | None = None,
        tools: Iterable[Tool] = (),
        middleware: Iterable[MiddlewareFactory] = (),
        observers: Iterable[Observer] = (),
        response_schema: Omittable[ResponseProto[Any] | type | None] = omit,
        hitl_hook: HumanHook | None = None,
    ) -> "AgentReply[Any, Any]":
        initial_event = ModelRequest.ensure_request(list(msg))

        context = self.context
        if dependencies:
            context.dependencies.update(dependencies)
        if variables:
            context.variables.update(variables)
        if prompt:
            context.prompt = list(prompt)

        client = config.create() if config else self.__client

        return await self.__agent._execute(
            initial_event,
            context=context,
            client=client,
            hitl_hook=hitl_hook,
            additional_tools=tools,
            additional_middleware=middleware,
            additional_observers=observers,
            response_schema=response_schema,
        )

    @overload
    def run(
        self,
        *msg: SendableMessage | Input,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: type[T2],
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentRun[T2, TAgent]": ...

    @overload
    def run(
        self,
        *msg: SendableMessage | Input,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: ResponseProto[T2],
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentRun[T2, TAgent]": ...

    @overload
    def run(
        self,
        *msg: SendableMessage | Input,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: None,
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentRun[str, TAgent]": ...

    @overload
    def run(
        self,
        *msg: SendableMessage | Input,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentRun[TAgent, TAgent]": ...

    def run(
        self,
        *msg: SendableMessage | Input,
        dependencies: dict[Any, Any] | None = None,
        variables: dict[Any, Any] | None = None,
        prompt: Iterable[str] = (),
        config: ModelConfig | None = None,
        tools: Iterable[Tool] = (),
        middleware: Iterable[MiddlewareFactory] = (),
        observers: Iterable[Observer] = (),
        response_schema: Omittable[ResponseProto[Any] | type | None] = omit,
        hitl_hook: HumanHook | None = None,
    ) -> "AgentRun[Any, Any]":
        """Observable, non-blocking continuation of this reply (counterpart to ``ask``).

        Continues the conversation on this reply's context/stream and reuses its
        client (unless ``config`` overrides it), returning an :class:`AgentRun`
        whose turn advances when ``result()`` is awaited.
        """
        initial_event = ModelRequest.ensure_request(list(msg))

        context = self.context
        if dependencies:
            context.dependencies.update(dependencies)
        if variables:
            context.variables.update(variables)
        if prompt:
            context.prompt = list(prompt)

        client = config.create() if config else self.__client

        return self.__agent._make_run(
            initial_event,
            context=context,
            config=config,
            tools=tools,
            middleware=middleware,
            observers=observers,
            response_schema=response_schema,
            hitl_hook=hitl_hook,
            client=client,
        )


class AgentRun(Generic[TResult, TAgent]):
    """An observable turn opened by ``Agent.run``.

    ``AgentRun`` is an async context manager. Entering it opens the turn *scope*
    — middlewares are instantiated and the turn's subscribers (LLM callback,
    HITL, tool executor) are registered on the stream — but the turn does not
    advance until the caller drives it. Because the scope's subscribers are
    already live, observation is **push-based**: subscribe to ``stream`` (or
    attach ``observers=``) before driving, and the callbacks fire inline as the
    turn runs. See
    ``docs/adr/0008-agent-run-turn-is-scoped-to-its-context-manager.md`` and its
    amendment ``docs/adr/0009-agent-run-start-drives-in-background.md``.

    The turn is driven by a single task owned by the scope:

    * ``await run.result()`` — drives the turn and returns its ``AgentReply``;
      idempotent, re-raising the same failure on retry and never re-running the
      turn. Cancelling the await (e.g. ``asyncio.wait_for(run.result(), timeout)``)
      cancels the turn.
    * ``run.start()`` — kicks off that same task without awaiting it, so the
      caller can do other work (e.g. pull events via ``run.stream.join()``)
      meanwhile, then ``await run.result()`` for the reply.

    Either way the task is owned by the scope: leaving the block cancels it if it
    is still running, so the turn never outlives the block. ``run.stream`` is the
    underlying stream, for ``subscribe`` / ``where`` / ``get`` / ``join``
    observation. Leaving the block without driving runs nothing.
    """

    def __init__(
        self,
        agent: "Agent[TAgent]",
        trigger: BaseEvent,
        *,
        context: Context,
        config: ModelConfig | None,
        tools: Iterable[Tool],
        middleware: Iterable[MiddlewareFactory],
        observers: Iterable[Observer],
        response_schema: Omittable[ResponseProto[Any] | type | None],
        hitl_hook: HumanHook | None,
        client: LLMClient | None = None,
    ) -> None:
        self.__agent = agent
        self.__trigger = trigger
        self.__context = context
        self.__config = config
        self.__tools = tools
        self.__middleware = middleware
        self.__observers = observers
        self.__response_schema = response_schema
        self.__hitl_hook = hitl_hook
        # A continuation (``AgentReply.run``) supplies the original turn's client
        # so it keeps using the same provider; a fresh run leaves it None and the
        # client is resolved from config on enter.
        self.__client = client

        self.__stack = AsyncExitStack()
        self.__driver: Callable[[], Awaitable[AgentReply[TResult, TAgent]]] | None = None
        # The single task driving the turn, created lazily by ``start()`` or
        # ``result()``. It holds the turn's outcome, so awaiting it again is
        # idempotent; ``__aexit__`` cancels it if it is still running.
        self.__task: asyncio.Task[AgentReply[TResult, TAgent]] | None = None

    @property
    def stream(self) -> Stream:
        """The stream the turn runs on — subscribe to it to observe the turn live."""
        return self.__context.stream

    def enqueue(self, *content: SendableMessage | Input) -> None:
        """Append a follow-up message to the turn's inbox (non-blocking).

        Forwards to the stream inbox. The running turn drains it at its next or
        final model call, so a message enqueued while the turn is driving is
        consumed by that turn; one enqueued before ``result()`` merges into the
        first model call; one enqueued after the turn completes waits for the
        next turn on this stream. Safe to call from a stream subscriber (inline
        during the drive) or a concurrent task — it only appends.
        """
        self.__context.enqueue(*content)

    async def __aenter__(self) -> "AgentRun[TResult, TAgent]":
        client = self.__client
        if client is None:
            client = await self.__agent._prepare_turn(self.__trigger, self.__context, self.__config)
        self.__driver = await self.__stack.enter_async_context(
            self.__agent._turn_scope(
                self.__trigger,
                context=self.__context,
                client=client,
                hitl_hook=self.__hitl_hook,
                additional_tools=self.__tools,
                additional_middleware=self.__middleware,
                additional_observers=self.__observers,
                response_schema=self.__response_schema,
            )
        )
        return self

    def __ensure_task(self) -> "asyncio.Task[AgentReply[TResult, TAgent]]":
        if self.__driver is None:
            raise RuntimeError("AgentRun driven outside its 'async with' block")
        if self.__task is None:
            self.__task = asyncio.ensure_future(self.__driver())
        return self.__task

    def start(self) -> None:
        """Drive the turn in a scope-owned background task (non-blocking).

        Use this to make the turn progress while the caller does other work —
        typically pulling events via ``run.stream.join()``. ``await result()``
        then returns the same reply (or re-raises the same failure) the task
        produced. Idempotent: a no-op once the turn is already being driven. The
        task is owned by the scope — leaving the block cancels it if it is still
        running, so the turn never outlives the block.
        """
        self.__ensure_task()

    async def result(self) -> "AgentReply[TResult, TAgent]":
        """Drive the turn and return its ``AgentReply`` (idempotent).

        Awaits the turn's task, creating it here if ``start()`` was not called.
        Repeated calls return the same reply or re-raise the same failure.
        Cancelling this await cancels the turn.
        """
        task = self.__ensure_task()
        try:
            return await task
        except asyncio.CancelledError:
            task.cancel()
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> bool:
        task = self.__task
        if task is not None:
            if not task.done():
                task.cancel()
            # Retrieve the outcome so a cancelled/failed background turn does not
            # surface as an unretrieved-task warning. The caller opted out of the
            # result by leaving the block without awaiting ``result()``, so its
            # cancellation or failure is swallowed here.
            with suppress(asyncio.CancelledError, Exception):
                await task
        await self.__stack.aclose()
        return False


_STREAM_TURN_LOCK_ATTR = "_ag2_turn_lock"


def _get_stream_turn_lock(stream: Any) -> asyncio.Lock:
    """Return (creating if needed) a per-stream asyncio.Lock.

    Attaching the lock to the stream object itself means:
      * A fresh stream per turn (the default subtask / subagent path)
        pays a trivial no-contention acquire — no behaviour change.
      * A stream shared across concurrent ``Agent.ask`` calls
        serializes those calls so subscribers registered by one turn
        never fire for events of another.

    The lock is allocated lazily on first use so Agent instantiation
    outside an event loop (which would bind the lock to the wrong loop)
    still works.
    """
    lock = getattr(stream, _STREAM_TURN_LOCK_ATTR, None)
    if lock is None:
        lock = asyncio.Lock()
        try:
            setattr(stream, _STREAM_TURN_LOCK_ATTR, lock)
        except (AttributeError, TypeError):
            # Stream uses __slots__ and doesn't declare our attr — fall
            # back to a per-id registry so the lock still persists.
            _stream_id_locks.setdefault(id(stream), lock)
            lock = _stream_id_locks[id(stream)]
    return lock


# Fallback for streams that reject attribute assignment. Keyed by
# ``id(stream)`` — asyncio.Lock has no weakref slot, so we can't key on
# a weak reference. Only populated for slotted streams without the
# turn-lock slot (MemoryStream declares it).
_stream_id_locks: dict[int, asyncio.Lock] = {}


class Agent(PluginTarget, Generic[TResult]):
    """The agentic unit of ag2.

    An Agent runs a model loop, invokes tools, honours middleware, surfaces
    events through observers, and optionally runs the harness primitives
    (assembly, compaction, aggregation, knowledge store, subtask spawning).

    A bare ``Agent(name, config=cfg)`` has zero harness middleware and
    behaves exactly like a plain LLM loop. Harness features are opt-in:

    * ``assembly=`` — assembly policies (e.g. ``ConversationPolicy``,
      ``SlidingWindow``, ``AlertPolicy``). When non-empty,
      ``AssemblerMiddleware`` and ``_HaltCheckMiddleware`` are wired in.
    * ``knowledge=KnowledgeConfig(store=...)`` — persistent knowledge store,
      compaction, aggregation.
    * ``tasks=TaskConfig(...)`` — opt in to the auto-injected ``run_subtask``
      / ``run_subtasks`` tools, and override the LLM config / prompt /
      tool-inheritance rules for sub-task Agents. Defaults to ``False``
      (sub-task tools disabled).
    """

    @overload
    def __init__(
        self,
        name: str,
        prompt: PromptType | Iterable[PromptType] = ...,
        *,
        config: ModelConfig | None = ...,
        hitl_hook: HumanHook | None = ...,
        tools: Iterable[Callable[..., Any] | Tool] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        response_schema: type[TResult],
        plugins: Iterable["Plugin"] = ...,
        knowledge: KnowledgeConfig | None = ...,
        tasks: TaskConfig | Literal[False] = ...,
        assembly: Iterable[AssemblyPolicy] = ...,
    ) -> None: ...

    @overload
    def __init__(
        self,
        name: str,
        prompt: PromptType | Iterable[PromptType] = ...,
        *,
        config: ModelConfig | None = ...,
        hitl_hook: HumanHook | None = ...,
        tools: Iterable[Callable[..., Any] | Tool] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        response_schema: ResponseProto[TResult],
        plugins: Iterable["Plugin"] = ...,
        knowledge: KnowledgeConfig | None = ...,
        tasks: TaskConfig | Literal[False] = ...,
        assembly: Iterable[AssemblyPolicy] = ...,
    ) -> None: ...

    @overload
    def __init__(
        self,
        name: str,
        prompt: PromptType | Iterable[PromptType] = ...,
        *,
        config: ModelConfig | None = ...,
        hitl_hook: HumanHook | None = ...,
        tools: Iterable[Callable[..., Any] | Tool] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        response_schema: types.UnionType,
        plugins: Iterable["Plugin"] = ...,
        knowledge: KnowledgeConfig | None = ...,
        tasks: TaskConfig | Literal[False] = ...,
        assembly: Iterable[AssemblyPolicy] = ...,
    ) -> None: ...

    @overload
    def __init__(
        self,
        name: str,
        prompt: PromptType | Iterable[PromptType] = ...,
        *,
        config: ModelConfig | None = ...,
        hitl_hook: HumanHook | None = ...,
        tools: Iterable[Callable[..., Any] | Tool] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        response_schema: None = ...,
        plugins: Iterable["Plugin"] = ...,
        knowledge: KnowledgeConfig | None = ...,
        tasks: TaskConfig | Literal[False] = ...,
        assembly: Iterable[AssemblyPolicy] = ...,
    ) -> None: ...

    def __init__(
        self,
        name: str,
        prompt: PromptType | Iterable[PromptType] = (),
        *,
        config: ModelConfig | None = None,
        hitl_hook: HumanHook | None = None,
        tools: Iterable[Callable[..., Any] | Tool] = (),
        middleware: Iterable[MiddlewareFactory] = (),
        observers: Iterable[Observer] = (),
        dependencies: dict[Any, Any] | None = None,
        variables: dict[Any, Any] | None = None,
        response_schema: (ResponseProto[TResult] | type[TResult] | types.UnionType | None) = None,
        plugins: Iterable["Plugin"] = (),
        knowledge: KnowledgeConfig | None = None,
        tasks: TaskConfig | Literal[False] = False,
        assembly: Iterable[AssemblyPolicy] = (),
    ) -> None:
        self._init_target(
            name,
            prompt=prompt,
            hitl_hook=hitl_hook,
            tools=tools,
            middleware=middleware,
            observers=observers,
            dependencies=dependencies,
            variables=variables,
            plugins=plugins,
        )
        self.config = config
        self._response_schema = ResponseSchema.ensure_schema(response_schema)

        # Auto-injected tools (subtask toolkit, knowledge tool) live here,
        # NOT in the public ``self.tools`` (which holds only user-supplied
        # tools). They are chained into the tool set at execution time but
        # kept out of ``self.tools`` so they are never inherited by spawned
        # subtasks — that would re-enable recursion (run_subtask) or leak the
        # parent's knowledge tool.
        self._additional_tools: list[Tool] = []

        # Task spawning. ``tasks=False`` (the default) means no auto-injected
        # ``run_subtask`` / ``run_subtasks`` tools. Pass ``tasks=TaskConfig(...)``
        # to opt in.
        if tasks is False:
            self._task_config: TaskConfig | None = None
        else:
            self._task_config = tasks
            self._additional_tools.append(_build_subtask_toolkit(self))

        # Knowledge store + compaction/aggregation strategies
        if knowledge:
            self._agent_dependencies[KnowledgeStore] = knowledge.store
            self._knowledge_context = _KnowledgeContext(knowledge, self.name)

            if knowledge.expose_tool:
                self._additional_tools.append(_make_knowledge_tool(knowledge.store))

            if knowledge.compact:
                self.add_middleware(_CompactionMiddlewareFactory(self.name, knowledge))

            if (trigger := knowledge.aggregate_trigger) and (
                trigger.every_n_turns > 0 or trigger.every_n_events > 0 or trigger.on_end
            ):
                self.add_middleware(_AggregationMiddlewareFactory(self.name, knowledge))
        else:
            self._knowledge_context = _FakeKnowledgeContext()

        # Assembly policies (empty by default; bare Agent has no harness).
        self._policies.extend(assembly)
        if self._policies:
            for w in AssemblerMiddleware.validate_order(self._policies):
                logger.warning("Assembly policy ordering: %s", w)

            self.add_middleware(_AssemblerMiddlewareFactory(self._policies))
            self.add_middleware(_HaltCheckMiddlewareFactory())

    def task(
        self,
        title: str,
        *,
        description: str = "",
        payload: dict[str, Any] | None = None,
        capability: str | None = None,
        ttl_seconds: int | None = None,
        context: Context | None = None,
        checkpoint_store: CheckpointStore | None = None,
        resume_from: str | None = None,
    ) -> Task:
        """Create a ``Task`` whose lifecycle this Agent owns.

        Returns an unentered ``Task``; use as ``async with agent.task(...)``.
        Events flow on ``context.stream`` if a ``ConversationContext`` is
        supplied; else the Task creates a private ``MemoryStream`` on entry
        and events fire on it (only observers attached to that private
        stream see them).

        Inside the ``async with`` block, ``ag2.task`` is stamped into
        ``context.dependencies`` so any tool annotated with ``TaskInject``
        resolves to this Task.

        ``capability`` tags the task with a capability name. When the
        agent is registered with the network, the ``TaskMirror`` calls
        ``Hub.record_observation`` on the terminal event so the matching
        ``Resume.observed[capability]`` track record updates.

        ``checkpoint_store`` plus ``resume_from`` opt the task into
        restart-recoverable work: ``checkpoint(state)`` writes via the
        store, and on the next run ``resume_from=<prior_task_id>``
        reads that state back into ``task.resumed_state``. Standalone
        agents that don't supply a store pay no cost — checkpoint
        calls become silent no-ops.
        """
        spec = TaskSpec(
            title=title,
            description=description,
            payload=dict(payload) if payload else {},
            capability=capability,
        )
        return Task(
            owner_id=self.name,
            spec=spec,
            context=context,
            ttl_seconds=ttl_seconds,
            checkpoint_store=checkpoint_store,
            resume_from=resume_from,
        )

    @overload
    async def ask(
        self,
        *msg: SendableMessage | Input,
        stream: Stream | None = ...,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: type[T2],
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentReply[T2, TResult]": ...

    @overload
    async def ask(
        self,
        msg: SendableMessage | Input,
        *,
        stream: Stream | None = ...,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: ResponseProto[T2],
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentReply[T2, TResult]": ...

    @overload
    async def ask(
        self,
        msg: SendableMessage | Input,
        *,
        stream: Stream | None = ...,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: None,
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentReply[str, TResult]": ...

    @overload
    async def ask(
        self,
        msg: SendableMessage | Input,
        *,
        stream: Stream | None = ...,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentReply[TResult, TResult]": ...

    async def ask(
        self,
        *msg: SendableMessage | Input,
        stream: Stream | None = None,
        dependencies: dict[Any, Any] | None = None,
        variables: dict[Any, Any] | None = None,
        prompt: Iterable[str] = (),
        config: ModelConfig | None = None,
        tools: Iterable[Tool] = (),
        middleware: Iterable[MiddlewareFactory] = (),
        observers: Iterable[Observer] = (),
        response_schema: Omittable[ResponseProto[Any] | type | None] = omit,
        hitl_hook: HumanHook | None = None,
    ) -> "AgentReply[Any, Any]":
        # ``ask`` is the blocking degenerate case of ``run``: open a run, drive
        # it to its result, and let the scope close.
        async with self._open_run(
            *msg,
            stream=stream,
            dependencies=dependencies,
            variables=variables,
            prompt=prompt,
            config=config,
            tools=tools,
            middleware=middleware,
            observers=observers,
            response_schema=response_schema,
            hitl_hook=hitl_hook,
        ) as handle:
            return await handle.result()

    @overload
    def run(
        self,
        *msg: SendableMessage | Input,
        stream: Stream | None = ...,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: type[T2],
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentRun[T2, TResult]": ...

    @overload
    def run(
        self,
        msg: SendableMessage | Input,
        *,
        stream: Stream | None = ...,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: ResponseProto[T2],
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentRun[T2, TResult]": ...

    @overload
    def run(
        self,
        msg: SendableMessage | Input,
        *,
        stream: Stream | None = ...,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: None,
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentRun[str, TResult]": ...

    @overload
    def run(
        self,
        msg: SendableMessage | Input,
        *,
        stream: Stream | None = ...,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentRun[TResult, TResult]": ...

    def run(
        self,
        *msg: SendableMessage | Input,
        stream: Stream | None = None,
        dependencies: dict[Any, Any] | None = None,
        variables: dict[Any, Any] | None = None,
        prompt: Iterable[str] = (),
        config: ModelConfig | None = None,
        tools: Iterable[Tool] = (),
        middleware: Iterable[MiddlewareFactory] = (),
        observers: Iterable[Observer] = (),
        response_schema: Omittable[ResponseProto[Any] | type | None] = omit,
        hitl_hook: HumanHook | None = None,
    ) -> "AgentRun[Any, Any]":
        """Observable counterpart to ``ask``: open a turn scope you can watch.

        Returns an :class:`AgentRun` async context manager. Entering it opens the
        turn scope (subscribers live) and yields a handle; the turn advances when
        ``result()`` is awaited, with events flowing through ``run.stream`` as it
        runs. Mirrors ``ask``'s signature.
        """
        return self._open_run(
            *msg,
            stream=stream,
            dependencies=dependencies,
            variables=variables,
            prompt=prompt,
            config=config,
            tools=tools,
            middleware=middleware,
            observers=observers,
            response_schema=response_schema,
            hitl_hook=hitl_hook,
        )

    def _open_run(
        self,
        *msg: SendableMessage | Input,
        stream: Stream | None,
        dependencies: dict[Any, Any] | None,
        variables: dict[Any, Any] | None,
        prompt: Iterable[str],
        config: ModelConfig | None,
        tools: Iterable[Tool],
        middleware: Iterable[MiddlewareFactory],
        observers: Iterable[Observer],
        response_schema: Omittable[ResponseProto[Any] | type | None],
        hitl_hook: HumanHook | None,
    ) -> "AgentRun[Any, Any]":
        """Build a context from public kwargs and open a run over a fresh request.

        The shared body of ``ask`` and ``run``: both delegate here so context
        construction lives in one place.
        """
        context = self.__build_context(
            stream or MemoryStream(),
            prompt=prompt,
            dependencies=dependencies,
            variables=variables,
        )
        return self._make_run(
            ModelRequest.ensure_request(msg),
            context=context,
            config=config,
            tools=tools,
            middleware=middleware,
            observers=observers,
            response_schema=response_schema,
            hitl_hook=hitl_hook,
        )

    def _make_run(
        self,
        trigger: BaseEvent,
        *,
        context: Context,
        config: ModelConfig | None,
        tools: Iterable[Tool],
        middleware: Iterable[MiddlewareFactory],
        observers: Iterable[Observer],
        response_schema: Omittable[ResponseProto[Any] | type | None],
        hitl_hook: HumanHook | None,
        client: LLMClient | None = None,
    ) -> "AgentRun[Any, Any]":
        """The launch primitive: wrap a ``(trigger, context)`` turn as an ``AgentRun``.

        Every public entry point (``ask`` / ``run`` / ``_ask`` / ``resume`` /
        ``AgentReply.run``) funnels through here, so there is a single place a
        turn is launched. ``client`` is passed by continuations (``AgentReply.run``)
        that reuse the original turn's client; a fresh run leaves it None.
        """
        return AgentRun(
            self,
            trigger,
            context=context,
            config=config,
            tools=tools,
            middleware=middleware,
            observers=observers,
            response_schema=response_schema,
            hitl_hook=hitl_hook,
            client=client,
        )

    async def resume(
        self,
        *events: BaseEvent,
        stream: Stream | None = None,
        dependencies: dict[Any, Any] | None = None,
        variables: dict[Any, Any] | None = None,
        prompt: Iterable[str] = (),
        config: ModelConfig | None = None,
        tools: Iterable[Tool] = (),
        middleware: Iterable[MiddlewareFactory] = (),
        observers: Iterable[Observer] = (),
        response_schema: Omittable[ResponseProto[Any] | type | None] = omit,
        hitl_hook: HumanHook | None = None,
    ) -> "AgentReply[Any, Any]":
        """Resume a turn from a recorded trajectory, driven by an arbitrary event.

        ``events`` is the full trajectory to replay: all but the last event seed
        the stream history, and the **last** event is used as the ``trigger``
        (such as a ``ToolResultsEvent``) that re-enters the agent loop — so a
        turn can be continued from any event.

        The agent's conversation state is restored by replacing the stream's
        history with the seeded prefix. If ``stream`` is omitted a fresh
        ``MemoryStream`` is created; if one is supplied, its existing history is
        replaced.
        """
        stream = stream or MemoryStream()
        *history, trigger = events
        await stream.history.replace(history)

        context = self.__build_context(
            stream,
            prompt=prompt,
            dependencies=dependencies,
            variables=variables,
        )

        async with self._make_run(
            trigger,
            context=context,
            config=config,
            tools=tools,
            middleware=middleware,
            observers=observers,
            response_schema=response_schema,
            hitl_hook=hitl_hook,
        ) as handle:
            return await handle.result()

    async def _ask(
        self,
        *msg: SendableMessage | Input,
        context: Context,
        config: ModelConfig | None = None,
        tools: Iterable[Tool] = (),
        middleware: Iterable[MiddlewareFactory] = (),
        observers: Iterable[Observer] = (),
        response_schema: Omittable[ResponseProto[Any] | type | None] = omit,
        hitl_hook: HumanHook | None = None,
    ) -> "AgentReply[Any, Any]":
        """`Agent.ask()` alternative method to call agent with prebuild `context`."""
        async with self._make_run(
            ModelRequest.ensure_request(msg),
            context=context,
            config=config,
            tools=tools,
            middleware=middleware,
            observers=observers,
            response_schema=response_schema,
            hitl_hook=hitl_hook,
        ) as handle:
            return await handle.result()

    async def _prepare_turn(
        self,
        trigger: BaseEvent,
        context: Context,
        config: ModelConfig | None,
    ) -> LLMClient:
        """Resolve the client and seed the prompt for a turn about to be driven.

        Shared prelude for every ``AgentRun`` (and thus every public entry point):
        pick the config, fail fast if none, and fill ``context.prompt`` from the
        system + dynamic prompts when the caller hasn't supplied one.
        """
        config = config or self.config
        if not config:
            raise ConfigNotProvidedError()

        if not context.prompt:
            context.prompt.extend(self._system_prompt)

            for dp in self._dynamic_prompt:
                p = await dp(trigger, context)
                context.prompt.append(p)

        return config.create()

    async def _execute(
        self,
        event: BaseEvent,
        *,
        context: Context,
        client: LLMClient,
        hitl_hook: HumanHook | None = None,
        additional_tools: Iterable[Tool] = (),
        additional_middleware: Iterable[MiddlewareFactory] = (),
        additional_observers: Iterable[Observer] = (),
        response_schema: Omittable[ResponseProto[Any] | type | None] = omit,
    ) -> "AgentReply[Any, Any]":
        """Blocking turn: open the scope and immediately drive it to completion.

        The shared entry for callers that already hold a ``client`` (notably
        ``AgentReply.ask`` and the A2A executor). ``run`` / ``ask`` route through
        ``AgentRun``, which enters the *same* ``_turn_scope`` but defers the drive
        to ``result()``.
        """
        async with self._turn_scope(
            event,
            context=context,
            client=client,
            hitl_hook=hitl_hook,
            additional_tools=additional_tools,
            additional_middleware=additional_middleware,
            additional_observers=additional_observers,
            response_schema=response_schema,
        ) as drive:
            return await drive()

    @asynccontextmanager
    async def _turn_scope(
        self,
        event: BaseEvent,
        *,
        context: Context,
        client: LLMClient,
        hitl_hook: HumanHook | None = None,
        additional_tools: Iterable[Tool] = (),
        additional_middleware: Iterable[MiddlewareFactory] = (),
        additional_observers: Iterable[Observer] = (),
        response_schema: Omittable[ResponseProto[Any] | type | None] = omit,
    ) -> "AsyncIterator[Callable[[], Awaitable[AgentReply[Any, Any]]]]":
        """Open a turn scope and yield a ``drive`` callable that runs it once.

        Sets up everything a turn needs and keeps it live across the ``yield``:
        the per-stream lock (serialisation), the knowledge context, the resolved
        tool schemas, the instantiated middlewares, the turn's subscribers (LLM
        callback, HITL, tool executor) and the observer lifecycle. The yielded
        ``drive`` advances the turn from ``event`` and returns the ``AgentReply``.

        Splitting *open the scope* from *drive the turn* is what lets ``AgentRun``
        be observable without a background task: a caller enters the scope (so the
        subscribers are live), subscribes to the stream, then calls ``drive`` —
        events flow inline as the turn runs. ``ask`` enters and drives in one step.

        The per-stream lock serialises turns on a shared stream. Tool executor
        subscribers (see ``tools/executor.py``) register on the stream while the
        scope is open; if two turns shared a stream and overlapped, both turns'
        subscribers would see every event — causing duplicate tool execution,
        racing ``set_result`` calls on ``context.stream.get`` futures, and
        orphaned tool_use records that break subsequent Anthropic turns. The lock
        is a lazy per-stream ``asyncio.Lock``; a fresh single-turn stream pays a
        no-contention acquire. Sub-tasks spawn on their own stream, so they never
        contend with the parent turn's lock.
        """
        stream_lock = _get_stream_turn_lock(context.stream)
        async with stream_lock, self._knowledge_context.enter(context):
            if response_schema is omit:
                final_schema = self._response_schema
            else:
                final_schema = ResponseSchema.ensure_schema(response_schema)

            # collect actual tools
            all_tools: tuple[Tool, ...] = tuple(chain(self.tools, self._additional_tools, additional_tools))

            all_schemas: list[ToolSchema] = []
            known_tools: set[str] = set()
            tool_search_schema: ToolSearchToolSchema | None = None
            for t in all_tools:
                schemas = await t.schemas(context)

                for schema in schemas:
                    if isinstance(schema, FunctionToolSchema):
                        known_tools.add(schema.function.name)
                    else:
                        known_tools.add(schema.type)

                    if isinstance(schema, ToolSearchToolSchema):
                        tool_search_schema = schema
                    else:
                        all_schemas.append(schema)

            if tool_search_schema is not None:
                all_schemas.append(tool_search_schema)

            # instantiate middlewares
            middleware_instances: list[BaseMiddleware] = []
            agent_turn: AgentTurn = _execute_turn
            llm_call: LLMCall = partial(
                client,
                tools=all_schemas,
                response_schema=final_schema,
                serializer=self._serializer,
            )

            for m in reversed(tuple(chain(self._middleware, additional_middleware))):
                mw = m(event, context)
                middleware_instances.append(mw)

                agent_turn = partial(mw.on_turn, agent_turn)
                llm_call = partial(mw.on_llm_call, llm_call)

            # construct LLM callback
            async def _call_client(event: BaseEvent, context: Context) -> None:
                # Skip the LLM trigger when re-entered with a request we
                # injected ourselves a few lines below. Identity is carried by
                # the DrainedModelRequest subtype rather than a mutable flag
                # on Context, so the contract is checkable by isinstance and
                # independent of subscriber ordering.
                if isinstance(event, DrainedModelRequest):
                    return

                # Drain any pending enqueued input into the conversation before
                # the LLM sees the next call. Re-emit as DrainedModelRequest
                # so observers/logging/history storage still see it, while the
                # isinstance check above short-circuits the recursive entry
                # instead of issuing a second LLM call.
                merged = _drain_pending(context)
                if merged is not None:
                    await context.send(DrainedModelRequest(merged.parts))

                messages = await context.stream.history.get_events()
                result = await llm_call(messages, context)
                # Emit usage at the point it is spent, decoupled from the
                # response, so token accounting never depends on a response
                # being produced. UsageReport reads these events alone.
                if result.usage:
                    await context.send(
                        UsageEvent(
                            result.usage,
                            kind="model_call",
                            model=result.model,
                            provider=result.provider,
                            finish_reason=result.finish_reason,
                        )
                    )
                await context.send(result)

            with ExitStack() as stack:
                stack.enter_context(
                    context.stream.where(ModelRequest | ToolResultsEvent).sub_scope(_call_client),
                )

                hitl_hook_maker = wrap_hitl(hitl_hook) if hitl_hook else self._hitl_hook
                if hitl_hook_maker is not None:
                    stack.enter_context(
                        context.stream.where(HumanInputRequest).sub_scope(
                            hitl_hook_maker(middleware_instances),
                            interrupt=True,
                        ),
                    )

                else:
                    stack.enter_context(
                        context.stream.where(HumanInputRequest).sub_scope(
                            default_hitl_hook(middleware_instances),
                        ),
                    )

                self._tool_executor.register(
                    stack,
                    context,
                    tools=all_tools,
                    known_tools=known_tools,
                    middleware=middleware_instances,
                )

                async with _observer_lifecycle(
                    tuple(chain(self._observers, additional_observers)),
                    stack,
                    context,
                ):

                    async def drive() -> "AgentReply[Any, Any]":
                        message = await agent_turn(event, context)
                        return AgentReply(
                            message,
                            context=context,
                            agent=self,
                            client=client,
                            provider=self.dependency_provider,
                            response_schema=final_schema,
                        )

                    yield drive

    async def _spawn_subtask(self, task: str, ctx: Context) -> str:
        """Spawn a subtask Agent and delegate via ``run_task``.

        The subtask inherits the parent's user-supplied tools (filtered by
        ``TaskConfig.include_tools`` / ``exclude_tools``) plus
        ``TaskConfig.extra_tools``. It is constructed with ``tasks=False``
        (the default) so the child has **no** ``run_subtask`` tools —
        recursive delegation is impossible by construction. ``run_task``
        emits the ``TaskStarted`` / ``TaskCompleted`` / ``TaskFailed``
        lifecycle events and handles dependency/variable copy and HITL
        bridging.
        """
        tc = self._task_config
        if tc is None:
            return "Error: subtask spawning is disabled on this Agent (pass tasks=TaskConfig(...) to enable)."

        # Inherit only the parent's user-supplied tools — never the
        # auto-injected subtask toolkit or knowledge tool (excluded by
        # identity), so the child can't recurse or see parent-only tooling.
        inherited = _filter_subtask_tools(self.tools, tc.include_tools, tc.exclude_tools)

        bare = Agent(
            name=f"subtask-{uuid4().hex[:8]}",
            prompt=tc.prompt,
            config=tc.config or self.config,
            tools=[*inherited, *tc.extra_tools],
            tasks=False,
        )

        result = await _run_task(bare, task, parent_context=ctx)
        if not result.completed:
            return f"Error: {result.error}"
        return result.result or ""

    def __build_context(
        self,
        stream: Stream,
        *,
        prompt: Iterable[str] = (),
        dependencies: dict[Any, Any] | None = None,
        variables: dict[Any, Any] | None = None,
    ) -> Context:
        return Context(
            stream,
            prompt=list(prompt),
            dependencies=self._agent_dependencies | (dependencies or {}),
            variables=self._agent_variables | (variables or {}),
            dependency_provider=self.dependency_provider,
        )

    def as_tool(
        self,
        *,
        description: str,
        name: str | None = None,
        stream: StreamFactory | None = None,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        return subagent_tool(
            self,
            description=description,
            name=name,
            stream=stream,
            middleware=middleware,
        )


class _KnowledgeContext:
    def __init__(
        self,
        config: "KnowledgeConfig",
        agent_name: str,
    ) -> None:
        self.config = config
        self._agent_name = agent_name

        self.__lock: asyncio.Lock | None = None
        self.__bootstrapped = None

    @asynccontextmanager
    async def enter(self, context: "Context") -> AsyncIterator[None]:
        store = self.config.store

        if not self.__bootstrapped:
            if self.__lock is None:
                self.__lock = asyncio.Lock()

            async with self.__lock:
                if not self.__bootstrapped:
                    if not await store.exists("/.initialized"):
                        await store.write("/.initialized", self._agent_name)
                        bootstrap = self.config.bootstrap or DefaultBootstrap(mention_tool=self.config.expose_tool)
                        await bootstrap.bootstrap(store, self._agent_name)

                    self.__bootstrapped = True

        yield None

        if self.config.write_event_log:
            try:
                events = list(await context.stream.history.get_events())
                await EventLogWriter(store).persist(context.stream.id, events)

            except Exception as exc:
                logger.exception("Event log persistence failed for %s", self._agent_name)
                with suppress(Exception):
                    await context.send(
                        EventLogFailed(
                            agent=self._agent_name,
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )
                    )


class _FakeKnowledgeContext:
    @asynccontextmanager
    async def enter(self, context: "Context") -> AsyncIterator[None]:
        yield


async def _execute_turn(event: BaseEvent, context: Context) -> ModelResponse:
    # Drain stream-level inbox before the first model call: any messages
    # enqueued by background tasks from a previous ``ask`` (or any other
    # producer) get merged into this turn's initial ModelRequest so the LLM
    # sees them as one user turn, ordered before the new request.
    if isinstance(event, ModelRequest):
        leftover = _drain_pending(context)
        if leftover is not None:
            event = ModelRequest([*leftover.parts, *event.parts])

    # Sending the event triggers _call_client via the stream subscriber,
    # which drains any pending enqueued input before the LLM call.
    async with context.stream.get(ModelResponse) as result:
        await context.send(event)
        message: ModelResponse = await result

    while True:
        if message.tool_calls and not message.response_force:
            # Sending tool_calls triggers the tool executor, which produces a
            # ToolResultsEvent (or, for `final` tool results, a ModelResponse
            # directly). _call_client (auto-subscribed to ToolResultsEvent)
            # drains pending input and calls the LLM; we capture whichever
            # ModelResponse lands first.
            async with context.stream.get(ModelResponse) as result:
                await context.send(message.tool_calls)
                message = await result
            continue

        # Model has nothing more to do this turn. Drain any leftover enqueued
        # input into one more request to give it a chance to react. Background
        # tasks are not awaited — if they finish in time their enqueue lands
        # here, otherwise their result is lost (see ``spawn_background``).
        merged = _drain_pending(context)
        if merged is not None:
            async with context.stream.get(ModelResponse) as result:
                await context.send(merged)
                message = await result
            continue

        return message


def _drain_pending(context: Context) -> ModelRequest | None:
    """Drain the pending queue and return a single merged ``ModelRequest``.

    Returns ``None`` if the queue was empty (so callers can short-circuit).
    Multiple enqueues are merged because the wire form only cares about the
    flat list of parts — one merged request keeps history compact.
    """
    queue = context.pending_messages
    if not queue:
        return None
    parts: list[Input] = [part for request in queue for part in request.parts]
    queue.clear()
    return ModelRequest(parts)


@asynccontextmanager
async def _observer_lifecycle(
    observers: Sequence[Observer],
    stack: ExitStack,
    context: Context,
) -> AsyncIterator[None]:
    """Register ``observers`` on ``stack`` and bracket the turn with lifecycle events.

    Observers subscribe to the stream under the caller's ``ExitStack``, then an
    ``ObserverStarted`` is emitted for each so an observer can see its own start.
    On exit ``ObserverCompleted`` is emitted for each *before* the ``ExitStack``
    unwinds the subscriptions, so an observer subscribed to its own completion
    event still receives it.
    """
    for obs in observers:
        obs.register(stack, context)
        await context.send(ObserverStarted(name=getattr(obs, "name", type(obs).__name__)))

    try:
        yield

    finally:
        for obs in observers:
            with suppress(Exception):
                await context.send(ObserverCompleted(name=getattr(obs, "name", type(obs).__name__)))


_RUN_SUBTASK_DESCRIPTION = (
    "Spawn an isolated subtask agent for self-contained focused work. "
    "The subtask runs with a fresh conversation history and inherits this "
    "agent's tools, then returns its result as a string. "
    "You can call this tool multiple times in parallel within a single "
    "response — each call runs concurrently in its own context. For "
    "deliberate fan-out with one tool call, use run_subtasks instead."
)

_RUN_SUBTASKS_DESCRIPTION = (
    "Run multiple subtasks bundled into one tool call. Each task runs in "
    "its own isolated subtask agent (fresh history, parent's tools). "
    "Set parallel=True (default) to dispatch them concurrently — preferred "
    "when the tasks are independent. parallel=False runs them sequentially "
    "and is useful only when later tasks depend on earlier results."
)


def _build_subtask_toolkit(agent: "Agent[Any]") -> Toolkit:
    """Construct the ``run_subtask`` / ``run_subtasks`` tools for ``agent``.

    Called once per Agent instance from ``__init__``. The closures capture
    ``agent`` so the resulting Tools can be reused across every turn without
    re-allocation (per AGENTS.md: no nested function creation in runtime
    execution paths).
    """
    toolkit = Toolkit()

    @toolkit.tool(name="run_subtask", description=_RUN_SUBTASK_DESCRIPTION)
    async def run_subtask(task: str, ctx: Context) -> str:
        return await agent._spawn_subtask(task, ctx)

    @toolkit.tool(name="run_subtasks", description=_RUN_SUBTASKS_DESCRIPTION)
    async def run_subtasks(ctx: Context, tasks: list[str], parallel: bool = True) -> str:
        if parallel:
            raw = await asyncio.gather(
                *(agent._spawn_subtask(t, ctx) for t in tasks),
                return_exceptions=True,
            )
            results = [r if not isinstance(r, BaseException) else f"Error: {r}" for r in raw]
        else:
            results = []
            for t in tasks:
                try:
                    results.append(await agent._spawn_subtask(t, ctx))
                except Exception as e:
                    results.append(f"Error: {e}")

        parts = [f"## {task_desc}\n\n{result}" for task_desc, result in zip(tasks, results)]
        return "\n\n---\n\n".join(parts)

    return toolkit


def _filter_subtask_tools(
    tools: Iterable[FunctionTool],
    include: Iterable[str] | None,
    exclude: Iterable[str],
) -> list[FunctionTool]:
    """Apply ``include_tools`` / ``exclude_tools`` filters to ``tools``.

    ``include`` is an allowlist of tool names — ``None`` (the default) lets
    every tool through. ``exclude`` is always applied as a blocklist after
    the allowlist. Tool identity is by ``schema.function.name``.
    """
    include_set = set(include) if include is not None else None
    exclude_set = set(exclude)
    result: list[FunctionTool] = []
    for t in tools:
        name = t.schema.function.name
        if include_set is not None and name not in include_set:
            continue
        if name in exclude_set:
            continue
        result.append(t)
    return result


def _make_knowledge_tool(store: KnowledgeStore) -> Tool:
    """Build the ``knowledge`` action-group tool bound to ``store``.

    Called once at Agent ``__init__`` so the LLM-visible tool definition is
    stable across turns and we don't re-allocate the closure per turn.
    """

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
            result = await store.read(path)
            return result if result is not None else f"Not found: {path}"

        elif action == "write":
            if not content:
                return "Error: content is required for write action."
            await store.write(path, content)
            return f"Written to {path}"

        elif action == "list":
            entries = await store.list(path)
            if not entries:
                return f"Empty: {path}"
            skill_path = f"{path.rstrip('/')}/SKILL.md"
            skill = await store.read(skill_path)
            listing = "\n".join(entries)
            if skill:
                listing = f"{skill}\n---\n{listing}"
            return listing

        elif action == "delete":
            await store.delete(path)
            return f"Deleted: {path}"

        else:
            return f"Unknown action: {action}. Available: read, write, list, delete."

    return knowledge


class _HaltCheckMiddleware(BaseMiddleware):
    """Catches ``HaltEvent`` on the stream and short-circuits the LLM call.

    ``AlertPolicy`` emits ``HaltEvent`` when a FATAL alert is detected.
    This middleware subscribes in ``on_turn`` (scoped to a single turn so
    the subscription never outlives the ``_execute`` that created it) and,
    on any subsequent ``on_llm_call``, returns a synthetic ``HALTED``
    response instead of invoking the model.
    """

    def __init__(self, event: BaseEvent, context: Context) -> None:
        super().__init__(event, context)
        self._halted = False
        self._halt_reason = ""

    async def on_turn(
        self,
        call_next: Callable[..., Any],
        event: BaseEvent,
        context: Context,
    ) -> Any:
        from .events.alert import HaltEvent

        async def _on_halt(evt: HaltEvent) -> None:
            self._halted = True
            self._halt_reason = evt.reason

        sub_id = context.stream.where(HaltEvent).subscribe(_on_halt)
        try:
            return await call_next(event, context)
        finally:
            context.stream.unsubscribe(sub_id)

    async def on_llm_call(
        self,
        call_next: Callable[..., Any],
        events: Any,
        context: Context,
    ) -> ModelResponse:
        if self._halted:
            from ag2.events import ModelMessage

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

    def __init__(self, policies: Iterable[AssemblyPolicy]) -> None:
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
        # Count only conversational events — transient artifacts and persisted
        # telemetry (UsageEvent) must not drive compaction, even if persist_all.
        conversation_events = [e for e in events if is_conversational(e)]
        event_count = len(conversation_events)

        # Prevent double compaction — skip if count hasn't grown since last
        if event_count <= self._last_compact_event_count:
            return result

        should_compact = False
        if self._trigger.max_events > 0 and event_count > self._trigger.max_events:
            should_compact = True
        if self._trigger.max_tokens > 0:
            estimated = sum(estimated_tokens(e, self._trigger.chars_per_token) for e in conversation_events)
            if estimated > self._trigger.max_tokens:
                should_compact = True

        if should_compact:
            strategy_name = type(self._strategy).__name__
            await context.send(
                CompactionStarted(
                    agent=self._actor_name,
                    strategy=strategy_name,
                    event_count=len(events),
                )
            )
            try:
                compacted = await self._strategy.compact(events, context, self._store)
            except Exception as exc:
                logger.exception("Compaction failed for %s", self._actor_name)
                with suppress(Exception):
                    await context.send(
                        CompactionFailed(
                            agent=self._actor_name,
                            strategy=strategy_name,
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )
                    )
                return result

            await context.stream.history.replace(compacted)
            self._last_compact_event_count = len([e for e in compacted if is_conversational(e)])

            usage = getattr(self._strategy, "last_usage", {})
            await context.send(
                CompactionCompleted(
                    agent=self._actor_name,
                    strategy=strategy_name,
                    events_before=len(events),
                    events_after=len(compacted),
                    llm_calls=1 if usage else 0,
                    usage=usage,
                )
            )

        return result


class _CompactionMiddlewareFactory:
    """Factory for _CompactionMiddleware."""

    def __init__(self, actor_name: str, config: KnowledgeConfig) -> None:
        self._actor_name = actor_name
        self.config = config

    def __call__(self, event: BaseEvent, context: Context) -> _CompactionMiddleware:
        return _CompactionMiddleware(
            event,
            context,
            actor_name=self._actor_name,
            strategy=self.config.compact,
            store=self.config.store,
            trigger=self.config.compact_trigger,
        )


class _AggregationMiddleware(BaseMiddleware):
    """Triggers aggregation after agent turns when thresholds are exceeded.

    Counts are derived from stream history — this middleware holds no
    state of its own. That matters because ``_execute`` builds a fresh
    middleware instance on every ``ask()``; any per-instance counter
    would reset between turns and make ``every_n_turns=N`` for ``N>1``
    effectively dead.

    ``every_n_turns`` counts :class:`ModelRequest` events in history
    (one per user ask). ``every_n_events`` fires when the total history
    count crosses a multiple of the threshold during the current turn,
    which handles non-uniform growth (e.g. tool-heavy turns). ``on_end``
    fires unconditionally once the turn completes.
    """

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

    async def on_turn(
        self,
        call_next: Callable[..., Any],
        event: BaseEvent,
        context: Context,
    ) -> Any:
        events_before = list(await context.stream.history.get_events())

        try:
            result = await call_next(event, context)
        finally:
            events_after = list(await context.stream.history.get_events())
            count_after = len(events_after)

            should_aggregate = False

            if self._trigger.on_end:
                should_aggregate = True

            if self._trigger.every_n_turns > 0:
                turn_count = sum(1 for e in events_after if isinstance(e, ModelRequest))
                if turn_count > 0 and turn_count % self._trigger.every_n_turns == 0:
                    should_aggregate = True

            if self._trigger.every_n_events > 0:
                threshold = self._trigger.every_n_events
                # Count conversational/work events only, not telemetry (e.g. UsageEvent).
                _countable = (ModelRequest, ModelResponse, ToolCallsEvent, ToolResultsEvent)
                n_before = sum(1 for e in events_before if isinstance(e, _countable))
                n_after = sum(1 for e in events_after if isinstance(e, _countable))
                if n_after // threshold > n_before // threshold:
                    should_aggregate = True

            if should_aggregate:
                strategy_name = type(self._strategy).__name__
                with suppress(Exception):
                    await context.send(
                        AggregationStarted(
                            agent=self._actor_name,
                            strategy=strategy_name,
                            event_count=count_after,
                        )
                    )
                try:
                    await self._strategy.aggregate(events_after, context, self._store)
                    usage = getattr(self._strategy, "last_usage", {})
                    await context.send(
                        AggregationCompleted(
                            agent=self._actor_name,
                            strategy=strategy_name,
                            event_count=count_after,
                            llm_calls=1 if usage else 0,
                            usage=usage,
                        )
                    )
                except Exception as exc:
                    logger.exception("Aggregation failed for %s", self._actor_name)
                    with suppress(Exception):
                        await context.send(
                            AggregationFailed(
                                agent=self._actor_name,
                                strategy=strategy_name,
                                error_type=type(exc).__name__,
                                error=str(exc),
                            )
                        )

        return result


class _AggregationMiddlewareFactory:
    """Factory for _AggregationMiddleware."""

    def __init__(self, actor_name: str, config: "KnowledgeConfig") -> None:
        self._actor_name = actor_name
        self._strategy = config.aggregate
        self._store = config.store
        self._trigger = config.aggregate_trigger

    def __call__(self, event: BaseEvent, context: Context) -> _AggregationMiddleware:
        return _AggregationMiddleware(
            event,
            context,
            actor_name=self._actor_name,
            strategy=self._strategy,
            store=self._store,
            trigger=self._trigger,
        )
