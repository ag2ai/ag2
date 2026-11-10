# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Agent — the agentic unit of ag2.

An ``Agent`` runs a model loop, invokes tools, honours middleware, surfaces
events through observers, and optionally runs the harness primitives
(assembly policies, compaction/aggregation, knowledge store, subtask
spawning).

A bare ``Agent(name, config=cfg)`` has zero harness middleware — it behaves
exactly like a plain LLM loop. Harness features are opt-in: use ``middleware=``
for context policies, ``plugins=[KnowledgePlugin(...)]`` for a knowledge store,
or use ``Agent.as_tool()`` for explicit subagent delegation. The legacy
``tasks=TaskConfig(...)`` API is deprecated and will be removed in 1.4.0.
"""

import asyncio
import json
import logging
import threading
import types
import warnings
import weakref
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, suppress
from dataclasses import dataclass
from functools import partial
from itertools import chain
from typing import Any, Generic, Literal, TypeGuard, TypeVar, overload
from uuid import uuid4

from fast_depends import Provider
from fast_depends.library.serializer import SerializerProto
from pydantic import ValidationError
from typing_extensions import TypeVar as TypeVar313
from typing_extensions import deprecated

from ag2.events import BinaryResult

from ._model_input import model_inputs
from ._run_lifecycle import RunLifecycle
from .annotations import Context
from .assembly import AssemblerMiddleware, AssemblyPolicy
from .config import LLMClient, ModelConfig
from .context import StreamId
from .events import (
    BaseEvent,
    DrainedModelRequest,
    HumanInputRequest,
    Input,
    ModelReasoning,
    ModelRequest,
    ModelResponse,
    ProviderReplay,
    StatusObservation,
    ToolResultsEvent,
    UsageEvent,
    is_conversational,
)
from .events.lifecycle import (
    ObserverCompleted,
    ObserverStarted,
)
from .exceptions import ConfigNotProvidedError, HumanInputError
from .history import CANCELLED_TOOL_RESULT, HUMAN_INPUT_ABANDONED_TOOL_RESULT, History, close_unanswered_tool_calls
from .hitl import HumanHook, default_hitl_hook, wrap_hitl
from .knowledge._legacy import LegacyKnowledgeAdapter
from .knowledge.config import KnowledgeConfig
from .knowledge.plugin import validate_knowledge_middleware
from .middleware.base import (
    AgentTurn,
    BaseMiddleware,
    LLMCall,
    Middleware,
    MiddlewareFactory,
    ScopedMiddlewareFactory,
    ToolMiddleware,
)
from .middleware.describe import DescribedMiddleware
from .middleware.halt import HaltCheckMiddleware
from .observers import Observer
from .plugin import Plugin, PluginTarget, PromptType
from .response import ResponseProto, ResponseSchema
from .scopes import merge_variables, overlay_variables
from .stream import MemoryStream, Stream
from .task import CheckpointStore, Task, TaskSpec
from .tools.builtin.tool_search import ToolSearchToolSchema
from .tools.final import FunctionTool, Toolkit
from .tools.precedence import resolve_tools
from .tools.schemas import ToolSchema
from .tools.subagents._forwarding import PARENT_CANCELLED, DelegationRegistry, Forwarder
from .tools.subagents.run_task import run_task as _run_task
from .tools.subagents.subagent_tool import StreamOrFactory, SubagentTool, subagent_tool
from .tools.tool import Tool
from .types import Omit, Omittable, RequestOrigin, SendableMessage, omit
from .usage import UsageReport
from .utils import (
    AGENT_CONTEXT_DEPENDENCY_KEY,
    CONFIGURED_DEPENDENCIES_CONTEXT_KEY,
    MODEL_CONFIG_CONTEXT_DEPENDENCY_KEY,
)

logger = logging.getLogger(__name__)

TResult = TypeVar313("TResult", default=str)
TAgent = TypeVar313("TAgent", default=str)
T2 = TypeVar("T2")


TASK_CONFIG_DEPRECATION = (
    "TaskConfig, Agent(tasks=...) and the auto-injected run_subtask / run_subtasks tools are deprecated "
    "and will be removed in 1.4.0. Expose explicitly constructed worker Agents through Agent.as_tool() instead."
)

ASSEMBLY_DEPRECATION = (
    "Agent(assembly=...) is deprecated and will be removed in 1.4.0. Register "
    "middleware=[Middleware(AssemblerMiddleware, policies=[...]), Middleware(HaltCheckMiddleware)] instead; "
    "import AssemblerMiddleware from ag2.assembly and Middleware / HaltCheckMiddleware from ag2.middleware."
)


KNOWLEDGE_DEPRECATION = (
    "Agent(knowledge=...) is deprecated and will be removed in 1.4.0. "
    "Use plugins=[KnowledgePlugin(store=..., ...)] instead; import KnowledgePlugin from ag2.knowledge."
)


AGENT_TASK_DEPRECATION = (
    "Agent.task() is deprecated and will be removed in 1.4.0. "
    "Use Task(owner_id=agent.name, spec=TaskSpec(title=..., description=..., payload=..., capability=...), ...) "
    "instead; import Task and TaskSpec from ag2.task. Pass context, ttl_seconds, checkpoint_store and resume_from "
    "directly to Task."
)


@deprecated(TASK_CONFIG_DEPRECATION, category=None)
@dataclass
class TaskConfig:
    """Groups task-spawning Agent parameters.

    Deprecated; will be removed in 1.4.0 along with ``Agent(tasks=...)`` and
    the auto-injected ``run_subtask`` / ``run_subtasks`` tools. Construct
    worker Agents explicitly and pass their ``Agent.as_tool()`` tools to
    the parent instead.

    By default a subtask Agent inherits **all** of the parent's user-supplied
    tools (everything passed via ``tools=``). Subtask Agents never receive the
    auto-injected ``run_subtask`` / ``run_subtasks`` tools, so recursive
    delegation is structurally impossible — no depth limiting required.

    Use ``include_tools`` (allowlist) and ``exclude_tools`` (blocklist) to
    narrow what the subtask sees, and ``extra_tools`` to add capabilities the
    parent doesn't have. ``max_concurrency`` optionally bounds the total
    number of sub-tasks running for this Agent across both spawning tools.
    """

    config: ModelConfig | None = None
    prompt: str = "You are a task agent. Complete the assigned task thoroughly and concisely. Return only the result."
    include_tools: Iterable[str] | None = None
    exclude_tools: Iterable[str] = ()
    extra_tools: Iterable[Callable[..., Any] | Tool] = ()
    max_concurrency: int | None = None

    def __post_init__(self) -> None:
        if self.max_concurrency is not None and self.max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1 when set, got {self.max_concurrency}.")


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
        variables: Mapping[str, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        plugins: Iterable[Plugin] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: type[T2],
        capability: str | None = ...,
        ttl_seconds: int | None = ...,
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentReply[T2, TAgent]": ...

    @overload
    async def ask(
        self,
        *msg: SendableMessage | Input,
        dependencies: dict[Any, Any] | None = ...,
        variables: Mapping[str, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        plugins: Iterable[Plugin] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: ResponseProto[T2],
        capability: str | None = ...,
        ttl_seconds: int | None = ...,
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentReply[T2, TAgent]": ...

    @overload
    async def ask(
        self,
        *msg: SendableMessage | Input,
        dependencies: dict[Any, Any] | None = ...,
        variables: Mapping[str, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        plugins: Iterable[Plugin] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: None,
        capability: str | None = ...,
        ttl_seconds: int | None = ...,
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentReply[str, TAgent]": ...

    @overload
    async def ask(
        self,
        *msg: SendableMessage | Input,
        dependencies: dict[Any, Any] | None = ...,
        variables: Mapping[str, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        plugins: Iterable[Plugin] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        capability: str | None = ...,
        ttl_seconds: int | None = ...,
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentReply[TAgent, TAgent]": ...

    async def ask(
        self,
        *msg: SendableMessage | Input,
        dependencies: dict[Any, Any] | None = None,
        variables: Mapping[str, Any] | None = None,
        prompt: Iterable[str] = (),
        config: ModelConfig | None = None,
        tools: Iterable[Tool] = (),
        plugins: Iterable[Plugin] = (),
        middleware: Iterable[MiddlewareFactory] = (),
        observers: Iterable[Observer] = (),
        response_schema: Omittable[ResponseProto[Any] | type | None] = omit,
        capability: str | None = None,
        ttl_seconds: int | None = None,
        hitl_hook: HumanHook | None = None,
    ) -> "AgentReply[Any, Any]":
        return await self.__agent._execute(
            ModelRequest.ensure_request(list(msg)),
            context=self.context,
            client=None if config else self.__client,
            config=config,
            context_overrides=_ContextOverrides(dict(dependencies or {}), dict(variables or {}), tuple(prompt)),
            capability=capability,
            ttl_seconds=ttl_seconds,
            hitl_hook=hitl_hook,
            additional_tools=tools,
            additional_plugins=plugins,
            additional_middleware=middleware,
            additional_observers=observers,
            response_schema=response_schema,
        )

    @overload
    def run(
        self,
        *msg: SendableMessage | Input,
        dependencies: dict[Any, Any] | None = ...,
        variables: Mapping[str, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        plugins: Iterable[Plugin] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: type[T2],
        capability: str | None = ...,
        ttl_seconds: int | None = ...,
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentRun[T2, TAgent]": ...

    @overload
    def run(
        self,
        *msg: SendableMessage | Input,
        dependencies: dict[Any, Any] | None = ...,
        variables: Mapping[str, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        plugins: Iterable[Plugin] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: ResponseProto[T2],
        capability: str | None = ...,
        ttl_seconds: int | None = ...,
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentRun[T2, TAgent]": ...

    @overload
    def run(
        self,
        *msg: SendableMessage | Input,
        dependencies: dict[Any, Any] | None = ...,
        variables: Mapping[str, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        plugins: Iterable[Plugin] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: None,
        capability: str | None = ...,
        ttl_seconds: int | None = ...,
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentRun[str, TAgent]": ...

    @overload
    def run(
        self,
        *msg: SendableMessage | Input,
        dependencies: dict[Any, Any] | None = ...,
        variables: Mapping[str, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        plugins: Iterable[Plugin] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        capability: str | None = ...,
        ttl_seconds: int | None = ...,
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentRun[TAgent, TAgent]": ...

    def run(
        self,
        *msg: SendableMessage | Input,
        dependencies: dict[Any, Any] | None = None,
        variables: Mapping[str, Any] | None = None,
        prompt: Iterable[str] = (),
        config: ModelConfig | None = None,
        tools: Iterable[Tool] = (),
        plugins: Iterable[Plugin] = (),
        middleware: Iterable[MiddlewareFactory] = (),
        observers: Iterable[Observer] = (),
        response_schema: Omittable[ResponseProto[Any] | type | None] = omit,
        capability: str | None = None,
        ttl_seconds: int | None = None,
        hitl_hook: HumanHook | None = None,
    ) -> "AgentRun[Any, Any]":
        """Observable, non-blocking continuation of this reply (counterpart to ``ask``).

        Continues the conversation on this reply's context/stream and reuses its
        client (unless ``config`` overrides it), returning an :class:`AgentRun`
        whose turn advances when ``result()`` is awaited.
        """
        initial_event = ModelRequest.ensure_request(list(msg))

        client = None if config else self.__client

        return self.__agent._make_run(
            initial_event,
            context=self.context,
            context_overrides=_ContextOverrides(dict(dependencies or {}), dict(variables or {}), tuple(prompt)),
            config=config,
            tools=tools,
            plugins=plugins,
            middleware=middleware,
            observers=observers,
            response_schema=response_schema,
            capability=capability,
            ttl_seconds=ttl_seconds,
            hitl_hook=hitl_hook,
            client=client,
        )


def _owned_delegation(context: Context) -> Forwarder | None:
    """The delegation executing on this context's own stream, if any."""
    delegation: Forwarder | None = context.dependencies.get(Forwarder)
    if delegation is not None and delegation.stream.id != context.stream.id:
        return None
    return delegation


@dataclass(frozen=True, slots=True)
class _ContextOverrides:
    """Continuation values applied only after acquiring the stream turn lock."""

    dependencies: Mapping[Any, Any]
    variables: Mapping[str, Any]
    prompt: tuple[str, ...]

    def apply(self, context: Context) -> None:
        context.dependencies.update(self.dependencies)
        context.variables = overlay_variables(context.variables, self.variables)
        if self.prompt:
            context.prompt = list(self.prompt)


class _PluginPrompt(str):
    """Give a bound prompt fragment its own identity without changing its text."""

    __slots__ = ()

    def __copy__(self) -> "_PluginPrompt":
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> "_PluginPrompt":
        return self


def _remove_plugin_prompts(context: Context, fragments: Sequence[str]) -> None:
    owned = {id(fragment) for fragment in fragments}
    context.prompt[:] = [fragment for fragment in context.prompt if id(fragment) not in owned]


def _set_for_turn(stack: AsyncExitStack, dependencies: dict[Any, Any], key: Any, value: Any) -> None:
    """Set a framework dependency for one turn and restore what was there after it."""
    if (previous := dependencies.get(key)) is None:
        stack.callback(dependencies.pop, key, None)
    else:
        stack.callback(dependencies.__setitem__, key, previous)
    dependencies[key] = value


def _remove_plugin_default(context: Context, attribute: str, key: Any, value: Any) -> None:
    target: dict[Any, Any] = getattr(context, attribute)
    if target.get(key) is value:
        target.pop(key, None)


class _TurnPlugins:
    """Snapshot the contributions of plugins passed to one invocation."""

    def __init__(self, plugins: Iterable[Plugin]) -> None:
        self.static_prompt: list[str] = []
        self.dynamic_prompt: list[Callable[..., Awaitable[str]]] = []
        self.tools: list[Tool] = []
        self.middleware: list[MiddlewareFactory] = []
        self.observers: list[Observer] = []
        self.policies: list[AssemblyPolicy] = []
        self.dependencies: dict[Any, Any] = {}
        self.variables: dict[Any, Any] = {}
        self.hitl_hook: HumanHook | None = None
        self.has_conflicting_hitl_hooks = False
        for plugin in plugins:
            self.static_prompt.extend(plugin._system_prompt)
            self.dynamic_prompt.extend(plugin._dynamic_prompt)
            self.tools.extend(FunctionTool.ensure_tool(t) for t in plugin._tools)
            self.middleware.extend(plugin._middleware)
            self.observers.extend(plugin._observers)
            self.policies.extend(plugin._policies)
            self.dependencies.update(plugin._dependencies)
            self.variables.update(plugin._variables)
            if plugin._hitl_hook is not None:
                if self.hitl_hook is not None:
                    self.has_conflicting_hitl_hooks = True
                else:
                    self.hitl_hook = plugin._hitl_hook


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
        plugins: Iterable[Plugin] = (),
        middleware: Iterable[MiddlewareFactory],
        observers: Iterable[Observer],
        response_schema: Omittable[ResponseProto[Any] | type | None],
        capability: str | None = None,
        ttl_seconds: int | None = None,
        hitl_hook: HumanHook | None,
        client: LLMClient | None = None,
        context_overrides: _ContextOverrides | None = None,
    ) -> None:
        self.__agent = agent
        self.__trigger = trigger
        self.__context = context
        self.__context_overrides = context_overrides
        self.__config = config
        self.__tools = tools
        self.__plugins = tuple(plugins)
        self.__middleware = middleware
        self.__observers = observers
        self.__response_schema = response_schema
        self.__hitl_hook = hitl_hook
        self.__capability = capability
        self.__ttl_seconds = ttl_seconds
        # A continuation (``AgentReply.run``) supplies the original turn's client
        # so it keeps using the same provider; a fresh run leaves it None and the
        # client is resolved from config on enter.
        self.__client = client

        self.__stack = AsyncExitStack()
        self.__driver: Callable[[], Awaitable[AgentReply[TResult, TAgent]]] | None = None
        self.__lifecycle: RunLifecycle | None = None
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
        during the drive) or a concurrent task — it appends at once and
        announces the message with ``MessageEnqueued`` from a background task.

        The message counts as user input: when the turn consumes it, a new
        ``History.since_last_user_input()`` interval starts.
        """
        self.__context.enqueue(*content, origin="user")

    async def __aenter__(self) -> "AgentRun[TResult, TAgent]":
        self.__driver, self.__lifecycle = await self.__stack.enter_async_context(
            self.__agent._turn_scope(
                self.__trigger,
                context=self.__context,
                client=self.__client,
                config=self.__config,
                capability=self.__capability,
                ttl_seconds=self.__ttl_seconds,
                context_overrides=self.__context_overrides,
                additional_plugins=self.__plugins,
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
    ) -> Literal[False]:
        task = self.__task
        failure: BaseException | None = None
        try:
            if task is not None:
                if not task.done():
                    task.cancel()
                # Retrieve the outcome even when the caller opted out of result().
                # A task cancelled before its first coroutine step still needs a
                # terminal lifecycle; middleware scope teardown retains its contract.
                try:
                    await task
                except (asyncio.CancelledError, Exception) as error:
                    failure = error
            if failure is not None and self.__lifecycle is not None:
                await self.__lifecycle.reject(failure)
        finally:
            # Always release the stream turn lock, including when the driver
            # raised KeyboardInterrupt / SystemExit.
            await self.__stack.aclose()
        return False


_STREAM_TURN_LOCK_ATTR = "_ag2_turn_lock"

# Registry mapping ``stream.id`` -> that stream's turn lock. Weak-valued: each
# stream instance holds a strong reference to its own lock (see below), so an
# entry lives exactly as long as some object for that id does, and is reclaimed
# once the last one is collected.
_stream_locks_by_id: "weakref.WeakValueDictionary[StreamId, asyncio.Lock]" = weakref.WeakValueDictionary()


def _get_stream_turn_lock(stream: Stream) -> asyncio.Lock:
    """Return (creating if needed) a per-stream asyncio.Lock.

    A stream's identity is its ``id`` — that is the key ``History`` is built
    on — not the Python object holding it. Two objects constructed from the
    same ``(id, storage)`` pair (``persistent_stream()``, a session resumed
    from storage, a stream reconstructed in user code) denote one stream and
    take one lock, so:
      * A fresh stream per turn (the default subtask / subagent path)
        pays a trivial no-contention acquire — no behaviour change.
      * Concurrent ``Agent.ask`` calls on a stream, however each caller got
        hold of it, serialize so subscribers registered by one turn never
        fire for events of another.

    The lock is cached on the stream instance, which keeps the common case a
    single attribute read and keeps the registry entry alive.

    The lock is allocated lazily on first use so Agent instantiation
    outside an event loop (which would bind the lock to the wrong loop)
    still works.
    """
    # ``_STREAM_TURN_LOCK_ATTR`` is ours, not part of the ``Stream`` protocol,
    # so it has to be read defensively — unlike ``id``, which the protocol
    # guarantees.
    cached = getattr(stream, _STREAM_TURN_LOCK_ATTR, None)
    if isinstance(cached, asyncio.Lock):
        return cached

    lock = _stream_locks_by_id.get(stream.id)
    if lock is None:
        lock = _stream_locks_by_id[stream.id] = asyncio.Lock()

    # Cache on the instance: keeps the common case a single getattr, and the
    # strong reference keeps this stream's registry entry alive. A stream that
    # rejects the assignment (``__slots__`` without our attr) still serializes
    # via the registry, it just re-looks-up each turn.
    with suppress(AttributeError, TypeError):
        setattr(stream, _STREAM_TURN_LOCK_ATTR, lock)
    return lock


class Agent(PluginTarget, Generic[TResult]):
    """The agentic unit of ag2.

    An Agent runs a model loop, invokes tools, honours middleware, surfaces
    events through observers, and optionally runs the harness primitives
    (assembly, compaction, aggregation, knowledge store, subtask spawning).

    A bare ``Agent(name, config=cfg)`` has zero harness middleware and
    behaves exactly like a plain LLM loop. Harness features are opt-in:

    * ``middleware=`` — register context policies with
      ``Middleware(AssemblerMiddleware, policies=[...])``. Add
      ``Middleware(HaltCheckMiddleware)`` after it when using ``AlertPolicy``.
      The legacy ``assembly=`` parameter is deprecated and will be removed
      in 1.4.0; it keeps wiring both middleware automatically until then.
    * ``plugins=[KnowledgePlugin(store=...)]`` — persistent knowledge store,
      compaction, aggregation.
    * ``tasks=TaskConfig(...)`` — opt in to the auto-injected ``run_subtask``
      / ``run_subtasks`` tools, and override the LLM config / prompt /
      tool-inheritance rules for sub-task Agents. Defaults to ``False``
      (sub-task tools disabled). Deprecated; will be removed in 1.4.0.
      Use worker Agents exposed through ``Agent.as_tool()`` instead.
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
        variables: Mapping[str, Any] | None = ...,
        response_schema: type[TResult],
        plugins: Iterable["Plugin"] = ...,
        knowledge: KnowledgeConfig | None = ...,
        tasks: TaskConfig | Literal[False] = ...,
        assembly: Iterable[AssemblyPolicy] | None = ...,
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
        variables: Mapping[str, Any] | None = ...,
        response_schema: ResponseProto[TResult],
        plugins: Iterable["Plugin"] = ...,
        knowledge: KnowledgeConfig | None = ...,
        tasks: TaskConfig | Literal[False] = ...,
        assembly: Iterable[AssemblyPolicy] | None = ...,
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
        variables: Mapping[str, Any] | None = ...,
        response_schema: types.UnionType,
        plugins: Iterable["Plugin"] = ...,
        knowledge: KnowledgeConfig | None = ...,
        tasks: TaskConfig | Literal[False] = ...,
        assembly: Iterable[AssemblyPolicy] | None = ...,
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
        variables: Mapping[str, Any] | None = ...,
        response_schema: None = ...,
        plugins: Iterable["Plugin"] = ...,
        knowledge: KnowledgeConfig | None = ...,
        tasks: TaskConfig | Literal[False] = ...,
        assembly: Iterable[AssemblyPolicy] | None = ...,
    ) -> None: ...

    # A schema only known at run time (e.g. forwarded from a caller's argument):
    # the result type cannot be inferred.
    @overload
    def __init__(
        self: "Agent[Any]",
        name: str,
        prompt: PromptType | Iterable[PromptType] = ...,
        *,
        config: ModelConfig | None = ...,
        hitl_hook: HumanHook | None = ...,
        tools: Iterable[Callable[..., Any] | Tool] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        dependencies: dict[Any, Any] | None = ...,
        variables: Mapping[str, Any] | None = ...,
        response_schema: ResponseProto[Any] | type | types.UnionType | None = ...,
        plugins: Iterable["Plugin"] = ...,
        knowledge: KnowledgeConfig | None = ...,
        tasks: TaskConfig | Literal[False] = ...,
        assembly: Iterable[AssemblyPolicy] | None = ...,
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
        variables: Mapping[str, Any] | None = None,
        response_schema: (ResponseProto[TResult] | type[TResult] | types.UnionType | None) = None,
        plugins: Iterable["Plugin"] = (),
        knowledge: KnowledgeConfig | None = None,
        tasks: TaskConfig | Literal[False] = False,
        assembly: Iterable[AssemblyPolicy] | None = None,
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
        # ``None`` is treated as disabled: older docs showed it as the default.
        if tasks is False or tasks is None:
            self._task_config: TaskConfig | None = None
        else:
            warnings.warn(TASK_CONFIG_DEPRECATION, DeprecationWarning, stacklevel=2)
            self._task_config = tasks
            self._additional_tools.append(_build_subtask_toolkit(self))
        # One Semaphore per running loop: sharing one raises once a second loop
        # awaits it contended (the hazard ag2/extensions/docker/sandbox.py also
        # calls out). Provisioned by ``_task_slots_for``.
        self._task_slots: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            weakref.WeakKeyDictionary()
        )
        self._task_slots_lock = threading.Lock()

        # Temporary compatibility adapter; remove with knowledge= in 1.4.0.
        legacy_knowledge = LegacyKnowledgeAdapter(knowledge) if knowledge is not None else None
        if legacy_knowledge is not None:
            warnings.warn(KNOWLEDGE_DEPRECATION, DeprecationWarning, stacklevel=2)
            self._apply_plugin(legacy_knowledge.plugin)
            self._agent_dependencies.update(legacy_knowledge.dependencies)
            self._additional_tools.extend(legacy_knowledge.tools)
        validate_knowledge_middleware(self._middleware)

        # Assembly policies (empty by default; bare Agent has no harness).
        if assembly is not None:
            warnings.warn(ASSEMBLY_DEPRECATION, DeprecationWarning, stacklevel=2)
            self._policies.extend(assembly)
        if self._policies:
            for w in AssemblerMiddleware.validate_order(self._policies):
                logger.warning("Assembly policy ordering: %s", w)

            # Policies may come from constructor plugins alone. Keep the implicit
            # assembler for them unless the user already registered one.
            if not _has_user_assembler(self._middleware):
                self.add_middleware(_AssemblerMiddlewareFactory(self._policies))
                self.add_middleware(Middleware(HaltCheckMiddleware))

    # --- Composition -----------------------------------------------------
    # Read-only views of what this agent is made of, for comparing two agents,
    # asserting on composition in a test, or logging it. Each returns a copy or
    # a read-only view, so callers cannot mutate the agent through them.

    @property
    def system_prompt(self) -> tuple[str, ...]:
        """Static system prompt fragments, in order."""

        return tuple(self._system_prompt)

    @property
    def dynamic_prompt(self) -> tuple[PromptType, ...]:
        """Dynamic prompt hooks, in registration order."""

        return tuple(self._dynamic_prompt)

    @property
    def middleware(self) -> tuple[DescribedMiddleware, ...]:
        """Agent-level middleware, in registration order.

        Each entry pairs the middleware object with its description. Entries are
        built on access, so compare ``entry.middleware`` rather than the entry.
        """

        return tuple(DescribedMiddleware(m) for m in self._middleware)

    @property
    def dependencies(self) -> Mapping[Any, Any]:
        """Dependencies injected into tool calls, keyed as they were supplied."""

        return types.MappingProxyType(self._agent_dependencies)

    @property
    def variables(self) -> Mapping[Any, Any]:
        """Variables available to this agent."""

        return types.MappingProxyType(self._agent_variables)

    @property
    def has_hitl_hook(self) -> bool:
        """Whether a hook answers this agent's questions to a human, rather than the caller's."""

        return self._hitl_hook is not None

    @property
    def serializer(self) -> SerializerProto:
        """The serializer that encodes this agent's tool results."""

        return self._serializer

    @property
    def response_schema(self) -> ResponseProto[TResult] | None:
        """The response schema, or ``None`` when replies are free text."""

        return self._response_schema

    @property
    def observers(self) -> tuple[Observer, ...]:
        """Observers attached to this agent, in registration order."""

        return tuple(self._observers)

    @property
    def tasks(self) -> TaskConfig | None:
        """Sub-task configuration, or ``None`` when sub-tasks are not enabled.

        Deprecated; will be removed in 1.4.0 with ``Agent(tasks=...)``.
        """

        return self._task_config

    @property
    def assembly(self) -> tuple[AssemblyPolicy, ...]:
        """Policies registered via legacy ``assembly=`` or plugins, in application order.

        Policies registered directly through middleware are not included.
        The ``assembly=`` constructor parameter will be removed in 1.4.0.
        """

        return tuple(self._policies)

    @deprecated(AGENT_TASK_DEPRECATION)
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

        Deprecated; will be removed in 1.4.0. Construct
        ``Task(owner_id=self.name, spec=TaskSpec(title=title, ...), ...)``
        directly instead. The remaining Task parameters are unchanged.

        Returns an unentered ``Task``, used as an async context manager.
        Events flow on ``context.stream`` if a ``ConversationContext`` is
        supplied; else the Task creates a private ``MemoryStream`` on entry
        and events fire on it (only observers attached to that private
        stream see them).

        Inside the ``async with`` block, tools annotated with ``TaskInject``
        resolve to this Task — per asyncio task, so sibling tool calls that
        each open a task resolve their own, and a nested block resolves to
        the innermost enclosing task. ``context.dependencies`` also carries
        an ``ag2.task`` slot for the block (best-effort under overlapping
        siblings; kept for direct readers of the dict).

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
        variables: Mapping[str, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        plugins: Iterable[Plugin] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: type[T2],
        capability: str | None = ...,
        ttl_seconds: int | None = ...,
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentReply[T2, TResult]": ...

    @overload
    async def ask(
        self,
        *msg: SendableMessage | Input,
        stream: Stream | None = ...,
        dependencies: dict[Any, Any] | None = ...,
        variables: Mapping[str, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        plugins: Iterable[Plugin] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: ResponseProto[T2],
        capability: str | None = ...,
        ttl_seconds: int | None = ...,
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentReply[T2, TResult]": ...

    @overload
    async def ask(
        self,
        *msg: SendableMessage | Input,
        stream: Stream | None = ...,
        dependencies: dict[Any, Any] | None = ...,
        variables: Mapping[str, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        plugins: Iterable[Plugin] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: None,
        capability: str | None = ...,
        ttl_seconds: int | None = ...,
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentReply[str, TResult]": ...

    @overload
    async def ask(
        self,
        *msg: SendableMessage | Input,
        stream: Stream | None = ...,
        dependencies: dict[Any, Any] | None = ...,
        variables: Mapping[str, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        plugins: Iterable[Plugin] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        capability: str | None = ...,
        ttl_seconds: int | None = ...,
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentReply[TResult, TResult]": ...

    async def ask(
        self,
        *msg: SendableMessage | Input,
        stream: Stream | None = None,
        dependencies: dict[Any, Any] | None = None,
        variables: Mapping[str, Any] | None = None,
        prompt: Iterable[str] = (),
        config: ModelConfig | None = None,
        tools: Iterable[Tool] = (),
        plugins: Iterable[Plugin] = (),
        middleware: Iterable[MiddlewareFactory] = (),
        observers: Iterable[Observer] = (),
        response_schema: Omittable[ResponseProto[Any] | type | None] = omit,
        capability: str | None = None,
        ttl_seconds: int | None = None,
        hitl_hook: HumanHook | None = None,
    ) -> "AgentReply[Any, Any]":
        """Run a turn to completion.

        ``capability`` or ``ttl_seconds`` enables the task lifecycle for this
        invocation. TTL is an externally observed deadline, not a local timeout.
        """
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
            plugins=plugins,
            middleware=middleware,
            observers=observers,
            response_schema=response_schema,
            capability=capability,
            ttl_seconds=ttl_seconds,
            hitl_hook=hitl_hook,
        ) as handle:
            return await handle.result()

    @overload
    def run(
        self,
        *msg: SendableMessage | Input,
        stream: Stream | None = ...,
        dependencies: dict[Any, Any] | None = ...,
        variables: Mapping[str, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        plugins: Iterable[Plugin] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: type[T2],
        capability: str | None = ...,
        ttl_seconds: int | None = ...,
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentRun[T2, TResult]": ...

    @overload
    def run(
        self,
        *msg: SendableMessage | Input,
        stream: Stream | None = ...,
        dependencies: dict[Any, Any] | None = ...,
        variables: Mapping[str, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        plugins: Iterable[Plugin] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: ResponseProto[T2],
        capability: str | None = ...,
        ttl_seconds: int | None = ...,
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentRun[T2, TResult]": ...

    @overload
    def run(
        self,
        *msg: SendableMessage | Input,
        stream: Stream | None = ...,
        dependencies: dict[Any, Any] | None = ...,
        variables: Mapping[str, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        plugins: Iterable[Plugin] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: None,
        capability: str | None = ...,
        ttl_seconds: int | None = ...,
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentRun[str, TResult]": ...

    @overload
    def run(
        self,
        *msg: SendableMessage | Input,
        stream: Stream | None = ...,
        dependencies: dict[Any, Any] | None = ...,
        variables: Mapping[str, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        plugins: Iterable[Plugin] = ...,
        middleware: Iterable[MiddlewareFactory] = ...,
        observers: Iterable[Observer] = ...,
        capability: str | None = ...,
        ttl_seconds: int | None = ...,
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentRun[TResult, TResult]": ...

    def run(
        self,
        *msg: SendableMessage | Input,
        stream: Stream | None = None,
        dependencies: dict[Any, Any] | None = None,
        variables: Mapping[str, Any] | None = None,
        prompt: Iterable[str] = (),
        config: ModelConfig | None = None,
        tools: Iterable[Tool] = (),
        plugins: Iterable[Plugin] = (),
        middleware: Iterable[MiddlewareFactory] = (),
        observers: Iterable[Observer] = (),
        response_schema: Omittable[ResponseProto[Any] | type | None] = omit,
        capability: str | None = None,
        ttl_seconds: int | None = None,
        hitl_hook: HumanHook | None = None,
    ) -> "AgentRun[Any, Any]":
        """Observable counterpart to ``ask``: open a turn scope you can watch.

        Returns an :class:`AgentRun` async context manager. Entering it opens the
        turn scope (subscribers live) and yields a handle; the turn advances when
        ``result()`` is awaited, with events flowing through ``run.stream`` as it
        runs. Mirrors ``ask``'s signature. ``capability`` or ``ttl_seconds``
        enables task lifecycle events; an undriven run emits none. The terminal
        is published after scope cleanup. TTL is externally observed.
        """
        return self._open_run(
            *msg,
            stream=stream,
            dependencies=dependencies,
            variables=variables,
            prompt=prompt,
            config=config,
            tools=tools,
            plugins=plugins,
            middleware=middleware,
            observers=observers,
            response_schema=response_schema,
            capability=capability,
            ttl_seconds=ttl_seconds,
            hitl_hook=hitl_hook,
        )

    def _open_run(
        self,
        *msg: SendableMessage | Input,
        stream: Stream | None,
        dependencies: dict[Any, Any] | None,
        variables: Mapping[str, Any] | None,
        prompt: Iterable[str],
        config: ModelConfig | None,
        tools: Iterable[Tool],
        plugins: Iterable[Plugin] = (),
        middleware: Iterable[MiddlewareFactory],
        observers: Iterable[Observer],
        response_schema: Omittable[ResponseProto[Any] | type | None],
        capability: str | None = None,
        ttl_seconds: int | None = None,
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
            plugins=plugins,
            middleware=middleware,
            observers=observers,
            response_schema=response_schema,
            capability=capability,
            ttl_seconds=ttl_seconds,
            hitl_hook=hitl_hook,
        )

    def _make_run(
        self,
        trigger: BaseEvent,
        *,
        context: Context,
        config: ModelConfig | None,
        tools: Iterable[Tool],
        plugins: Iterable[Plugin] = (),
        middleware: Iterable[MiddlewareFactory],
        observers: Iterable[Observer],
        response_schema: Omittable[ResponseProto[Any] | type | None],
        capability: str | None = None,
        ttl_seconds: int | None = None,
        hitl_hook: HumanHook | None,
        client: LLMClient | None = None,
        context_overrides: _ContextOverrides | None = None,
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
            plugins=plugins,
            middleware=middleware,
            observers=observers,
            response_schema=response_schema,
            capability=capability,
            ttl_seconds=ttl_seconds,
            hitl_hook=hitl_hook,
            client=client,
            context_overrides=context_overrides,
        )

    async def resume(
        self,
        *events: BaseEvent,
        stream: Stream | None = None,
        dependencies: dict[Any, Any] | None = None,
        variables: Mapping[str, Any] | None = None,
        prompt: Iterable[str] = (),
        config: ModelConfig | None = None,
        tools: Iterable[Tool] = (),
        plugins: Iterable[Plugin] = (),
        middleware: Iterable[MiddlewareFactory] = (),
        observers: Iterable[Observer] = (),
        response_schema: Omittable[ResponseProto[Any] | type | None] = omit,
        capability: str | None = None,
        ttl_seconds: int | None = None,
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
            plugins=plugins,
            middleware=middleware,
            observers=observers,
            response_schema=response_schema,
            capability=capability,
            ttl_seconds=ttl_seconds,
            hitl_hook=hitl_hook,
        ) as handle:
            return await handle.result()

    async def _ask(
        self,
        *msg: SendableMessage | Input,
        context: Context,
        config: ModelConfig | None = None,
        tools: Iterable[Tool] = (),
        plugins: Iterable[Plugin] = (),
        middleware: Iterable[MiddlewareFactory] = (),
        observers: Iterable[Observer] = (),
        response_schema: Omittable[ResponseProto[Any] | type | None] = omit,
        capability: str | None = None,
        ttl_seconds: int | None = None,
        hitl_hook: HumanHook | None = None,
    ) -> "AgentReply[Any, Any]":
        """`Agent.ask()` alternative method to call agent with prebuild `context`."""
        async with self._make_run(
            ModelRequest.ensure_request(msg),
            context=context,
            config=config,
            tools=tools,
            plugins=plugins,
            middleware=middleware,
            observers=observers,
            response_schema=response_schema,
            capability=capability,
            ttl_seconds=ttl_seconds,
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

        context.dependencies[MODEL_CONFIG_CONTEXT_DEPENDENCY_KEY] = config

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
        client: LLMClient | None,
        config: ModelConfig | None = None,
        context_overrides: _ContextOverrides | None = None,
        capability: str | None = None,
        ttl_seconds: int | None = None,
        hitl_hook: HumanHook | None = None,
        additional_tools: Iterable[Tool] = (),
        additional_plugins: Iterable[Plugin] = (),
        additional_middleware: Iterable[MiddlewareFactory] = (),
        additional_observers: Iterable[Observer] = (),
        response_schema: Omittable[ResponseProto[Any] | type | None] = omit,
    ) -> "AgentReply[Any, Any]":
        """Blocking turn: open the scope and immediately drive it to completion.

        The shared entry for adapters that already hold a ``client`` (such as
        the A2A executor). ``run`` / ``ask`` route through
        ``AgentRun``, which enters the *same* ``_turn_scope`` but defers the drive
        to ``result()``.
        """
        async with self._turn_scope(
            event,
            context=context,
            client=client,
            config=config,
            context_overrides=context_overrides,
            capability=capability,
            ttl_seconds=ttl_seconds,
            hitl_hook=hitl_hook,
            additional_tools=additional_tools,
            additional_plugins=additional_plugins,
            additional_middleware=additional_middleware,
            additional_observers=additional_observers,
            response_schema=response_schema,
        ) as (drive, _):
            return await drive()

    @asynccontextmanager
    async def _turn_scope(
        self,
        event: BaseEvent,
        *,
        context: Context,
        client: LLMClient | None = None,
        config: ModelConfig | None = None,
        context_overrides: _ContextOverrides | None = None,
        capability: str | None = None,
        ttl_seconds: int | None = None,
        hitl_hook: HumanHook | None = None,
        additional_tools: Iterable[Tool] = (),
        additional_plugins: Iterable[Plugin] = (),
        additional_middleware: Iterable[MiddlewareFactory] = (),
        additional_observers: Iterable[Observer] = (),
        response_schema: ResponseProto[Any] | type | None | Omit = omit,
    ) -> AsyncGenerator[tuple[Callable[[], Awaitable[AgentReply[Any, Any]]], RunLifecycle | None]]:
        """Resolve invocation plugins and bracket their context contributions."""
        plugins = _TurnPlugins(additional_plugins)
        additional_middleware = tuple(additional_middleware)
        validate_knowledge_middleware(chain(self._middleware, plugins.middleware, additional_middleware))
        async with _get_stream_turn_lock(context.stream), AsyncExitStack() as stack:
            # Served entry points (A2A, A2UI, NLIP) build their own context; the
            # running agent is always this one for the duration of the turn.
            _set_for_turn(stack, context.dependencies, AGENT_CONTEXT_DEPENDENCY_KEY, self)
            # What this agent is configured with, so a delegation can tell it
            # apart from values the caller supplied for this request.
            configured = context.dependencies.get(CONFIGURED_DEPENDENCIES_CONTEXT_KEY, {}) | self._agent_dependencies
            _set_for_turn(stack, context.dependencies, CONFIGURED_DEPENDENCIES_CONTEXT_KEY, configured)
            delegation = _owned_delegation(context)
            lifecycle: RunLifecycle | None = None
            if delegation is not None:
                await stack.enter_async_context(delegation.execution(context))
                lifecycle = delegation.lifecycle(context, capability=capability, ttl_seconds=ttl_seconds)
            elif capability is not None or ttl_seconds is not None:
                # The title is published to observers and mirrored to the hub,
                # so it must not carry the caller's prompt.
                lifecycle = RunLifecycle(
                    self.name,
                    TaskSpec(title=self.name, capability=capability),
                    context,
                    ttl_seconds=ttl_seconds,
                )
            if lifecycle is not None:
                # Entered before any further setup so a setup failure still
                # publishes a terminal.
                await stack.enter_async_context(lifecycle.scope())
            if context_overrides is not None:
                context_overrides.apply(context)
            if plugins.has_conflicting_hitl_hooks:
                warnings.warn("Multiple invocation plugins set hitl_hook; the first wins.", stacklevel=2)
            for attribute, defaults in (
                ("dependencies", plugins.dependencies),
                ("variables", plugins.variables),
            ):
                target = getattr(context, attribute)
                for key, value in defaults.items():
                    if key not in target:
                        if attribute == "variables":
                            context.variables = overlay_variables(context.variables, {key: value})
                            target = context.variables
                        else:
                            target[key] = value
                            configured[key] = value
                        stack.callback(_remove_plugin_default, context, attribute, key, target[key])

            if client is None:
                client = await self._prepare_turn(event, context, config)

            if plugins.static_prompt or plugins.dynamic_prompt:
                fragments: list[str] = [_PluginPrompt(prompt) for prompt in plugins.static_prompt]
                context.prompt = [*context.prompt, *fragments]
                stack.callback(_remove_plugin_prompts, context, fragments)
                for prompt in plugins.dynamic_prompt:
                    fragment = _PluginPrompt(await prompt(event, context))
                    fragments.append(fragment)
                    context.prompt.append(fragment)

            async with self._drive_scope(
                event,
                context=context,
                client=client,
                hitl_hook=hitl_hook,
                plugin_hitl_hook=plugins.hitl_hook,
                additional_tools=(*plugins.tools, *additional_tools),
                additional_middleware=(*plugins.middleware, *additional_middleware),
                additional_observers=(*plugins.observers, *additional_observers),
                additional_policies=plugins.policies,
                response_schema=response_schema,
                lifecycle=lifecycle,
            ) as drive:
                yield drive, lifecycle

    @asynccontextmanager
    async def _drive_scope(
        self,
        event: BaseEvent,
        *,
        context: Context,
        client: LLMClient,
        hitl_hook: HumanHook | None = None,
        additional_tools: Iterable[Tool] = (),
        additional_policies: Iterable[AssemblyPolicy] = (),
        plugin_hitl_hook: HumanHook | None = None,
        additional_middleware: Iterable[MiddlewareFactory] = (),
        additional_observers: Iterable[Observer] = (),
        response_schema: Omittable[ResponseProto[Any] | type | None] = omit,
        lifecycle: RunLifecycle | None = None,
    ) -> "AsyncGenerator[Callable[[], Awaitable[AgentReply[Any, Any]]]]":
        """Open a turn scope and yield a ``drive`` callable that runs it once.

        Sets up everything a turn needs and keeps it live across the ``yield``:
        the per-stream lock (serialisation), middleware scopes, the resolved
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
        additional_middleware = tuple(additional_middleware)
        async with AsyncExitStack() as scope_stack:
            scoped_factories = tuple(chain(self._middleware, additional_middleware))
            for factory in scoped_factories:
                if isinstance(factory, ScopedMiddlewareFactory):
                    await scope_stack.enter_async_context(factory.scope(event, context))
            final_schema: ResponseProto[Any] | None
            if isinstance(response_schema, Omit):
                final_schema = self._response_schema
            else:
                final_schema = ResponseSchema.ensure_schema(response_schema)

            # collect actual tools
            all_tools: tuple[Tool, ...] = tuple(chain(self.tools, self._additional_tools, additional_tools))

            all_schemas: list[ToolSchema] = []
            tool_search_schema: ToolSearchToolSchema | None = None
            resolved_tools = await resolve_tools(all_tools, context)
            for schema in resolved_tools.schemas:
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

            factories = list(self._middleware)
            turn_policies = list(additional_policies)
            if self._policies or turn_policies:
                _attach_policies(factories, self._policies, turn_policies)

            for m in reversed(tuple(chain(factories, additional_middleware))):
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
                    await context.send(DrainedModelRequest(merged.parts, origin=merged.origin))

                messages = [
                    message
                    for message in await context.stream.history.get_events()
                    if is_conversational(message)
                    or isinstance(message, (ModelReasoning, ProviderReplay, StatusObservation))
                ]
                result = await llm_call(model_inputs(messages), context)
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
                if hitl_hook_maker is None and plugin_hitl_hook is not None:
                    hitl_hook_maker = wrap_hitl(plugin_hitl_hook)
                if hitl_hook_maker is not None:
                    stack.enter_context(
                        context.stream.where(HumanInputRequest).sub_scope(
                            hitl_hook_maker(middleware_instances),
                            interrupt=True,
                        ),
                    )

                elif (delegation := _owned_delegation(context)) is not None:
                    stack.enter_context(
                        context.stream.where(HumanInputRequest).sub_scope(
                            delegation.bridge, interrupt=True, sync_to_thread=False
                        )
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
                    tools=resolved_tools.tools,
                    known_tools=resolved_tools.known_tools,
                    middleware=middleware_instances,
                )

                async with _observer_lifecycle(
                    tuple(chain(self._observers, additional_observers)),
                    stack,
                    context,
                ):

                    async def drive() -> "AgentReply[Any, Any]":
                        delegation = _owned_delegation(context)
                        try:
                            if (
                                delegation is not None
                                and delegation.origin is not None
                                and delegation.origin.context is context
                            ):
                                await delegation.start(context)
                            if lifecycle is not None:
                                await lifecycle.start()
                            message = await agent_turn(event, context)
                            registry = context.dependencies.get(DelegationRegistry)
                            if delegation is not None:
                                await context.dependencies[DelegationRegistry].settle_descendants(delegation.path)
                            elif (
                                lifecycle is not None
                                and registry is not None
                                and registry.context.stream.id == context.stream.id
                            ):
                                await registry.settle_descendants(())
                        except BaseException as exc:
                            if lifecycle is not None:
                                lifecycle.failure = exc
                                if (
                                    delegation is not None
                                    and delegation.origin is not None
                                    and delegation.origin.halted
                                ):
                                    lifecycle.cancel_reason = delegation.origin.reason
                            registry = context.dependencies.get(DelegationRegistry)
                            if registry is not None and (
                                delegation is not None or registry.context.stream.id == context.stream.id
                            ):
                                path = delegation.path if delegation is not None else ()
                                registry.cancel_descendants(path, str(exc) or PARENT_CANCELLED)
                                await registry.settle_descendants(path)
                            if isinstance(exc, (HumanInputError, asyncio.CancelledError)):
                                await close_unanswered_tool_calls(
                                    context.stream.history,
                                    result=HUMAN_INPUT_ABANDONED_TOOL_RESULT
                                    if isinstance(exc, HumanInputError)
                                    else CANCELLED_TOOL_RESULT,
                                )
                            raise
                        if lifecycle is not None:
                            lifecycle.result = message.content
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
        recursive delegation is impossible by construction. The delegated run
        uses the common task lifecycle; ``run_task`` handles dependency/variable
        copying and HITL bridging.
        """
        tc = self._task_config
        if tc is None:
            return "Error: subtask spawning is disabled on this Agent (pass tasks=TaskConfig(...) to enable)."
        if tc.max_concurrency is not None:
            async with self._task_slots_for(tc.max_concurrency):
                return await self._run_subtask(task, ctx, tc)
        return await self._run_subtask(task, ctx, tc)

    def _task_slots_for(self, max_concurrency: int) -> asyncio.Semaphore:
        """The subtask semaphore for the running loop, created on first use.

        A ``threading.Lock`` because several OS threads can drive one Agent, and
        two racing creators would leave the loser capped by a semaphore nothing
        else holds. Closed loops are dropped on the way past: a contended
        ``asyncio.Semaphore`` caches its loop, keeping its own weak key alive.
        """
        loop = asyncio.get_running_loop()
        with self._task_slots_lock:
            slots = self._task_slots.get(loop)
            if slots is None:
                for closed in [cached for cached in self._task_slots if cached.is_closed()]:
                    del self._task_slots[closed]
                slots = asyncio.Semaphore(max_concurrency)
                self._task_slots[loop] = slots
            return slots

    async def _run_subtask(self, task: str, ctx: Context, tc: TaskConfig) -> str:
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
        variables: Mapping[str, Any] | None = None,
    ) -> Context:
        context = Context(
            stream,
            prompt=list(prompt),
            dependencies=self._agent_dependencies | (dependencies or {}),
            variables=merge_variables(self._agent_variables, variables),
            dependency_provider=self.dependency_provider,
        )
        context.dependencies[AGENT_CONTEXT_DEPENDENCY_KEY] = self
        return context

    def as_tool(
        self,
        *,
        description: str,
        name: str | None = None,
        stream: StreamOrFactory | None = None,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> SubagentTool:
        """Expose this agent as a delegation tool for another agent.

        ``stream=`` accepts ``None`` (a fresh stream per delegation), a
        ``Stream`` instance (reused by every delegation, which makes this
        agent stateful), or a ``StreamFactory``. See ``subagent_tool``.
        """
        return subagent_tool(
            self,
            description=description,
            name=name,
            stream=stream,
            middleware=middleware,
        )


async def _execute_turn(event: BaseEvent, context: Context) -> ModelResponse:
    # Drain stream-level inbox before the first model call: any messages
    # enqueued by background tasks from a previous ``ask`` (or any other
    # producer) get merged into this turn's initial ModelRequest so the LLM
    # sees them as one user turn, ordered before the new request.
    if isinstance(event, ModelRequest):
        leftover = _drain_pending(context)
        if leftover is not None:
            event = ModelRequest([*leftover.parts, *event.parts], origin=event.origin)

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
    flat list of parts — one merged request keeps history compact. The merged
    request is user input if any queued message was.
    """
    queue = context.pending_messages
    if not queue:
        return None
    parts: list[Input] = [part for request in queue for part in request.parts]
    origin: RequestOrigin = "user" if any(request.origin == "user" for request in queue) else "service"
    queue.clear()
    return ModelRequest(parts, origin=origin)


@asynccontextmanager
async def _observer_lifecycle(
    observers: Sequence[Observer],
    stack: ExitStack,
    context: Context,
) -> AsyncGenerator[None]:
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
            # One sub-task that could not reach a human sinks the whole call:
            # rendering it as text would make it the tool's output, and the
            # model would read an unaskable approval as a subtask that failed.
            for r in raw:
                if isinstance(r, HumanInputError):
                    raise r
            results = [r if not isinstance(r, BaseException) else f"Error: {r}" for r in raw]
        else:
            results = []
            for t in tasks:
                try:
                    results.append(await agent._spawn_subtask(t, ctx))
                except HumanInputError:
                    raise
                except Exception as e:
                    results.append(f"Error: {e}")

        parts = [f"## {task_desc}\n\n{result}" for task_desc, result in zip(tasks, results)]
        return "\n\n---\n\n".join(parts)

    return toolkit


def _filter_subtask_tools(
    tools: Iterable[Tool],
    include: Iterable[str] | None,
    exclude: Iterable[str],
) -> list[Tool]:
    """Apply ``include_tools`` / ``exclude_tools`` filters to ``tools``.

    ``include`` is an allowlist of tool names — ``None`` (the default) lets
    every tool through. ``exclude`` is always applied as a blocklist after
    the allowlist. Tool identity is by ``Tool.name``.
    """
    include_set = set(include) if include is not None else None
    exclude_set = set(exclude)
    result: list[Tool] = []
    for t in tools:
        name = t.name
        if include_set is not None and name not in include_set:
            continue
        if name in exclude_set:
            continue
        result.append(t)
    return result


def _is_user_assembler(factory: MiddlewareFactory) -> TypeGuard[Middleware]:
    return isinstance(factory, Middleware) and issubclass(factory.cls, AssemblerMiddleware)


def _has_user_assembler(factories: Iterable[MiddlewareFactory]) -> bool:
    return any(_is_user_assembler(f) for f in factories)


def _attach_policies(
    factories: list[MiddlewareFactory],
    agent_policies: Sequence[AssemblyPolicy],
    turn_policies: Sequence[AssemblyPolicy],
) -> None:
    """Hand agent and invocation-plugin policies to the turn's assembler, in place.

    The assembler is the one ``assembly=`` wired implicitly, or else the first
    ``Middleware(AssemblerMiddleware, ...)`` the user registered; a user
    assembler's own policies run first. With neither, an assembler and a halt
    check are appended, as ``assembly=`` would have done.
    """
    for index, factory in enumerate(factories):
        if isinstance(factory, _AssemblerMiddlewareFactory):
            if not turn_policies:
                # Already runs ``agent_policies``, validated at construction.
                return
            policies = [*agent_policies, *turn_policies]
            factories[index] = _AssemblerMiddlewareFactory(policies)
            break
        if _is_user_assembler(factory):
            policies = [*factory.options.get("policies", ()), *agent_policies, *turn_policies]
            factories[index] = Middleware(factory.cls, **{**factory.options, "policies": policies})
            break
    else:
        policies = [*agent_policies, *turn_policies]
        factories.extend((_AssemblerMiddlewareFactory(policies), Middleware(HaltCheckMiddleware)))

    for warning in AssemblerMiddleware.validate_order(policies):
        logger.warning("Assembly policy ordering: %s", warning)


class _AssemblerMiddlewareFactory:
    """Factory for AssemblerMiddleware."""

    def __init__(self, policies: Iterable[AssemblyPolicy]) -> None:
        self._policies = policies

    def __call__(self, event: BaseEvent, context: Context) -> AssemblerMiddleware:
        return AssemblerMiddleware(event, context, policies=self._policies)
