# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Actor — the agentic unit of autogen.beta.

An ``Actor`` runs a model loop, invokes tools, honours middleware, surfaces
events through observers, and optionally runs the harness primitives
(assembly policies, compaction/aggregation, knowledge store, subtask
spawning).

A bare ``Actor(name, config=cfg)`` has zero harness middleware — it behaves
exactly like a plain LLM loop. Harness features are opt-in: pass ``assembly=``
for context policies, ``knowledge=KnowledgeConfig(...)`` for a knowledge store,
or ``tasks=TaskConfig(...)`` for subtask spawning defaults.
"""

from __future__ import annotations

import asyncio
import json
import logging
import types
import warnings
from collections.abc import Awaitable, Callable, Iterable
from contextlib import AsyncExitStack, ExitStack, suppress
from dataclasses import dataclass
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar, overload
from uuid import uuid4

from fast_depends import Provider
from pydantic import ValidationError
from typing_extensions import TypeVar as TypeVar313

from autogen.beta.events import BinaryResult
from autogen.beta.tools.builtin.web_search import WEB_SEARCH_TOOL_NAME, WebSearchToolSchema

from .aggregate import AggregateStrategy, AggregateTrigger
from .annotations import Context
from .assembly import AssemblerMiddleware, AssemblyPolicy
from .compact import CompactStrategy, CompactTrigger
from .config import LLMClient, ModelConfig
from .events import (
    BaseEvent,
    HumanInputRequest,
    Input,
    ModelRequest,
    ModelResponse,
    ToolResultsEvent,
)
from .events.conditions import Condition
from .events.lifecycle import (
    AggregationCompleted,
    CompactionCompleted,
    ObserverCompleted,
    ObserverStarted,
)
from .exceptions import ConfigNotProvidedError
from .history import History
from .hitl import HumanHook, default_hitl_hook, wrap_hitl
from .knowledge import DefaultBootstrap, EventLogWriter, KnowledgeStore, StoreBootstrap
from .middleware.base import AgentTurn, BaseMiddleware, LLMCall, MiddlewareFactory, ToolMiddleware
from .observer import Observer
from .observer import observer as observer_factory
from .response import ResponseProto, ResponseSchema
from .stream import MemoryStream, Stream
from .tools.executor import ToolExecutor
from .tools.final import FunctionParameters, FunctionTool, FunctionToolSchema, tool
from .tools.schemas import ToolSchema
from .tools.tool import Tool
from .types import ClassInfo, Omittable, omit
from .utils import CONTEXT_OPTION_NAME, build_model

if TYPE_CHECKING:
    from .conversable import ConversableAdapter
    from .tools.subagents import StreamFactory


logger = logging.getLogger(__name__)


TResult = TypeVar313("TResult", default=str)
TAgent = TypeVar313("TAgent", default=str)
T2 = TypeVar("T2")


PromptHook: TypeAlias = Callable[..., str] | Callable[..., Awaitable[str]]
PromptType: TypeAlias = str | PromptHook


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


class AgentReply(Generic[TResult, TAgent]):
    def __init__(
        self,
        response: ModelResponse,
        *,
        context: "Context",
        client: "LLMClient",
        agent: "Actor[TAgent]",
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

    @overload
    async def ask(
        self,
        *msg: str | Input,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        middleware: Iterable["MiddlewareFactory"] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: type[T2],
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentReply[T2, TAgent]": ...

    @overload
    async def ask(
        self,
        *msg: str | Input,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        middleware: Iterable["MiddlewareFactory"] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: ResponseProto[T2],
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentReply[T2, TAgent]": ...

    @overload
    async def ask(
        self,
        *msg: str | Input,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        middleware: Iterable["MiddlewareFactory"] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: None,
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentReply[str, TAgent]": ...

    @overload
    async def ask(
        self,
        *msg: str | Input,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        middleware: Iterable["MiddlewareFactory"] = ...,
        observers: Iterable[Observer] = ...,
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentReply[TAgent, TAgent]": ...

    async def ask(
        self,
        *msg: str | Input,
        dependencies: dict[Any, Any] | None = None,
        variables: dict[Any, Any] | None = None,
        prompt: Iterable[str] = (),
        config: ModelConfig | None = None,
        tools: Iterable[Tool] = (),
        middleware: Iterable["MiddlewareFactory"] = (),
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


class Actor(Generic[TResult]):
    """The agentic unit of autogen.beta.

    An Actor runs a model loop, invokes tools, honours middleware, surfaces
    events through observers, and optionally runs the harness primitives
    (assembly, compaction, aggregation, knowledge store, subtask spawning).

    A bare ``Actor(name, config=cfg)`` has zero harness middleware and
    behaves exactly like a plain LLM loop. Harness features are opt-in:

    * ``assembly=`` — assembly policies (e.g. ``ConversationPolicy``,
      ``SlidingWindow``, ``AlertPolicy``). When non-empty,
      ``AssemblerMiddleware`` and ``_HaltCheckMiddleware`` are wired in.
    * ``knowledge=KnowledgeConfig(store=...)`` — persistent knowledge store,
      compaction, aggregation.
    * ``tasks=TaskConfig(...)`` — override LLM config/prompt for the auto
      injected ``run_subtask`` / ``run_subtasks`` tools.
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
        middleware: Iterable["MiddlewareFactory"] = ...,
        observers: Iterable[Observer] = ...,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        response_schema: type[TResult],
        plugins: Iterable["Plugin"] = ...,
        knowledge: KnowledgeConfig | None = ...,
        tasks: TaskConfig | None = ...,
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
        middleware: Iterable["MiddlewareFactory"] = ...,
        observers: Iterable[Observer] = ...,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        response_schema: ResponseProto[TResult],
        plugins: Iterable["Plugin"] = ...,
        knowledge: KnowledgeConfig | None = ...,
        tasks: TaskConfig | None = ...,
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
        middleware: Iterable["MiddlewareFactory"] = ...,
        observers: Iterable[Observer] = ...,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        response_schema: types.UnionType,
        plugins: Iterable["Plugin"] = ...,
        knowledge: KnowledgeConfig | None = ...,
        tasks: TaskConfig | None = ...,
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
        middleware: Iterable["MiddlewareFactory"] = ...,
        observers: Iterable[Observer] = ...,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        response_schema: None = ...,
        plugins: Iterable["Plugin"] = ...,
        knowledge: KnowledgeConfig | None = ...,
        tasks: TaskConfig | None = ...,
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
        middleware: Iterable["MiddlewareFactory"] = (),
        observers: Iterable[Observer] = (),
        dependencies: dict[Any, Any] | None = None,
        variables: dict[Any, Any] | None = None,
        response_schema: ResponseProto[TResult] | type[TResult] | types.UnionType | None = None,
        plugins: Iterable["Plugin"] = (),
        knowledge: KnowledgeConfig | None = None,
        tasks: TaskConfig | None = None,
        assembly: Iterable[AssemblyPolicy] = (),
    ):
        self.name = name
        self.config = config

        self._agent_dependencies = dependencies or {}
        self._agent_variables = variables or {}

        self._middleware = list(middleware)
        self._observers = list(observers)

        self.dependency_provider = Provider()
        self.tools: list[FunctionTool] = []
        for t in tools:
            self.add_tool(t)

        self._hitl_hook = wrap_hitl(hitl_hook) if hitl_hook else None
        self.__tool_executor = ToolExecutor()

        self._system_prompt: list[str] = []
        self._dynamic_prompt: list[Callable[[ModelRequest, Context], Awaitable[str]]] = []

        self._response_schema = ResponseSchema.ensure_schema(response_schema)

        if isinstance(prompt, str) or callable(prompt):
            prompt = [prompt]

        for p in prompt:
            if isinstance(p, str):
                self._system_prompt.append(p)
            else:
                self._dynamic_prompt.append(_wrap_prompt_hook(p))

        for p in plugins:
            p.register(self)

        # Task spawning
        tc = tasks or TaskConfig(config=config)
        self._task_config: ModelConfig | None = tc.config or config
        self._task_prompt = tc.prompt

        # Knowledge store + compaction/aggregation strategies
        kc = knowledge
        self._knowledge_store = kc.store if kc else None
        self._bootstrap = kc.bootstrap if kc else None
        self._bootstrap_done: bool = False
        self._bootstrap_lock: asyncio.Lock | None = None
        self._compact_strategy = kc.compact if kc else None
        self._compact_trigger = kc.compact_trigger if kc and kc.compact_trigger else CompactTrigger()
        self._aggregate_strategy = kc.aggregate if kc else None
        self._aggregate_trigger = kc.aggregate_trigger if kc and kc.aggregate_trigger else AggregateTrigger()

        # Assembly policies (empty by default; bare Actor has no harness).
        self._policies: list[AssemblyPolicy] = list(assembly)
        if self._policies:
            for w in AssemblerMiddleware.validate_order(self._policies):
                logger.warning("Assembly policy ordering: %s", w)

    def hitl_hook(self, func: HumanHook) -> HumanHook:
        if self._hitl_hook is not None:
            warnings.warn(
                "You already set HITL hook, provided value overrides it",
                category=RuntimeWarning,
                stacklevel=2,
            )

        self._hitl_hook = wrap_hitl(func)
        return func

    @overload
    def prompt(
        self,
        func: None = None,
    ) -> Callable[[PromptHook], PromptHook]: ...

    @overload
    def prompt(
        self,
        func: PromptHook,
    ) -> PromptHook: ...

    def prompt(
        self,
        func: PromptHook | None = None,
    ) -> PromptHook | Callable[[PromptHook], PromptHook]:
        def wrapper(f: PromptHook) -> PromptHook:
            self._dynamic_prompt.append(_wrap_prompt_hook(f))
            return f

        if func:
            return wrapper(func)
        return wrapper

    @overload
    def tool(
        self,
        function: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        schema: FunctionParameters | None = None,
        sync_to_thread: bool = True,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> Tool: ...

    @overload
    def tool(
        self,
        function: None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        schema: FunctionParameters | None = None,
        sync_to_thread: bool = True,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> Callable[[Callable[..., Any]], Tool]: ...

    def add_middleware(self, m: MiddlewareFactory) -> "Actor[TResult]":
        """Append middleware as the innermost wrapper in the chain.

        The added middleware is called last on turn entry and first on turn exit,
        executing closer to the LLM call than any middleware already registered.
        """
        self._middleware.append(m)
        return self

    def insert_middleware(self, m: MiddlewareFactory) -> "Actor[TResult]":
        """Insert middleware as the outermost wrapper in the chain.

        The inserted middleware is called first on turn entry and last on turn exit,
        executing before all middleware already registered on the actor.
        """
        self._middleware.insert(0, m)
        return self

    def add_tool(self, t: Callable[..., Any] | Tool) -> "Actor[TResult]":
        self.tools.append(FunctionTool.ensure_tool(t, provider=self.dependency_provider))
        return self

    def add_observer(self, observer: Observer) -> None:
        """Register an observer (before calling ask())."""
        self._observers.append(observer)

    def tool(
        self,
        function: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        schema: FunctionParameters | None = None,
        sync_to_thread: bool = True,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> Tool | Callable[[Callable[..., Any]], Tool]:
        def make_tool(f: Callable[..., Any]) -> Tool:
            t = tool(
                f,
                name=name,
                description=description,
                schema=schema,
                sync_to_thread=sync_to_thread,
                middleware=middleware,
            )
            self.add_tool(t)
            return t

        if function:
            return make_tool(function)

        return make_tool

    @overload
    def observer(
        self,
        condition: ClassInfo | Condition,
        callback: Callable[..., Any],
    ) -> Callable[..., Any]: ...

    @overload
    def observer(
        self,
        condition: ClassInfo | Condition,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...

    def observer(
        self,
        condition: ClassInfo | Condition,
        callback: Callable[..., Any] | None = None,
    ) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
        def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
            obs = observer_factory(condition, func)
            self._observers.append(obs)
            return func

        if callback is not None:
            return wrapper(callback)
        return wrapper

    @overload
    async def ask(
        self,
        *msg: str | Input,
        stream: Stream | None = ...,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        middleware: Iterable["MiddlewareFactory"] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: type[T2],
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentReply[T2, TResult]": ...

    @overload
    async def ask(
        self,
        msg: str | Input,
        *,
        stream: Stream | None = ...,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        middleware: Iterable["MiddlewareFactory"] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: ResponseProto[T2],
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentReply[T2, TResult]": ...

    @overload
    async def ask(
        self,
        msg: str | Input,
        *,
        stream: Stream | None = ...,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        middleware: Iterable["MiddlewareFactory"] = ...,
        observers: Iterable[Observer] = ...,
        response_schema: None,
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentReply[str, TResult]": ...

    @overload
    async def ask(
        self,
        msg: str | Input,
        *,
        stream: Stream | None = ...,
        dependencies: dict[Any, Any] | None = ...,
        variables: dict[Any, Any] | None = ...,
        prompt: Iterable[str] = ...,
        config: ModelConfig | None = ...,
        tools: Iterable[Tool] = ...,
        middleware: Iterable["MiddlewareFactory"] = ...,
        observers: Iterable[Observer] = ...,
        hitl_hook: HumanHook | None = ...,
    ) -> "AgentReply[TResult, TResult]": ...

    async def ask(
        self,
        *msg: str | Input,
        stream: Stream | None = None,
        dependencies: dict[Any, Any] | None = None,
        variables: dict[Any, Any] | None = None,
        prompt: Iterable[str] = (),
        config: ModelConfig | None = None,
        tools: Iterable[Tool] = (),
        middleware: Iterable["MiddlewareFactory"] = (),
        observers: Iterable[Observer] = (),
        response_schema: Omittable[ResponseProto[Any] | type | None] = omit,
        hitl_hook: HumanHook | None = None,
    ) -> "AgentReply[Any, Any]":
        config = config or self.config
        if not config:
            raise ConfigNotProvidedError()
        client = config.create()

        stream = stream or MemoryStream()

        initial_event = ModelRequest.ensure_request(msg)

        context = Context(
            stream,
            prompt=list(prompt),
            dependencies=self._agent_dependencies | (dependencies or {}),
            variables=self._agent_variables | (variables or {}),
            dependency_provider=self.dependency_provider,
        )

        if not context.prompt:
            context.prompt.extend(self._system_prompt)

            for dp in self._dynamic_prompt:
                p = await dp(initial_event, context)
                context.prompt.append(p)

        return await self._execute(
            initial_event,
            context=context,
            client=client,
            hitl_hook=hitl_hook,
            additional_tools=tools,
            additional_middleware=middleware,
            additional_observers=observers,
            response_schema=response_schema,
        )

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

    def _build_subtask_tools(self) -> list[Tool]:
        """Create ``run_subtask`` / ``run_subtasks`` tools.

        Each invocation spawns a fresh, bare ``Actor`` (configured by
        ``TaskConfig``) and delegates via
        :func:`~autogen.beta.tools.subagents.run_task.run_task`, which owns
        ``TaskStarted`` / ``TaskCompleted`` / ``TaskFailed`` emission,
        dependency + variable copy, HITL bridging, and depth counting.
        """
        actor = self

        @tool
        async def run_subtask(task: str, ctx: Context) -> str:
            """Run a subtask agent to handle isolated compute work autonomously.

            Use this when a sub-problem can be researched or solved
            independently. The subtask agent runs with its own LLM context
            and returns the result.
            """
            return await actor._spawn_subtask(task, ctx)

        @tool
        async def run_subtasks(ctx: Context, tasks: list[str], parallel: bool = True) -> str:
            """Run multiple subtask agents at once.

            Args:
                ctx: The context to run the tasks in.
                tasks: List of subtask descriptions.
                parallel: Run concurrently (default True) or sequentially.
            """
            if parallel:
                raw = await asyncio.gather(
                    *(actor._spawn_subtask(t, ctx) for t in tasks),
                    return_exceptions=True,
                )
                results = [r if not isinstance(r, BaseException) else f"Error: {r}" for r in raw]
            else:
                results = []
                for t in tasks:
                    try:
                        results.append(await actor._spawn_subtask(t, ctx))
                    except Exception as e:
                        results.append(f"Error: {e}")

            parts = []
            for task_desc, result in zip(tasks, results):
                parts.append(f"## {task_desc}\n\n{result}")
            return "\n\n---\n\n".join(parts)

        return [run_subtask, run_subtasks]

    async def _spawn_subtask(self, task: str, ctx: Context) -> str:
        """Spawn a fresh bare ``Actor`` and delegate via ``run_task``.

        A new ``Actor`` is constructed on every call so sibling subtasks
        do not accumulate state between invocations. ``run_task`` emits the
        ``TaskStarted`` / ``TaskCompleted`` / ``TaskFailed`` lifecycle
        events and handles dependency/variable copy and HITL bridging.
        """
        from .tools.subagents.run_task import run_task

        bare = Actor(
            name=f"subtask-{uuid4().hex[:8]}",
            prompt=self._task_prompt,
            config=self._task_config,
        )

        result = await run_task(bare, task, parent_context=ctx)
        if not result.completed:
            return f"Error: {result.error}"
        return result.result or ""

    async def _execute(
        self,
        event: BaseEvent,
        *,
        context: Context,
        client: LLMClient,
        hitl_hook: HumanHook | None = None,
        additional_tools: Iterable[Tool] = (),
        additional_middleware: Iterable["MiddlewareFactory"] = (),
        additional_observers: Iterable[Observer] = (),
        response_schema: Omittable[ResponseProto[Any] | type | None] = omit,
    ) -> "AgentReply[Any, Any]":
        additional_observers = list(additional_observers)
        subtask_tools = self._build_subtask_tools()

        # Bootstrap the knowledge store on first use, guarded by an asyncio
        # lock so concurrent asks on the same Actor can't double-bootstrap.
        # The lock is created lazily so Actor can be instantiated outside an
        # event loop (asyncio.Lock binds to the running loop on first use).
        if self._knowledge_store and not self._bootstrap_done:
            if self._bootstrap_lock is None:
                self._bootstrap_lock = asyncio.Lock()
            async with self._bootstrap_lock:
                if not self._bootstrap_done:
                    if not await self._knowledge_store.exists("/.initialized"):
                        await self._knowledge_store.write("/.initialized", self.name)
                        bootstrap = self._bootstrap or DefaultBootstrap()
                        await bootstrap.bootstrap(self._knowledge_store, self.name)
                    self._bootstrap_done = True

        knowledge_tools = self._build_knowledge_tool() if self._knowledge_store else []

        if self._knowledge_store:
            context.dependencies[KnowledgeStore] = self._knowledge_store

        all_observers = list(chain(self._observers, additional_observers))

        # Build harness middleware chain. Assembler + halt-check only wire in
        # when the user has provided assembly policies; compaction and
        # aggregation middleware have independent gates.
        harness_middleware: list[MiddlewareFactory] = []

        if self._policies:
            harness_middleware.append(_AssemblerMiddlewareFactory(self._policies))
            harness_middleware.append(_HaltCheckMiddlewareFactory())

        if self._compact_strategy:
            harness_middleware.append(
                _CompactionMiddlewareFactory(
                    self.name,
                    self._compact_strategy,
                    self._knowledge_store,
                    self._compact_trigger,
                )
            )

        if self._aggregate_strategy and self._knowledge_store:
            trigger = self._aggregate_trigger
            if trigger.every_n_turns > 0 or trigger.every_n_events > 0 or trigger.on_end:
                harness_middleware.append(
                    _AggregationMiddlewareFactory(
                        self.name,
                        self._aggregate_strategy,
                        self._knowledge_store,
                        trigger,
                    )
                )

        try:
            if response_schema is omit:
                final_schema = self._response_schema
            else:
                final_schema = ResponseSchema.ensure_schema(response_schema)

            all_tools: list[Tool] = list(
                chain(
                    self.tools,
                    additional_tools,
                    subtask_tools,
                    knowledge_tools,
                )
            )

            all_schemas: list[ToolSchema] = []
            known_tools: set[str] = set()
            for t in all_tools:
                schemas = await t.schemas(context)
                all_schemas.extend(schemas)

                for schema in schemas:
                    if isinstance(schema, FunctionToolSchema):
                        known_tools.add(schema.function.name)
                    elif isinstance(schema, WebSearchToolSchema):
                        known_tools.add(WEB_SEARCH_TOOL_NAME)

            middleware_instances: list[BaseMiddleware] = []
            agent_turn: AgentTurn = _execute_turn
            llm_call: LLMCall = partial(
                client,
                tools=all_schemas,
                response_schema=final_schema,
            )

            for m in reversed(
                list(
                    chain(
                        self._middleware,
                        harness_middleware,
                        additional_middleware,
                    )
                )
            ):
                mw = m(event, context)
                middleware_instances.append(mw)

                agent_turn = partial(mw.on_turn, agent_turn)
                llm_call = partial(mw.on_llm_call, llm_call)

            async def _call_client(context: Context) -> None:
                messages = await context.stream.history.get_events()
                result = await llm_call(messages, context)
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

                self.__tool_executor.register(
                    stack,
                    context,
                    tools=all_tools,
                    known_tools=known_tools,
                    middleware=middleware_instances,
                )

                for obs in all_observers:
                    obs.register(stack, context)

                # Observers are live — emit Started so they can see their own
                # lifecycle event if they subscribe to it.
                for obs in all_observers:
                    await context.send(
                        ObserverStarted(name=getattr(obs, "name", type(obs).__name__))
                    )

                try:
                    message = await agent_turn(event, context)
                    reply = AgentReply(
                        message,
                        context=context,
                        agent=self,
                        client=client,
                        provider=self.dependency_provider,
                        response_schema=final_schema,
                    )
                finally:
                    # Emit Completed while observers are still registered,
                    # so observers subscribed to their own lifecycle event
                    # see it before the ExitStack unregisters them.
                    for obs in all_observers:
                        with suppress(Exception):
                            await context.send(
                                ObserverCompleted(name=getattr(obs, "name", type(obs).__name__))
                            )

                return reply
        finally:
            if self._knowledge_store:
                try:
                    events = list(await context.stream.history.get_events())
                    await EventLogWriter(self._knowledge_store).persist(context.stream.id, events)
                except Exception:
                    logger.exception("Event log persistence failed for %s", self.name)

    def as_tool(
        self,
        *,
        description: str,
        name: str | None = None,
        stream: "StreamFactory | None" = None,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        from .tools.subagents import subagent_tool

        return subagent_tool(
            self,
            description=description,
            name=name,
            stream=stream,
            middleware=middleware,
        )

    def as_conversable(self) -> "ConversableAdapter":
        from .conversable import ConversableAdapter

        return ConversableAdapter(self)


async def _execute_turn(event: BaseEvent, context: Context) -> ModelResponse:
    async with context.stream.get(ModelResponse) as result:
        await context.send(event)
        message: ModelResponse = await result

    while message.tool_calls and not message.response_force:
        async with context.stream.get(ModelResponse) as result:
            await context.send(message.tool_calls)
            message = await result

    return message


def _wrap_prompt_hook(func: PromptHook) -> Callable[[ModelRequest, Context], Awaitable[str]]:
    call_model = build_model(func)

    async def wrapper(event: ModelRequest, context: Context) -> str:
        async with AsyncExitStack() as stack:
            r = await call_model.asolve(
                event,
                stack=stack,
                cache_dependencies={},
                dependency_provider=context.dependency_provider,
                **{CONTEXT_OPTION_NAME: context},
            )
        return r

    return wrapper


class Plugin:
    def __init__(
        self,
        *,
        prompt: PromptType | Iterable[PromptType] = (),
        hitl_hook: HumanHook | None = None,
        tools: Iterable[Callable[..., Any] | Tool] = (),
        middleware: Iterable[MiddlewareFactory] = (),
        observers: Iterable[Observer] = (),
        dependencies: dict[Any, Any] | None = None,
        variables: dict[Any, Any] | None = None,
    ) -> None:
        self._tools = list(tools)
        self._middleware = list(middleware)
        self._observers = list(observers)
        self._dependencies = dependencies or {}
        self._variables = variables or {}
        self._hitl_hook = hitl_hook

        self._system_prompt: list[str] = []
        self._dynamic_prompt: list[Callable[[ModelRequest, Context], Awaitable[str]]] = []

        if isinstance(prompt, str) or callable(prompt):
            prompt = [prompt]
        for p in prompt:
            if isinstance(p, str):
                self._system_prompt.append(p)
            else:
                self._dynamic_prompt.append(_wrap_prompt_hook(p))

    def register(self, actor: "Actor[Any]") -> None:
        """Apply this plugin's contributions to an Actor instance."""
        for t in self._tools:
            actor.add_tool(t)

        for m in self._middleware:
            actor.add_middleware(m)

        if self._hitl_hook is not None:
            if actor._hitl_hook is not None:
                warnings.warn(
                    f"Actor '{actor.name}' already has a HITL hook; the plugin's hook will be ignored.",
                    stacklevel=2,
                )
            else:
                actor._hitl_hook = wrap_hitl(self._hitl_hook)

        actor._agent_dependencies = self._dependencies | actor._agent_dependencies
        actor._agent_variables.update(self._variables)

        actor._observers.extend(self._observers)
        actor._system_prompt.extend(self._system_prompt)
        actor._dynamic_prompt.extend(self._dynamic_prompt)

    def hitl_hook(self, func: HumanHook) -> HumanHook:
        if self._hitl_hook is not None:
            warnings.warn(
                "You already set HITL hook, provided value overrides it",
                category=RuntimeWarning,
                stacklevel=2,
            )
        self._hitl_hook = func
        return func

    @overload
    def prompt(
        self,
        func: PromptHook,
    ) -> PromptHook: ...

    @overload
    def prompt(
        self,
        func: None = None,
    ) -> Callable[[PromptHook], PromptHook]: ...

    def prompt(
        self,
        func: PromptHook | None = None,
    ) -> PromptHook | Callable[[PromptHook], PromptHook]:
        def wrapper(f: PromptHook) -> PromptHook:
            self._dynamic_prompt.append(_wrap_prompt_hook(f))
            return f

        if func:
            return wrapper(func)
        return wrapper

    @overload
    def tool(
        self,
        function: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        schema: FunctionParameters | None = None,
        sync_to_thread: bool = True,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool: ...

    @overload
    def tool(
        self,
        function: None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        schema: FunctionParameters | None = None,
        sync_to_thread: bool = True,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> Callable[[Callable[..., Any]], FunctionTool]: ...

    def tool(
        self,
        function: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        schema: FunctionParameters | None = None,
        sync_to_thread: bool = True,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool | Callable[[Callable[..., Any]], FunctionTool]:
        def make_tool(f: Callable[..., Any]) -> FunctionTool:
            t = tool(
                f,
                name=name,
                description=description,
                schema=schema,
                sync_to_thread=sync_to_thread,
                middleware=middleware,
            )
            self._tools.append(t)
            return t

        if function:
            return make_tool(function)
        return make_tool

    @overload
    def observer(
        self,
        condition: ClassInfo | Condition,
        callback: Callable[..., Any],
    ) -> Callable[..., Any]: ...

    @overload
    def observer(
        self,
        condition: ClassInfo | Condition,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...

    def observer(
        self,
        condition: ClassInfo | Condition,
        callback: Callable[..., Any] | None = None,
    ) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
        def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
            obs = observer_factory(condition, func)
            self._observers.append(obs)
            return func

        if callback is not None:
            return wrapper(callback)
        return wrapper


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
        # Count only non-transient events — transient events (chunks, lifecycle)
        # should not influence compaction decisions even if persist_all=True.
        conversation_events = [e for e in events if not getattr(type(e), "__transient__", False)]
        event_count = len(conversation_events)

        # Prevent double compaction — skip if count hasn't grown since last
        if event_count <= self._last_compact_event_count:
            return result

        should_compact = False
        if self._trigger.max_events > 0 and event_count > self._trigger.max_events:
            should_compact = True
        if self._trigger.max_tokens > 0:
            estimated = sum(len(str(e)) for e in conversation_events) // self._trigger.chars_per_token
            if estimated > self._trigger.max_tokens:
                should_compact = True

        if should_compact:
            compacted = await self._strategy.compact(events, context, self._store)
            await context.stream.history.replace(compacted)
            self._last_compact_event_count = len([e for e in compacted if not getattr(type(e), "__transient__", False)])

            usage = getattr(self._strategy, "last_usage", {})
            await context.send(
                CompactionCompleted(
                    actor=self._actor_name,
                    strategy=type(self._strategy).__name__,
                    events_before=len(events),
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
        count_before = len(list(await context.stream.history.get_events()))

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
                if count_after // threshold > count_before // threshold:
                    should_aggregate = True

            if should_aggregate:
                try:
                    await self._strategy.aggregate(events_after, context, self._store)
                    usage = getattr(self._strategy, "last_usage", {})
                    await context.send(
                        AggregationCompleted(
                            actor=self._actor_name,
                            strategy=type(self._strategy).__name__,
                            event_count=count_after,
                            llm_calls=1 if usage else 0,
                            usage=usage,
                        )
                    )
                except Exception:
                    logger.exception("Aggregation failed for %s", self._actor_name)

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
