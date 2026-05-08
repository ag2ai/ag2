# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager, suppress
from typing import Any, Protocol, overload

from fast_depends import Provider
from fast_depends.pydantic import PydanticSerializer

from autogen.beta.agent import HumanHook, Plugin, PromptHook, PromptType, _wrap_prompt_hook, wrap_hitl
from autogen.beta.context import ConversationContext, Stream
from autogen.beta.events import HumanInputRequest, ModelRequest, ObserverCompleted, ObserverStarted
from autogen.beta.middleware import ToolMiddleware
from autogen.beta.observers import Observer
from autogen.beta.stream import MemoryStream
from autogen.beta.tools.executor import ToolExecutor
from autogen.beta.tools.final import FunctionParameters, FunctionTool, FunctionToolSchema
from autogen.beta.tools.final import tool as _tool
from autogen.beta.tools.schemas import ToolSchema
from autogen.beta.tools.tool import Tool


class RealtimeSTTConfig(Protocol):
    """A speech-to-text config that holds an open bidirectional session.

    Unlike `STTConfig` (one-shot transcribe), realtime configs run for the
    duration of the `session()` context manager. The session subscribes to
    `RecordedAudioEvent` on the supplied context's stream, pumps captured
    audio into the provider, and emits transcription events back onto the
    same stream.

    Framework-level concepts (such as the agent's prompt) flow in via the
    keyword parameters of `session()`, allowing `LiveAgent` to inject them
    into the provider's session payload at startup.
    """

    def session(
        self,
        context: ConversationContext,
        *,
        instructions: Iterable[str] = (),
        tools: Iterable[ToolSchema] = (),
    ) -> AbstractAsyncContextManager[None]: ...


class LiveAgent:
    """Realtime STT agent. Open a session via `agent.run()`.

    If `stream` is omitted, owns a fresh `MemoryStream`; otherwise binds to
    the supplied one. `run()` is an async context manager that yields the
    owned `ConversationContext` so peers (Player, Recorder) can share it.

    `prompt` accepts the same shapes as `Agent.prompt` — a string, a
    `PromptHook` callable, or any iterable mixing both. Callable hooks are
    resolved once at session open against the `ConversationContext` (no
    `ModelRequest` — realtime is session-scoped, not request-scoped). The
    resulting iterable of strings is forwarded as `instructions` to the
    provider's session, which is responsible for joining them.
    """

    def __init__(
        self,
        name: str,
        prompt: PromptType | Iterable[PromptType] = (),
        *,
        config: RealtimeSTTConfig,
        hitl_hook: HumanHook | None = None,
        tools: Iterable[Callable[..., Any] | Tool] = (),
        # middleware
        observers: Iterable[Observer] = (),
        dependencies: dict[Any, Any] | None = None,
        variables: dict[Any, Any] | None = None,
        # response_schema
        plugins: Iterable[Plugin] = (),
        # knowledge
        # tasks
        # assembly
        stream: Stream | None = None,
    ) -> None:
        self.name = name

        self._config = config
        self._stream = stream

        self._dependencies: dict[Any, Any] = dependencies or {}
        self._variables: dict[Any, Any] = variables or {}
        self._hitl_hook = wrap_hitl(hitl_hook) if hitl_hook else None

        self._dependency_provider = Provider()
        self._tools: list[Tool] = []
        for t in tools:
            self.add_tool(t)

        self._observers: list[Observer] = []
        for obs in observers:
            self.add_observer(obs)

        self._tool_executor = ToolExecutor(
            PydanticSerializer(
                pydantic_config={"arbitrary_types_allowed": True},
                use_fastdepends_errors=False,
            ),
        )

        if isinstance(prompt, str) or callable(prompt):
            prompt = [prompt]
        self._prompt: list[PromptType] = list(prompt)

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
            self._prompt.append(f)
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
            t = _tool(
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

    def add_tool(self, t: Callable[..., Any] | Tool) -> "LiveAgent":
        self._tools.append(FunctionTool.ensure_tool(t, provider=self._dependency_provider))
        return self

    def add_observer(self, observer: Observer) -> None:
        """Register an observer (before calling run())."""
        self._observers.append(observer)

    @asynccontextmanager
    async def run(
        self,
        *,
        dependencies: dict[Any, Any] | None = None,
        variables: dict[Any, Any] | None = None,
        prompt: Iterable[str] = (),
        config: RealtimeSTTConfig | None = None,
        tools: Iterable[Callable[..., Any] | Tool] = (),
        observers: Iterable[Observer] = (),
        hitl_hook: HumanHook | None = None,
    ) -> AsyncIterator[ConversationContext]:
        stream = self._stream if self._stream is not None else MemoryStream()

        context = ConversationContext(
            stream=stream,
            dependency_provider=self._dependency_provider,
            dependencies=self._dependencies | (dependencies or {}),
            variables=self._variables | (variables or {}),
        )

        active_config = config if config is not None else self._config
        active_hitl = wrap_hitl(hitl_hook) if hitl_hook else self._hitl_hook

        all_tools: list[Tool] = self._tools + [
            FunctionTool.ensure_tool(t, provider=self._dependency_provider) for t in tools
        ]
        all_observers: list[Observer] = self._observers + list(observers)

        async with AsyncExitStack() as s:
            if active_hitl is not None:
                s.enter_context(
                    stream.where(HumanInputRequest).sub_scope(
                        active_hitl(()),
                        interrupt=True,
                    ),
                )

            all_schemas: list[ToolSchema] = []
            known_tools: set[str] = set()
            for t in all_tools:
                schemas = await t.schemas(context)
                all_schemas.extend(schemas)
                for schema in schemas:
                    if isinstance(schema, FunctionToolSchema):
                        known_tools.add(schema.function.name)
                    else:
                        known_tools.add(schema.type)

            if all_tools:
                self._tool_executor.register(
                    s,
                    context,
                    tools=all_tools,
                    known_tools=known_tools,
                )

            instructions = list(prompt) if prompt else await self._resolve_instructions(context)
            await s.enter_async_context(
                active_config.session(context, instructions=instructions, tools=all_schemas),
            )

            for obs in all_observers:
                obs.register(s, context)

            for obs in all_observers:
                await context.send(ObserverStarted(name=getattr(obs, "name", type(obs).__name__)))

            try:
                yield context

            finally:
                for obs in all_observers:
                    with suppress(Exception):
                        await context.send(
                            ObserverCompleted(name=getattr(obs, "name", type(obs).__name__)),
                        )

    async def _resolve_instructions(self, context: ConversationContext) -> list[str]:
        request = ModelRequest([])
        parts: list[str] = []
        for p in self._prompt:
            if isinstance(p, str):
                parts.append(p)
            else:
                parts.append(await _wrap_prompt_hook(p)(request, context))
        return parts
