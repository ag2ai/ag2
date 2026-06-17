# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager, suppress
from typing import Any, Protocol

from fast_depends.library.serializer import SerializerProto

from autogen.beta.agent import HumanHook, Plugin, PluginTarget, PromptType, wrap_hitl
from autogen.beta.context import ConversationContext, Stream
from autogen.beta.events import HumanInputRequest, ModelRequest, ObserverCompleted, ObserverStarted
from autogen.beta.middleware.base import BaseMiddleware, MiddlewareFactory
from autogen.beta.observers import Observer
from autogen.beta.stream import MemoryStream
from autogen.beta.tools.final import FunctionTool, FunctionToolSchema
from autogen.beta.tools.schemas import ToolSchema
from autogen.beta.tools.tool import Tool
from autogen.beta.usage import UsageReport


class RealtimeConfig(Protocol):
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
        serializer: SerializerProto,
    ) -> AbstractAsyncContextManager[None]: ...


class LiveAgent(PluginTarget):
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
        config: RealtimeConfig,
        hitl_hook: HumanHook | None = None,
        tools: Iterable[Callable[..., Any] | Tool] = (),
        middleware: Iterable[MiddlewareFactory] = (),
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
        self._config = config
        self._stream = stream

    @staticmethod
    async def usage_report(context: ConversationContext) -> UsageReport:
        """Aggregate token usage over the live session's event log."""
        events = await context.stream.history.get_events()
        return UsageReport.from_events(events)

    @asynccontextmanager
    async def run(
        self,
        *,
        dependencies: dict[Any, Any] | None = None,
        variables: dict[Any, Any] | None = None,
        prompt: Iterable[str] = (),
        config: RealtimeConfig | None = None,
        tools: Iterable[Callable[..., Any] | Tool] = (),
        middleware: Iterable[MiddlewareFactory] = (),
        observers: Iterable[Observer] = (),
        hitl_hook: HumanHook | None = None,
    ) -> AsyncIterator[ConversationContext]:
        stream = self._stream if self._stream is not None else MemoryStream()

        context = ConversationContext(
            stream=stream,
            dependency_provider=self.dependency_provider,
            dependencies=self._agent_dependencies | (dependencies or {}),
            variables=self._agent_variables | (variables or {}),
        )

        active_config = config if config is not None else self._config
        active_hitl = wrap_hitl(hitl_hook) if hitl_hook else self._hitl_hook

        all_tools: list[Tool] = self.tools + [
            FunctionTool.ensure_tool(t, provider=self.dependency_provider) for t in tools
        ]
        all_observers: list[Observer] = self._observers + list(observers)

        initial_event = ModelRequest([])
        middleware_instances: list[BaseMiddleware] = [
            m(initial_event, context) for m in (*self._middleware, *middleware)
        ]

        async with AsyncExitStack() as s:
            if active_hitl is not None:
                s.enter_context(
                    stream.where(HumanInputRequest).sub_scope(
                        active_hitl(middleware_instances),
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
                    middleware=middleware_instances,
                )

            instructions = list(prompt) if prompt else await self._resolve_instructions(context)

            # enter Provider session
            await s.enter_async_context(
                active_config.session(
                    context,
                    instructions=instructions,
                    tools=all_schemas,
                    serializer=self._serializer,
                )
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
        parts: list[str] = list(self._system_prompt)
        for hook in self._dynamic_prompt:
            parts.append(await hook(request, context))
        return parts
