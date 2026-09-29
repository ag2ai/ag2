# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator, Callable, Iterable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager, suppress
from typing import Any, Protocol

from fast_depends.library.serializer import SerializerProto

from ag2.agent import HumanHook, Plugin, PluginTarget, PromptType, wrap_hitl
from ag2.annotations import Context
from ag2.context import ConversationContext, Stream
from ag2.events import (
    DrainedModelRequest,
    HumanInputRequest,
    Input,
    MessageEnqueued,
    ModelRequest,
    ObserverCompleted,
    ObserverStarted,
)
from ag2.middleware.base import BaseMiddleware, MiddlewareFactory
from ag2.observers import Observer
from ag2.stream import MemoryStream
from ag2.tools.final import FunctionTool, FunctionToolSchema
from ag2.tools.schemas import ToolSchema
from ag2.tools.tool import Tool
from ag2.usage import UsageReport


class RealtimeConfig(Protocol):
    """A config that holds an open bidirectional audio session.

    Unlike `STTConfig` (one-shot transcribe), realtime configs run for the
    duration of the `session()` context manager. The session subscribes to
    `RecordedAudioEvent` on the supplied context's stream, pumps captured
    audio into the provider, and emits transcription events back onto the
    same stream.

    The session also consumes `ModelRequest` from the stream: each one is a
    user turn pushed into the running conversation (typed text, or messages
    `LiveAgent` drains from the inbox). The model answers it at the next point
    the provider allows — immediately when the model is silent, otherwise after
    the current response, without interrupting it. The session adds it to the
    provider's conversation as early as that allows: at once where the provider
    accepts content during a response (OpenAI), at the response boundary where
    any content would cut the response off (Gemini). Nothing enforces this
    subscription, so each provider needs a test that a pushed `ModelRequest`
    reaches its connection. A provider that publishes a `ModelRequest` of its
    own (such as a transcript of captured audio) publishes a marker subclass
    and skips it in this subscription.

    A session accepts at least `TextInput` and `DataInput` parts, sending
    `DataInput` encoded by the `serializer` passed to `session()`. Any part it
    cannot send is treated according to where the request came from:

    - a `DrainedModelRequest` (the inbox `LiveAgent` drained): log a warning,
      drop that part and send the rest, so one bad attachment never fails the
      drain or the session;
    - any other `ModelRequest` (pushed directly on the stream): raise
      `UnsupportedInputError`, which reaches the caller of `context.send`.

    `LiveAgent` needs no separate STT/LLM/TTS parts. For a cascade of
    separate providers, see `STTConfig.pipe` and `TTSObserver`.

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
    """Realtime (Speech-to-Speech/S2S) agent. Open a session via `agent.run()`.

    If `stream` is omitted, owns a fresh `MemoryStream`; otherwise binds to
    the supplied one. `run()` is an async context manager that yields the
    owned `ConversationContext` so peers (Player, Recorder) can share it.

    `prompt` accepts the same shapes as `Agent.prompt` — a string, a
    `PromptHook` callable, or any iterable mixing both. Callable hooks are
    resolved once at session open against the `ConversationContext` (no
    `ModelRequest` — realtime is session-scoped, not request-scoped). The
    resulting iterable of strings is forwarded as `instructions` to the
    provider's session, which is responsible for joining them.

    `context.enqueue(...)` hands the running session a user turn — from the
    caller, a tool, or a background task. The agent drains the stream's inbox
    once the provider's session is open (delivering anything left on a shared
    stream) and again on every `MessageEnqueued`, whether or not the model is
    responding, and publishes what it drained as one `DrainedModelRequest`.
    The provider's session takes it from there and decides when to answer.
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
    ) -> AsyncGenerator[ConversationContext]:
        stream = self._stream if self._stream is not None else MemoryStream()

        context = ConversationContext(
            stream=stream,
            dependency_provider=self.dependency_provider,
            dependencies=self._agent_dependencies | (dependencies or {}),
            variables=self._agent_variables | (variables or {}),
        )

        active_config = config if config is not None else self._config
        active_hitl = wrap_hitl(hitl_hook) if hitl_hook else self._hitl_hook

        all_tools: list[Tool] = self.tools + [FunctionTool.ensure_tool(t) for t in tools]
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

            # The provider's session and the observers see `ModelRequest` from here on.
            s.enter_context(stream.where(MessageEnqueued).sub_scope(_on_message_enqueued))
            await _publish_inbox(context)

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


async def _on_message_enqueued(event: MessageEnqueued, context: Context) -> None:
    await _publish_inbox(context)


async def _publish_inbox(context: ConversationContext) -> None:
    """Publish the inbox as one `DrainedModelRequest`, removing only the messages it published.

    Assumes only the event loop removes from the inbox; other threads only append.
    """
    inbox = context.pending_messages
    count = len(inbox)
    if not count:
        return
    drained = inbox[:count]
    del inbox[:count]
    parts: list[Input] = [part for request in drained for part in request.parts]
    await context.send(DrainedModelRequest(parts))
