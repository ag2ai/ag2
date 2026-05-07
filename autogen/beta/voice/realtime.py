# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from typing import Any, Protocol

from fast_depends import Provider
from fast_depends.pydantic import PydanticSerializer

from autogen.beta.agent import HumanHook, PromptType, _wrap_prompt_hook, wrap_hitl
from autogen.beta.context import ConversationContext, Stream
from autogen.beta.events import HumanInputRequest, ModelRequest
from autogen.beta.stream import MemoryStream
from autogen.beta.tools.executor import ToolExecutor
from autogen.beta.tools.final import FunctionTool, FunctionToolSchema
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
        stream: Stream | None = None,
        dependencies: dict[Any, Any] | None = None,
        variables: dict[Any, Any] | None = None,
        hitl_hook: HumanHook | None = None,
        tools: Iterable[Callable[..., Any] | Tool] = (),
    ) -> None:
        self.name = name

        self._config = config
        self._stream = stream

        self._dependencies: dict[Any, Any] = dependencies or {}
        self._variables: dict[Any, Any] = variables or {}
        self._hitl_hook = wrap_hitl(hitl_hook) if hitl_hook else None

        self._dependency_provider = Provider()
        self._tools: list[Tool] = [FunctionTool.ensure_tool(t, provider=self._dependency_provider) for t in tools]
        self._tool_executor = ToolExecutor(
            PydanticSerializer(
                pydantic_config={"arbitrary_types_allowed": True},
                use_fastdepends_errors=False,
            ),
        )

        if isinstance(prompt, str) or callable(prompt):
            prompt = [prompt]
        self._prompt: list[PromptType] = list(prompt)

    @asynccontextmanager
    async def run(self) -> AsyncIterator[ConversationContext]:
        stream = self._stream if self._stream is not None else MemoryStream()

        context = ConversationContext(
            stream=stream,
            dependency_provider=self._dependency_provider,
            dependencies=self._dependencies,
            variables=self._variables,
        )

        async with AsyncExitStack() as s:
            if self._hitl_hook is not None:
                s.enter_context(
                    stream.where(HumanInputRequest).sub_scope(
                        self._hitl_hook(()),
                        interrupt=True,
                    ),
                )

            all_schemas: list[ToolSchema] = []
            known_tools: set[str] = set()
            for t in self._tools:
                schemas = await t.schemas(context)
                all_schemas.extend(schemas)
                for schema in schemas:
                    if isinstance(schema, FunctionToolSchema):
                        known_tools.add(schema.function.name)
                    else:
                        known_tools.add(schema.type)

            if self._tools:
                self._tool_executor.register(
                    s,
                    context,
                    tools=self._tools,
                    known_tools=known_tools,
                )

            instructions = await self._resolve_instructions(context)
            await s.enter_async_context(
                self._config.session(context, instructions=instructions, tools=all_schemas),
            )

            yield context

    async def _resolve_instructions(self, context: ConversationContext) -> list[str]:
        request = ModelRequest([])
        parts: list[str] = []
        for p in self._prompt:
            if isinstance(p, str):
                parts.append(p)
            else:
                parts.append(await _wrap_prompt_hook(p)(request, context))
        return parts
