# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from types import TracebackType
from typing import Any, Protocol

from fast_depends import Provider

from autogen.beta.agent import HumanHook, PromptType, _wrap_prompt_hook, wrap_hitl
from autogen.beta.context import ConversationContext, Stream
from autogen.beta.events import HumanInputRequest, ModelRequest
from autogen.beta.stream import MemoryStream


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
    ) -> AbstractAsyncContextManager[None]: ...


class LiveAgent:
    """Async context manager that opens a realtime STT session.

    If `stream` is omitted, owns a fresh `MemoryStream`; otherwise binds to
    the supplied one. Entering yields the owned `ConversationContext` so
    peers (Player, Recorder) can share it.

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
    ) -> None:
        self.name = name

        self._config = config
        self._stream = stream
        self._session: AsyncExitStack | None = None

        self._dependencies: dict[Any, Any] = dependencies or {}
        self._variables: dict[Any, Any] = variables or {}
        self._hitl_hook = wrap_hitl(hitl_hook) if hitl_hook else None

        if isinstance(prompt, str) or callable(prompt):
            prompt = [prompt]
        self._prompt: list[PromptType] = list(prompt)

    async def __aenter__(self) -> ConversationContext:
        if self._stream is None:
            self._stream = MemoryStream()

        context = ConversationContext(
            stream=self._stream,
            dependency_provider=Provider(),
            dependencies=self._dependencies,
            variables=self._variables,
        )

        s = self._session = await AsyncExitStack().__aenter__()

        if self._hitl_hook is not None:
            s.enter_context(
                self._stream.where(HumanInputRequest).sub_scope(
                    self._hitl_hook(()),
                    interrupt=True,
                ),
            )

        instructions = await self._resolve_instructions(context)
        await s.enter_async_context(self._config.session(context, instructions=instructions))

        return context

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._session is None:
            return

        session, self._session = self._session, None
        await session.__aexit__(exc_type, exc_value, traceback)

    async def _resolve_instructions(self, context: ConversationContext) -> list[str]:
        request = ModelRequest([])
        parts: list[str] = []
        for p in self._prompt:
            if isinstance(p, str):
                parts.append(p)
            else:
                parts.append(await _wrap_prompt_hook(p)(request, context))
        return parts
