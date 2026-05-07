# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Protocol

from autogen.beta.context import ConversationContext, Stream
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
        instructions: str | None = None,
    ) -> AbstractAsyncContextManager[None]: ...


class LiveAgent:
    """Async context manager that opens a realtime STT session.

    If `stream` is omitted, owns a fresh `MemoryStream`; otherwise binds to
    the supplied one. Entering yields the owned `ConversationContext` so
    peers (Player, Recorder) can share it.

    `prompt` is lowered into the provider's session as `instructions` when
    the session is opened.
    """

    def __init__(
        self,
        config: RealtimeSTTConfig,
        *,
        prompt: str | None = None,
        stream: Stream | None = None,
    ) -> None:
        self._config = config
        self._prompt = prompt
        self._stream = stream
        self._session: AbstractAsyncContextManager[None] | None = None

    async def __aenter__(self) -> ConversationContext:
        if self._stream is None:
            self._stream = MemoryStream()
        context = ConversationContext(stream=self._stream)
        self._session = self._config.session(context, instructions=self._prompt)
        await self._session.__aenter__()
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
