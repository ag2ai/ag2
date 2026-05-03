# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Protocol

from autogen.beta.context import ConversationContext, Stream


class RealtimeSTTConfig(Protocol):
    """A speech-to-text config that holds an open bidirectional session.

    Unlike `STTConfig` (one-shot transcribe), realtime configs run for the
    duration of the `session()` context manager — pumping audio into the
    provider and emitting transcription events on the supplied context.
    """

    def session(
        self,
        audio_stream: AsyncIterator[bytes],
        context: ConversationContext,
    ) -> AbstractAsyncContextManager[None]: ...


class LiveTranscription:
    """Async context manager that wires a microphone stream to a realtime STT
    session, emitting `TranscriptionChunkEvent` and `TranscriptionCompletedEvent`
    onto the supplied stream.
    """

    def __init__(
        self,
        audio_stream: AsyncIterator[bytes],
        config: RealtimeSTTConfig,
        *,
        stream: Stream,
    ) -> None:
        self._audio_stream = audio_stream
        self._config = config
        self._stream = stream
        self._session: AbstractAsyncContextManager[None] | None = None

    async def __aenter__(self) -> "LiveTranscription":
        context = ConversationContext(stream=self._stream)
        self._session = self._config.session(self._audio_stream, context)
        await self._session.__aenter__()
        return self

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
