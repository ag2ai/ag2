# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator
from typing import Protocol, TypeVar, runtime_checkable

T1 = TypeVar("T1", contravariant=True)
T2 = TypeVar("T2", covariant=True)


class AudioPlayer(Protocol[T1]):
    async def play(self, content: T1) -> None: ...


class TTSConfig(Protocol[T2]):
    async def synthesize(self, text: str) -> T2: ...


@runtime_checkable
class StreamingTTSConfig(TTSConfig[T2], Protocol[T2]):
    """A `TTSConfig` that can also emit audio incrementally.

    `CascadeConfig` prefers `stream` when the config offers it, so the reply
    starts playing at time-to-first-byte instead of after the last sample.
    Runtime-checkable: the session picks the path by asking whether the config
    has a `stream` method, so any config that grows one is used as streaming.
    """

    def stream(self, text: str) -> AsyncIterator[T2]: ...


@runtime_checkable
class TextStreamingTTSConfig(TTSConfig[T2], Protocol[T2]):
    """A `TTSConfig` that takes a reply's text as it streams, not sentence by sentence.

    `stream_text` is given every piece of one reply as the model produces it and
    yields audio until that text runs out, so the provider decides where to cut
    speech and keeps the voice's context across sentences. `Speech` checks for it
    before `StreamingTTSConfig` and then skips its own sentence buffer.

    The method is not named `stream`: runtime protocol checks only look at method
    names, so a `StreamingTTSConfig` would pass for this one.
    """

    def stream_text(self, text: AsyncIterator[str]) -> AsyncIterator[T2]: ...
