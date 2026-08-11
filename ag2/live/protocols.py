# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol, TypeVar

T1 = TypeVar("T1", contravariant=True)
T2 = TypeVar("T2", covariant=True)


class AudioPlayer(Protocol[T1]):
    async def play(self, content: T1) -> None: ...


class TTSConfig(Protocol[T2]):
    async def synthesize(self, text: str) -> T2: ...
