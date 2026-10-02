# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib

from ag2.knowledge import MemoryKnowledgeStore

__all__ = ("RacingKnowledgeStore",)


class RacingKnowledgeStore(MemoryKnowledgeStore):
    """A store that makes two readers of one path overlap, as a networked store does.

    The first read of a path ending in ``path_suffix`` returns its value only
    once a second read has started, or after ``overlap_wait`` seconds.
    """

    def __init__(self, path_suffix: str, *, overlap_wait: float = 0.2) -> None:
        super().__init__()
        self._path_suffix = path_suffix
        self._overlap_wait = overlap_wait
        self._readers = 0
        self._second_reader = asyncio.Event()

    async def read(self, path: str) -> str | None:
        content = await super().read(path)
        if path.endswith(self._path_suffix):
            self._readers += 1
            if self._readers >= 2:
                self._second_reader.set()
            else:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._second_reader.wait(), self._overlap_wait)
        return content
