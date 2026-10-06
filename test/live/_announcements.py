# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

from ag2.events import MessageEnqueued


class Announcements:
    """Counts `MessageEnqueued` events once every earlier subscriber has handled them.

    Subscribed after the session opens, so `LiveAgent`'s own subscriber runs
    first: once `wait(n)` returns, the session has finished reacting to the
    first `n` announcements.
    """

    def __init__(self) -> None:
        self._seen = 0
        self._changed = asyncio.Condition()

    async def on_enqueued(self, event: MessageEnqueued) -> None:
        async with self._changed:
            self._seen += 1
            self._changed.notify_all()

    async def wait(self, count: int) -> None:
        async with self._changed:
            await asyncio.wait_for(self._changed.wait_for(lambda: self._seen >= count), timeout=3.0)
