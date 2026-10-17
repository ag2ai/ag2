# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from anthropic.types import Message, Usage


class FakeStream:
    """Stand-in for Anthropic's streaming context manager.

    Yields the given raw stream events, then answers ``get_final_message`` with an
    empty assistant message — the client reads the blocks off the events, not off
    the final message.
    """

    def __init__(self, events: list[Any], *, stop_reason: str = "end_turn") -> None:
        self._events = events
        self._stop_reason = stop_reason

    def __aiter__(self) -> Any:
        async def gen() -> Any:
            for e in self._events:
                yield e

        return gen()

    async def get_final_message(self) -> Message:
        return Message.model_construct(
            id="m1",
            type="message",
            role="assistant",
            model="claude-sonnet-5",
            content=[],
            usage=Usage(input_tokens=1, output_tokens=1),
            stop_reason=self._stop_reason,
        )
