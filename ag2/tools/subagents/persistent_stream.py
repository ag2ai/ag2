# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING
from uuid import uuid4

from ag2.annotations import Context
from ag2.stream import MemoryStream

from .subagent_tool import StreamFactory

if TYPE_CHECKING:
    from ag2.agent import Agent


def persistent_stream() -> StreamFactory:
    def stream_factory(agent: "Agent", ctx: "Context") -> MemoryStream:
        key = f"ag:{agent.name}:stream"
        # Cache the stream object itself (not just its id) so every call for
        # the same sub-agent/context returns the same instance. The turn lock
        # Agent._execute attaches via `_get_stream_turn_lock` is keyed by
        # object identity, so a per-call rebuild would give concurrent
        # delegations independent locks and the lock would never serialize
        # overlapping turns on a shared persistent stream.
        if not (stream := ctx.dependencies.get(key)):
            stream = MemoryStream(
                storage=ctx.stream.history.storage,
                id=uuid4(),
            )
            ctx.dependencies[key] = stream

        return stream

    return stream_factory
