# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from ag2.annotations import Context
from ag2.stream import MemoryStream

from .subagent_tool import StreamFactory

if TYPE_CHECKING:
    from ag2.agent import Agent


def persistent_stream() -> StreamFactory:
    def stream_factory(agent: "Agent", ctx: "Context") -> MemoryStream:
        # Cache the stream instance, not just its id: the turn lock is keyed
        # by object identity, so a fresh object per call would silently get
        # its own independent lock.
        key = f"ag:{agent.name}:stream"
        if (stream := ctx.dependencies.get(key)) is None:
            stream = ctx.dependencies[key] = MemoryStream(storage=ctx.stream.history.storage)

        return stream

    return stream_factory
