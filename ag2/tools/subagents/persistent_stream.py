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
    """Return a StreamFactory that reuses one stream per agent and context.

    The first delegation caches the MemoryStream itself (not just its id)
    in ``context.dependencies``, so every later delegation in the same context
    reuses the same object. Same-object reuse matters beyond shared history:
    Agent._execute serializes concurrent turns through a per-stream lock
    attached to the stream instance, which only works if delegations into the
    same persistent sub-agent actually share one instance.
    """

    def stream_factory(agent: "Agent", ctx: "Context") -> MemoryStream:
        key = f"ag:{agent.name}:stream"
        stream = ctx.dependencies.get(key)
        if stream is None:
            # Concurrent first delegations may race to create; setdefault makes
            # the first object stored the one both callers get.
            stream = ctx.dependencies.setdefault(
                key,
                MemoryStream(
                    storage=ctx.stream.history.storage,
                    id=uuid4(),
                ),
            )
        return stream

    return stream_factory
