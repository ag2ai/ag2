# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING
from uuid import uuid4

from autogen.beta.annotations import Context
from autogen.beta.stream import MemoryStream

from .subagent_tool import StreamFactory

if TYPE_CHECKING:
    from autogen.beta.agent import Agent


def persistent_stream() -> StreamFactory:
    def stream_factory(agent: "Agent", ctx: "Context") -> MemoryStream:
        key = f"ag:{agent.name}:stream"
        if not (stream_id := ctx.dependencies.get(key)):
            stream_id = ctx.dependencies[key] = uuid4()

        return MemoryStream(
            storage=ctx.stream.history.storage,
            id=stream_id,
        )

    return stream_factory
