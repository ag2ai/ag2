# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias, runtime_checkable
from uuid import UUID

from fast_depends import Provider

from .events import BaseEvent, HumanInputRequest, HumanMessage

StreamId: TypeAlias = UUID


@runtime_checkable
class WritableStream(Protocol):
    id: StreamId

    async def send(self, event: BaseEvent, ctx: "Context") -> None: ...


@dataclass(slots=True)
class Context:
    stream: WritableStream
    dependency_provider: "Provider | None" = None

    prompt: list[str] = field(default_factory=list)

    dependencies: dict[Any, Any] = field(default_factory=dict)
    # store Context Variables as separated serializable field
    variables: dict[str, Any] = field(default_factory=dict)

    async def input(self, message: str, timeout: float | None = None) -> str:
        async with self.stream.get(HumanMessage) as response:
            await self.send(HumanInputRequest(content=message))
            return (await asyncio.wait_for(response, timeout)).content

    async def send(self, event: BaseEvent) -> None:
        await self.stream.send(event, self)
