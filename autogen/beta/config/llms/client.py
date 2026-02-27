from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from autogen.beta.events import BaseEvent
from autogen.beta.stream import Context
from autogen.beta.tools import Tool


@runtime_checkable
class LLMClient(Protocol):
    async def __call__(
        self,
        *messages: BaseEvent,
        ctx: Context,
        tools: Iterable[Tool],
    ) -> None: ...
