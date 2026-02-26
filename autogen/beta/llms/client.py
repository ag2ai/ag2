from typing import Protocol

from autogen.beta.events import BaseEvent
from autogen.beta.stream import Context


class LLMClient(Protocol):
    async def __call__(
        self,
        *messages: BaseEvent,
        ctx: Context,
    ) -> None: ...
