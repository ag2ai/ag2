from collections.abc import Iterable
from contextlib import ExitStack

from .events import BaseEvent, ModelRequest, ModelResponse, ToolCall, ToolResult
from .llms import LLMClient
from .stream import Context, Stream, Subscriber


class Agent:
    def __init__(
        self,
        client: LLMClient,
        *,
        stream: Stream | None = None,
        tools: Iterable[Subscriber] = (),
    ):
        self.stream = stream
        self.client = client
        self.tools = tools

    async def ask(
        self,
        msg: str,
        *,
        stream: Stream | None = None,
    ) -> ModelResponse:
        stream = stream or self.stream
        if not stream:
            raise ValueError("Stream required")

        return await self._execute(
            ModelRequest(prompt=msg, context=""),
            stream=stream,
        )

    async def restore(
        self,
        *,
        stream: Stream | None = None,
    ) -> ModelResponse:
        stream = stream or self.stream
        if not stream:
            raise ValueError("Stream required")

        events = list(await stream.history.get_events())

        last_msg = events[-1]
        if isinstance(last_msg, ModelResponse):
            return last_msg

        return await self._execute(
            last_msg,
            stream=stream,
        )

    async def _execute(
        self,
        event: BaseEvent,
        *,
        stream: Stream,
    ) -> ModelResponse:
        with ExitStack() as stack:
            stack.enter_context(stream.where(ModelRequest).sub_scope(self._call_client))
            stack.enter_context(stream.where(ToolResult).sub_scope(self._call_client))

            for tool in self.tools:
                stack.enter_context(stream.where(ToolCall.name == tool.__name__).sub_scope(tool))

            async with stream.get(ModelResponse) as result:
                await stream.send(event)
                return await result

    async def _call_client(self, event: BaseEvent, ctx: Context) -> None:
        await self.client(
            *await ctx.stream.history.get_events(),
            stream=ctx.stream,
        )
