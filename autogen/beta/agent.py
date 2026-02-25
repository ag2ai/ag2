from collections.abc import Iterable
from contextlib import ExitStack
from typing import Protocol

from .events import BaseEvent, ModelRequest, ModelResponse, ToolCall, ToolResult
from .llms import LLMClient
from .stream import Context, History, Stream, Subscriber


class Askable(Protocol):
    async def ask(
        self,
        msg: str,
    ) -> "Conversation":
        raise NotImplementedError


class Conversation(Askable):
    def __init__(
        self,
        message: ModelResponse,
        *,
        stream: Stream,
        agent: "Agent",
    ) -> None:
        self.message = message
        self.stream = stream
        self.__agent = agent

    async def ask(
        self,
        msg: str,
    ) -> "Conversation":
        return await self.__agent.ask(
            msg,
            stream=self.stream,
        )

    @property
    def history(self) -> History:
        return self.stream.history


class Agent(Askable):
    def __init__(
        self,
        client: LLMClient,
        *,
        tools: Iterable[Subscriber] = (),
    ):
        self.client = client
        self.tools = tools

    async def ask(
        self,
        msg: str,
        *,
        stream: Stream | None = None,
    ) -> "Conversation":
        stream = stream or Stream()

        return await self._execute(
            ModelRequest(prompt=msg, context=""),
            stream=stream,
        )

    async def restore(
        self,
        *,
        stream: Stream,
    ) -> "Conversation":
        stream = stream or Stream()

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
    ) -> "Conversation":
        with ExitStack() as stack:
            stack.enter_context(stream.where(ModelRequest).sub_scope(self._call_client))
            stack.enter_context(
                stream.where(ToolResult).sub_scope(self._call_client),
            )

            for tool in self.tools:
                stack.enter_context(stream.where(ToolCall.name == tool.__name__).sub_scope(tool))

            async with stream.get(ModelResponse) as result:
                await stream.send(event)

                return Conversation(
                    await result,
                    stream=stream,
                    agent=self,
                )

    async def _call_client(self, event: BaseEvent, ctx: Context) -> None:
        await self.client(
            *await ctx.stream.history.get_events(),
            stream=ctx.stream,
        )
