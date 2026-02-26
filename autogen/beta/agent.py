from collections.abc import Callable, Iterable
from contextlib import ExitStack
from typing import Any, Protocol, TypeAlias, overload, runtime_checkable

from fast_depends import Provider

from .events import BaseEvent, ModelRequest, ModelResponse, ToolCall, ToolResult
from .llms import LLMClient
from .stream import Context, History, Stream
from .tools import FunctionParameters, Tool, ToolsExecutor, tool


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
        ctx: Context,
        agent: "Agent",
    ) -> None:
        self.message = message

        self.ctx = ctx
        self.__agent = agent

    async def ask(
        self,
        msg: str,
    ) -> "Conversation":
        initial_event = ModelRequest(prompt=msg)
        return await self.__agent._execute(
            initial_event,
            ctx=self.ctx,
        )

    @property
    def history(self) -> History:
        return self.ctx.stream.history


@runtime_checkable
class PromptHook(Protocol):
    async def __call__(self, event: BaseEvent, ctx: Context) -> str: ...


PromptType: TypeAlias = str | PromptHook


class Agent(Askable):
    def __init__(
        self,
        name: str,
        prompt: PromptType | Iterable[PromptType] = (),
        *,
        client: LLMClient,
        tools: Iterable[Callable[..., Any] | Tool] = (),
    ):
        self.name = name
        self.client = client

        self.dependency_provider = Provider()
        self.__tools_executor = ToolsExecutor(Tool.ensure_tool(t, provider=self.dependency_provider) for t in tools)

        self._system_prompt: list[str] = []
        self._dynamic_prompt: list[PromptHook] = []

        if isinstance(prompt, (str, PromptHook)):
            prompt = [prompt]

        for p in prompt:
            if isinstance(p, str):
                self._system_prompt.append(p)
            else:
                self._dynamic_prompt.append(p)

    @property
    def tools(self) -> tuple[Tool, ...]:
        return tuple(self.__tools_executor.tools.values())

    def prompt(self, func: PromptHook) -> PromptHook:
        self._dynamic_prompt.append(func)
        return func

    @overload
    def tool(
        self,
        function: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        schema: FunctionParameters | None = None,
        sync_to_thread: bool = True,
    ) -> Tool: ...

    @overload
    def tool(
        self,
        function: None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        schema: FunctionParameters | None = None,
        sync_to_thread: bool = True,
    ) -> Callable[[Callable[..., Any]], Tool]: ...

    def tool(
        self,
        function: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        schema: FunctionParameters | None = None,
        sync_to_thread: bool = True,
    ) -> Tool | Callable[[Callable[..., Any]], Tool]:
        def make_tool(f: Callable[..., Any]) -> Tool:
            t = tool(
                f,
                name=name,
                description=description,
                schema=schema,
                sync_to_thread=sync_to_thread,
            )
            t = Tool.ensure_tool(t, provider=self.dependency_provider)
            self.__tools_executor.add_tool(t)
            return t

        if function:
            return make_tool(function)

        return make_tool

    async def ask(
        self,
        msg: str,
        *,
        stream: Stream | None = None,
        prompt: Iterable[str] = (),
    ) -> "Conversation":
        stream = stream or Stream()

        initial_event = ModelRequest(prompt=msg)

        ctx = Context(stream, prompt=list(prompt))

        if not ctx.prompt:
            ctx.prompt.extend(self._system_prompt)

            for dp in self._dynamic_prompt:
                p = await dp(initial_event, ctx)
                ctx.prompt.append(p)

        return await self._execute(
            initial_event,
            ctx=ctx,
        )

    async def _execute(self, event: BaseEvent, *, ctx: Context) -> "Conversation":
        with ExitStack() as stack:
            stack.enter_context(
                ctx.stream.where(ModelRequest).sub_scope(self._call_client),
            )
            stack.enter_context(
                ctx.stream.where(ToolResult).sub_scope(self._call_client),
            )

            stack.enter_context(
                ctx.stream.where(ToolCall).sub_scope(self.__tools_executor.execute_tool),
            )

            async with ctx.stream.get(ModelResponse) as result:
                await ctx.send(event)

                return Conversation(
                    await result,
                    ctx=ctx,
                    agent=self,
                )

    async def _call_client(self, event: BaseEvent, ctx: Context) -> None:
        await self.client(
            *await ctx.stream.history.get_events(),
            ctx=ctx,
        )
