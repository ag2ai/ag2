# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterable
from contextlib import ExitStack
from typing import Any, Protocol, TypeAlias, overload, runtime_checkable

from fast_depends import Provider

from .config import LLMClient, ModelConfig
from .events import BaseEvent, ModelRequest, ModelResponse, ToolCall, ToolCalls, ToolResults
from .stream import Context, History, MemoryStream, Stream
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
        ctx: "Context",
        client: "LLMClient",
        agent: "Agent",
    ) -> None:
        self.message = message

        self.ctx = ctx
        self.__client = client
        self.__agent = agent

    async def ask(
        self,
        msg: str,
    ) -> "Conversation":
        initial_event = ModelRequest(content=msg)
        return await self.__agent._execute(
            initial_event,
            ctx=self.ctx,
            client=self.__client,
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
        config: ModelConfig,
        tools: Iterable[Callable[..., Any] | Tool] = (),
    ):
        self.name = name
        self.config = config

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

    @overload
    def prompt(
        self,
        func: None = None,
    ) -> Callable[[PromptHook], PromptHook]: ...

    @overload
    def prompt(
        self,
        func: PromptHook,
    ) -> PromptHook: ...

    def prompt(
        self,
        func: PromptHook | None = None,
    ) -> PromptHook | Callable[[PromptHook], PromptHook]:
        def wrapper(f: PromptHook) -> PromptHook:
            self._dynamic_prompt.append(f)
            return f

        if func:
            return wrapper(func)
        return wrapper

    @overload
    def tool(
        self,
        function: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        schema: FunctionParameters | None = None,
        strict: bool | None = True,
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
        strict: bool | None = True,
        sync_to_thread: bool = True,
    ) -> Callable[[Callable[..., Any]], Tool]: ...

    def tool(
        self,
        function: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        schema: FunctionParameters | None = None,
        strict: bool | None = True,
        sync_to_thread: bool = True,
    ) -> Tool | Callable[[Callable[..., Any]], Tool]:
        def make_tool(f: Callable[..., Any]) -> Tool:
            t = tool(
                f,
                name=name,
                description=description,
                schema=schema,
                strict=strict,
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
        stream = stream or MemoryStream()

        initial_event = ModelRequest(content=msg)

        ctx = Context(stream, prompt=list(prompt))

        if not ctx.prompt:
            ctx.prompt.extend(self._system_prompt)

            for dp in self._dynamic_prompt:
                p = await dp(initial_event, ctx)
                ctx.prompt.append(p)

        client = self.config.create()

        return await self._execute(
            initial_event,
            ctx=ctx,
            client=client,
        )

    async def _execute(
        self,
        event: BaseEvent,
        *,
        ctx: Context,
        client: LLMClient,
    ) -> "Conversation":
        async def _call_client(event: BaseEvent, ctx: Context) -> None:
            await client(
                *await ctx.stream.history.get_events(),
                ctx=ctx,
                tools=self.tools,
            )

        with ExitStack() as stack:
            stack.enter_context(
                ctx.stream.where(ModelRequest | ToolResults).sub_scope(_call_client),
            )

            stack.enter_context(
                ctx.stream.where(ToolCall).sub_scope(self.__tools_executor.execute_tool),
            )

            stack.enter_context(
                ctx.stream.where(ToolCalls).sub_scope(self.__tools_executor.execute_tools),
            )

            async with ctx.stream.get(ModelResponse) as result:
                await ctx.send(event)
                message = await result

            while message.tool_calls:
                async with ctx.stream.get(ModelResponse) as result:
                    await ctx.send(message.tool_calls)
                    message = await result

            return Conversation(
                message,
                ctx=ctx,
                agent=self,
                client=client,
            )
