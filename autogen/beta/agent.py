# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Awaitable, Callable, Iterable
from contextlib import AsyncExitStack, ExitStack
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, overload

from fast_depends import Provider

from .annotations import Context
from .config import LLMClient, ModelConfig
from .events import (
    BaseEvent,
    HumanInputRequest,
    ModelRequest,
    ModelResponse,
    ToolResults,
)
from .exceptions import ConfigNotProvidedError
from .history import History
from .hitl import HumanHook, default_hitl_hook, wrap_hitl
from .middleware.base import AgentTurn, BaseMiddleware, LLMCall, MiddlewareFactory
from .stream import MemoryStream, Stream
from .tools.executor import ToolExecutor
from .tools.final import FunctionParameters, FunctionTool, tool
from .tools.schemas import ToolSchema
from .tools.tool import Tool
from .utils import CONTEXT_OPTION_NAME, build_model

if TYPE_CHECKING:
    from .conversable import ConversableAdapter


class Askable(Protocol):
    async def ask(
        self,
        msg: str,
        *,
        dependencies: dict[Any, Any] | None = None,
        variables: dict[Any, Any] | None = None,
        prompt: Iterable[str] = (),
        config: ModelConfig | None = None,
        tools: Iterable[Tool] = (),
        middleware: Iterable["MiddlewareFactory"] = (),
    ) -> "AgentReply":
        raise NotImplementedError


class AgentReply(Askable):
    def __init__(
        self,
        response: ModelResponse,
        *,
        context: "Context",
        client: "LLMClient",
        agent: "Agent",
    ) -> None:
        self.response = response
        self.context = context
        self.__client = client
        self.__agent = agent

    @property
    def content(self) -> str | None:
        """Text content of the model's response for this turn."""
        return self.response.content

    @property
    def history(self) -> History:
        return self.context.stream.history

    async def ask(
        self,
        msg: str,
        *,
        dependencies: dict[Any, Any] | None = None,
        variables: dict[Any, Any] | None = None,
        prompt: Iterable[str] = (),
        config: ModelConfig | None = None,
        tools: Iterable[Tool] = (),
        middleware: Iterable["MiddlewareFactory"] = (),
    ) -> "AgentReply":
        initial_event = ModelRequest(content=msg)

        context = self.context
        if dependencies:
            context.dependencies.update(dependencies)
        if variables:
            context.variables.update(variables)
        if prompt:
            context.prompt = list(prompt)

        client = config.create() if config else self.__client

        return await self.__agent._execute(
            initial_event,
            context=context,
            client=client,
            additional_tools=tools,
            additional_middleware=middleware,
        )


PromptHook: TypeAlias = Callable[..., str] | Callable[..., Awaitable[str]]
PromptType: TypeAlias = str | PromptHook


class Agent(Askable):
    def __init__(
        self,
        name: str,
        prompt: PromptType | Iterable[PromptType] = (),
        *,
        config: ModelConfig | None = None,
        hitl_hook: HumanHook | None = None,
        tools: Iterable[Callable[..., Any] | Tool] = (),
        middleware: Iterable["MiddlewareFactory"] = (),
        dependencies: dict[Any, Any] | None = None,
        variables: dict[Any, Any] | None = None,
    ):
        self.name = name
        self.config = config

        self._agent_dependencies = dependencies or {}
        self._agent_variables = variables or {}

        self._middleware = middleware
        self.dependency_provider = Provider()
        self.tools = [FunctionTool.ensure_tool(t, provider=self.dependency_provider) for t in tools]

        self.__hitl_hook = wrap_hitl(hitl_hook) if hitl_hook else default_hitl_hook
        self.__tool_executor = ToolExecutor()

        self._system_prompt: list[str] = []
        self._dynamic_prompt: list[Callable[[ModelRequest, Context], Awaitable[str]]] = []

        if isinstance(prompt, str) or callable(prompt):
            prompt = [prompt]

        for p in prompt:
            if isinstance(p, str):
                self._system_prompt.append(p)
            else:
                self._dynamic_prompt.append(_wrap_prompt_hook(p))

    def hitl_hook(self, func: HumanHook) -> HumanHook:
        if self.__hitl_hook is not default_hitl_hook:
            warnings.warn(
                "You already set HITL hook, provided value overrides it",
                category=RuntimeWarning,
                stacklevel=2,
            )

        self.__hitl_hook = wrap_hitl(func)
        return func

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
            t = FunctionTool.ensure_tool(t, provider=self.dependency_provider)
            self.tools.append(t)
            return t

        if function:
            return make_tool(function)

        return make_tool

    async def ask(
        self,
        msg: str,
        *,
        stream: Stream | None = None,
        dependencies: dict[Any, Any] | None = None,
        variables: dict[Any, Any] | None = None,
        prompt: Iterable[str] = (),
        config: ModelConfig | None = None,
        tools: Iterable[Tool] = (),
        middleware: Iterable["MiddlewareFactory"] = (),
    ) -> "AgentReply":
        config = config or self.config
        if not config:
            raise ConfigNotProvidedError()
        client = config.create()

        stream = stream or MemoryStream()

        initial_event = ModelRequest(content=msg)

        context = Context(
            stream,
            prompt=list(prompt),
            dependencies=self._agent_dependencies | (dependencies or {}),
            variables=self._agent_variables | (variables or {}),
            dependency_provider=self.dependency_provider,
        )

        if not context.prompt:
            context.prompt.extend(self._system_prompt)

            for dp in self._dynamic_prompt:
                p = await dp(initial_event, context)
                context.prompt.append(p)

        return await self._execute(
            initial_event,
            context=context,
            client=client,
            additional_tools=tools,
            additional_middleware=middleware,
        )

    async def _execute(
        self,
        event: BaseEvent,
        *,
        context: Context,
        client: LLMClient,
        additional_tools: Iterable[Tool] = (),
        additional_middleware: Iterable["MiddlewareFactory"] = (),
    ) -> "AgentReply":
        all_tools: list[Tool] = list(
            chain(
                self.tools,
                additional_tools,
            )
        )

        all_schemas: list[ToolSchema] = []
        known_tools: set[str] = set()
        for t in all_tools:
            schemas = await t.schemas(context)
            all_schemas.extend(schemas)

            for schema in schemas:
                if schema.type == "function":
                    known_tools.add(schema.function.name)

        middleware_instances: list[BaseMiddleware] = []
        agent_turn: AgentTurn = _execute_turn
        llm_call: LLMCall = partial(client, tools=all_schemas)

        for m in reversed(
            list(
                chain(
                    self._middleware,
                    additional_middleware,
                )
            )
        ):
            mw = m(event, context)
            middleware_instances.append(mw)

            agent_turn = partial(mw.on_turn, agent_turn)
            llm_call = partial(mw.on_llm_call, llm_call)

        async def _call_client(context: Context) -> None:
            result = await llm_call(await context.stream.history.get_events(), context)
            await context.send(result)

        with ExitStack() as stack:
            stack.enter_context(
                context.stream.where(ModelRequest | ToolResults).sub_scope(_call_client),
            )

            stack.enter_context(
                context.stream.where(HumanInputRequest).sub_scope(self.__hitl_hook(middleware_instances)),
            )

            self.__tool_executor.register(
                stack,
                context,
                tools=all_tools,
                known_tools=known_tools,
                middleware=middleware_instances,
            )

            message = await agent_turn(event, context)

            return AgentReply(
                message,
                context=context,
                agent=self,
                client=client,
            )

    def as_conversable(self) -> "ConversableAdapter":
        from .conversable import ConversableAdapter

        return ConversableAdapter(self)


async def _execute_turn(event: BaseEvent, context: Context) -> ModelResponse:
    async with context.stream.get(ModelResponse) as result:
        await context.send(event)
        message = await result

    while message.tool_calls and not message.response_force:
        async with context.stream.get(ModelResponse) as result:
            await context.send(message.tool_calls)
            message = await result

    return message


def _wrap_prompt_hook(func: PromptHook) -> Callable[[ModelRequest, Context], Awaitable[str]]:
    call_model = build_model(func)

    async def wrapper(event: ModelRequest, context: Context) -> str:
        async with AsyncExitStack() as stack:
            return await call_model.asolve(
                event,
                stack=stack,
                cache_dependencies={},
                dependency_provider=context.dependency_provider,
                **{CONTEXT_OPTION_NAME: context},
            )

    return wrapper
