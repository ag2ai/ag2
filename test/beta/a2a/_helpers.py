# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor as A2AAgentExecutorBase
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import (
    PushNotificationConfigStore,
    TaskStore,
    TaskUpdater,
)
from a2a.types import Part, Task, TaskState, TaskStatus
from typing_extensions import Self

from autogen.beta import Agent, Context
from autogen.beta.a2a import A2AConfig, A2AServer
from autogen.beta.a2a.testing import make_test_client_factory
from autogen.beta.config.client import LLMClient
from autogen.beta.config.config import ModelConfig
from autogen.beta.events import (
    BaseEvent,
    ModelMessage,
    ModelResponse,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultEvent,
)
from autogen.beta.testing import TestConfig, TrackingConfig


@dataclass(slots=True)
class A2APair:
    server: A2AServer
    server_agent: Agent
    client: Agent
    tracking: TrackingConfig


@dataclass(slots=True)
class ExecutorPair:
    server: A2AServer
    executor: A2AAgentExecutorBase
    client: Agent


class StatelessScript(ModelConfig):
    def __init__(
        self,
        initial: ModelResponse | ToolCallEvent | Iterable[ToolCallEvent] | str,
        after_tool: ModelResponse | str | None = None,
    ) -> None:
        self.initial = initial
        self.after_tool = after_tool

    def copy(self) -> Self:
        return self

    def create(self) -> "StatelessScriptClient":
        return StatelessScriptClient(self.initial, self.after_tool)

    def create_files_client(self) -> None:
        raise NotImplementedError


class StatelessScriptClient(LLMClient):
    def __init__(
        self,
        initial: ModelResponse | ToolCallEvent | Iterable[ToolCallEvent] | str,
        after_tool: ModelResponse | str | None,
    ) -> None:
        self._initial = initial
        self._after_tool = after_tool

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: Context,
        **kwargs: Any,
    ) -> ModelResponse:
        last_meaningful = next(
            (m for m in reversed(messages) if isinstance(m, ToolResultEvent)),
            None,
        )
        chosen = self._after_tool if last_meaningful is not None else self._initial
        if chosen is None:
            chosen = ""
        return await _materialize(chosen, context)


async def _materialize(value: Any, context: Context) -> ModelResponse:
    if isinstance(value, ModelResponse):
        return value
    if isinstance(value, str):
        message = ModelMessage(value)
        await context.send(message)
        return ModelResponse(message=message)
    if isinstance(value, ToolCallEvent):
        return ModelResponse(tool_calls=ToolCallsEvent([value]))
    if isinstance(value, Iterable):
        return ModelResponse(tool_calls=ToolCallsEvent(list(value)))
    raise TypeError(f"Cannot materialize response of type {type(value).__name__}")


class PromptThenAckExecutor(A2AAgentExecutorBase):
    def __init__(self, prompt: str) -> None:
        self._prompt = prompt
        self.received_user_text: str | None = None

    async def execute(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        msg = request_context.message
        if msg is None:
            return
        task_id = msg.task_id or uuid4().hex
        context_id = msg.context_id or uuid4().hex
        updater = TaskUpdater(event_queue, task_id, context_id)

        if request_context.current_task is None:
            await event_queue.enqueue_event(
                Task(
                    id=task_id,
                    context_id=context_id,
                    status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                ),
            )
            await updater.start_work()
            await updater.requires_input(
                message=updater.new_agent_message(parts=[Part(text=self._prompt)]),
            )
            return

        text = "".join(p.text for p in msg.parts if p.text)
        self.received_user_text = text
        await updater.complete(
            message=updater.new_agent_message(parts=[Part(text=f"echo: {text}")]),
        )

    async def cancel(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        task = request_context.current_task
        if task is None:
            return
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.cancel()


def make_pair(
    initial: ModelResponse | ToolCallEvent | Iterable[ToolCallEvent] | str,
    after_tool: ModelResponse | str | None = None,
    *,
    server_tools: Iterable[Callable[..., object]] = (),
    client_tools: Iterable[Callable[..., object]] = (),
    server_url: str = "http://test",
    streaming: bool = True,
    task_store: TaskStore | None = None,
    push_config_store: PushNotificationConfigStore | None = None,
) -> A2APair:
    tracking = TrackingConfig(StatelessScript(initial, after_tool))
    server_agent = Agent("server-agent", config=tracking)
    for tool in server_tools:
        server_agent.tool(tool)

    server_kwargs: dict[str, Any] = {}
    if task_store is not None:
        server_kwargs["task_store"] = task_store
    if push_config_store is not None:
        server_kwargs["push_config_store"] = push_config_store

    server = A2AServer(server_agent, **server_kwargs)
    factory = make_test_client_factory(server, url=server_url)

    client_config = A2AConfig(
        url=server_url,
        httpx_client_factory=factory,
        streaming=streaming,
    )
    client_agent = Agent("client-agent", config=client_config)
    for tool in client_tools:
        client_agent.tool(tool)

    return A2APair(server=server, server_agent=server_agent, client=client_agent, tracking=tracking)


def make_executor_pair(
    executor: A2AAgentExecutorBase,
    *,
    server_url: str = "http://test",
    streaming: bool = False,
    task_store: TaskStore | None = None,
    push_config_store: PushNotificationConfigStore | None = None,
    hitl_hook: Callable[..., Any] | None = None,
) -> ExecutorPair:
    server_agent = Agent("server-stub", config=TestConfig("unused"))

    server_kwargs: dict[str, Any] = {}
    if task_store is not None:
        server_kwargs["task_store"] = task_store
    if push_config_store is not None:
        server_kwargs["push_config_store"] = push_config_store

    server = A2AServer(server_agent, **server_kwargs)
    server._executor = executor  # type: ignore[attr-defined]
    factory = make_test_client_factory(server, url=server_url)

    client_kwargs: dict[str, Any] = {}
    if hitl_hook is not None:
        client_kwargs["hitl_hook"] = hitl_hook

    client = Agent(
        "client",
        config=A2AConfig(url=server_url, httpx_client_factory=factory, streaming=streaming),
        **client_kwargs,
    )
    return ExecutorPair(server=server, executor=executor, client=client)
