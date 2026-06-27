# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import grpc.aio
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

from ag2 import Agent, Context
from ag2.a2a import A2AConfig, A2AServer, build_card
from ag2.a2a.testing import (
    make_test_client_factory,
    make_test_rest_client_factory,
    pick_free_port,
)
from ag2.config.client import LLMClient
from ag2.config.config import ModelConfig
from ag2.events import (
    BaseEvent,
    ModelMessage,
    ModelResponse,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultEvent,
)
from ag2.testing import TestConfig, TrackingConfig


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


@dataclass(slots=True)
class RecordingPair:
    server: A2AServer
    server_agent: Agent
    client: Agent
    recording: "RecordingConfig"


@dataclass(slots=True)
class GrpcPair:
    server: A2AServer
    server_agent: Agent
    client: Agent
    tracking: TrackingConfig
    grpc_url: str
    grpc_server: grpc.aio.Server


# Stateless mock: A2A executor recreates the LLM client on every turn (per
# its stateless-flavor design), so an iter-based TestConfig would replay
# the first event forever. This mock decides what to return based on the
# message history instead.
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
        card_url=server_url,
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

    server = A2AServer(server_agent, executor=executor, **server_kwargs)
    factory = make_test_client_factory(server, url=server_url)

    client_kwargs: dict[str, Any] = {}
    if hitl_hook is not None:
        client_kwargs["hitl_hook"] = hitl_hook

    client = Agent(
        "client",
        config=A2AConfig(card_url=server_url, httpx_client_factory=factory, streaming=streaming),
        **client_kwargs,
    )
    return ExecutorPair(server=server, executor=executor, client=client)


# Captures the full ``messages`` list per LLM call. ``TrackingConfig`` only
# records ``messages[-1]``, which is enough for last-input assertions but not
# for verifying that the full prior-turn history was passed across the wire
# on a follow-up turn.
class RecordingConfig(ModelConfig):
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[list[BaseEvent]] = []

    def copy(self) -> Self:
        return self

    def create(self) -> "RecordingClient":
        return RecordingClient(self.response, self.calls)

    def create_files_client(self) -> None:
        raise NotImplementedError


class RecordingClient(LLMClient):
    def __init__(self, response: str, calls: list[list[BaseEvent]]) -> None:
        self._response = response
        self._calls = calls

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: Context,
        **kwargs: Any,
    ) -> ModelResponse:
        self._calls.append(list(messages))
        msg = ModelMessage(self._response)
        await context.send(msg)
        return ModelResponse(message=msg)


def make_recording_pair(
    response: str,
    *,
    server_url: str = "http://test",
    streaming: bool = False,
) -> RecordingPair:
    recording = RecordingConfig(response=response)
    server_agent = Agent("server-agent", config=recording)
    server = A2AServer(server_agent)
    factory = make_test_client_factory(server, url=server_url)

    client = Agent(
        "client-agent",
        config=A2AConfig(card_url=server_url, httpx_client_factory=factory, streaming=streaming),
    )
    return RecordingPair(server=server, server_agent=server_agent, client=client, recording=recording)


def make_rest_pair(
    initial: ModelResponse | ToolCallEvent | Iterable[ToolCallEvent] | str,
    after_tool: ModelResponse | str | None = None,
    *,
    server_url: str = "http://test",
    streaming: bool = False,
) -> A2APair:
    tracking = TrackingConfig(StatelessScript(initial, after_tool))
    server_agent = Agent("server-agent", config=tracking)
    server = A2AServer(server_agent)
    factory = make_test_rest_client_factory(server, url=server_url)

    client = Agent(
        "client-agent",
        config=A2AConfig(
            card_url=server_url,
            httpx_client_factory=factory,
            prefer="rest",
            streaming=streaming,
        ),
    )
    return A2APair(server=server, server_agent=server_agent, client=client, tracking=tracking)


async def start_grpc_pair(
    initial: ModelResponse | ToolCallEvent | Iterable[ToolCallEvent] | str,
    after_tool: ModelResponse | str | None = None,
    *,
    host: str = "127.0.0.1",
    streaming: bool = False,
) -> GrpcPair:
    tracking = TrackingConfig(StatelessScript(initial, after_tool))
    server_agent = Agent("server-agent", config=tracking)
    server = A2AServer(server_agent)

    grpc_url = f"{host}:{pick_free_port(host)}"
    card = build_card(server_agent, url=grpc_url, transports=("grpc",), grpc_url=grpc_url)
    grpc_server = server.build_grpc(bind=grpc_url, grpc_url=grpc_url, card=card)
    await grpc_server.start()

    client = Agent(
        "client-agent",
        config=A2AConfig(
            card_url=grpc_url,
            preset_card=card,
            prefer="grpc",
            streaming=streaming,
        ),
    )
    return GrpcPair(
        server=server,
        server_agent=server_agent,
        client=client,
        tracking=tracking,
        grpc_url=grpc_url,
        grpc_server=grpc_server,
    )
