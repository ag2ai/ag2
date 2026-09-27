# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterable
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

from ag2 import Agent
from ag2.a2a import A2AConfig, A2AServer, build_card
from ag2.a2a.testing import (
    make_test_client_factory,
    make_test_rest_client_factory,
    pick_free_port,
)
from ag2.testing import TestConfig, TrackingConfig, Turn
from test._helpers import LLMCalls


def a2a_config(agent: Agent) -> A2AConfig:
    """The agent's ``A2AConfig``.

    ``Agent.config`` is declared ``ModelConfig | None``, but the task and
    push-notification admin helpers take an ``A2AConfig`` — narrow once here
    instead of handing them an unchecked union at every call site.
    """
    config = agent.config
    assert isinstance(config, A2AConfig), f"{agent.name} is not backed by an A2AConfig"
    return config


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
    recording: LLMCalls


@dataclass(slots=True)
class GrpcPair:
    server: A2AServer
    server_agent: Agent
    client: Agent
    tracking: TrackingConfig
    grpc_url: str
    grpc_server: grpc.aio.Server


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
    *turns: Turn,
    server_tools: Iterable[Callable[..., object]] = (),
    client_tools: Iterable[Callable[..., object]] = (),
    server_url: str = "http://test",
    streaming: bool = True,
    task_store: TaskStore | None = None,
    push_config_store: PushNotificationConfigStore | None = None,
) -> A2APair:
    tracking = TrackingConfig(TestConfig(*turns, shared_script=True))
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


def make_recording_pair(
    response: str,
    *,
    server_url: str = "http://test",
    streaming: bool = False,
) -> RecordingPair:
    recording = LLMCalls()
    server_agent = Agent("server-agent", config=TestConfig(response), middleware=[recording.middleware()])
    server = A2AServer(server_agent)
    factory = make_test_client_factory(server, url=server_url)

    client = Agent(
        "client-agent",
        config=A2AConfig(card_url=server_url, httpx_client_factory=factory, streaming=streaming),
    )
    return RecordingPair(server=server, server_agent=server_agent, client=client, recording=recording)


def make_rest_pair(
    *turns: Turn,
    server_url: str = "http://test",
    streaming: bool = False,
) -> A2APair:
    tracking = TrackingConfig(TestConfig(*turns, shared_script=True))
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
    *turns: Turn,
    host: str = "127.0.0.1",
    streaming: bool = False,
) -> GrpcPair:
    tracking = TrackingConfig(TestConfig(*turns, shared_script=True))
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
