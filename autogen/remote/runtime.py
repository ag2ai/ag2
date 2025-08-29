# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Awaitable, Callable, Iterable, MutableMapping
from itertools import chain
from typing import Any, Protocol
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Response, status

from autogen.agentchat import ConversableAgent
from autogen.agentchat.conversable_agent import normilize_message_to_oai
from autogen.agentchat.group.context_variables import ContextVariables
from autogen.agentchat.group.group_tool_executor import GroupToolExecutor
from autogen.tools.tool import Tool

from .protocol import AgentBusMessage


class RemoteService(Protocol):
    """Interface to make AgentBus compatible with non AG2 systems."""

    name: str

    async def __call__(self, state: AgentBusMessage) -> AgentBusMessage | None:
        """Executable that consumes Conversation State and returns a new state."""
        ...


class HTTPAgentBus:
    def __init__(
        self,
        agents: Iterable[ConversableAgent],
        *,
        long_polling_interval: float = 10.0,
        additional_services: Iterable[RemoteService] = (),
    ) -> None:
        """Create HTTPAgentBus runtime.

        The runtime make passed agents able to process remote calls.

        Args:
            agents: agents to register as remote services
            long_polling_interval:
                timeout to response on task status calls for long living executions.
                Should be less then clients' HTTP request timeout.
            additional_services:
                additional services to register as remote services
        """
        self.app = FastAPI()

        for service in chain(map(AgentService, agents), additional_services):
            register_agent_endpoints(
                app=self.app,
                service=service,
                long_polling_interval=long_polling_interval,
            )

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Callable[[], Awaitable[MutableMapping[str, Any]]],
        send: Callable[[MutableMapping[str, Any]], Awaitable[None]],
    ) -> None:
        """ASGI interface."""
        await self.app(scope, receive, send)


def register_agent_endpoints(
    app: FastAPI,
    service: RemoteService,
    long_polling_interval: float,
) -> None:
    tasks: dict[UUID, asyncio.Task[AgentBusMessage | None]] = {}

    @app.get(f"/{service.name}" + "/{task_id}", response_model=AgentBusMessage | None)
    async def remote_call_result(task_id: UUID) -> Response | AgentBusMessage | None:
        if task_id not in tasks:
            raise HTTPException(
                detail=f"`{task_id}` task not found",
                status_code=status.HTTP_404_NOT_FOUND,
            )

        task = tasks[task_id]

        await asyncio.wait(
            (task, asyncio.create_task(asyncio.sleep(long_polling_interval))),
            return_when=asyncio.FIRST_COMPLETED,
        )

        if not task.done():
            return Response(status_code=status.HTTP_425_TOO_EARLY)

        try:
            reply = task.result()  # Task inner errors raising here
        finally:
            # TODO: how to clear hanged tasks?
            tasks.pop(task_id, None)

        if reply is None:
            return Response(status_code=status.HTTP_204_NO_CONTENT)

        return reply

    @app.post(f"/{service.name}", status_code=status.HTTP_202_ACCEPTED)
    async def remote_call_starter(state: AgentBusMessage) -> UUID:
        task, task_id = asyncio.create_task(service(state)), uuid4()
        tasks[task_id] = task
        return task_id


class AgentService(RemoteService):
    def __init__(self, agent: ConversableAgent) -> None:
        self.name = agent.name
        self.agent = agent

    async def __call__(self, state: AgentBusMessage) -> AgentBusMessage | None:
        # TODO: support input guardrails
        context_variables = ContextVariables(state.context_variables)

        tool_executor = GroupToolExecutor()
        for tool in self.agent.tools:
            # TODO: inject ChatContext to tool
            new_tool = tool_executor._make_tool_copy_with_context_variables(Tool(func_or_tool=tool), context_variables)
            tool_executor.register_for_execution(serialize=False, silent_override=True)(new_tool)

        local_history: list[dict[str, Any]] = []

        while True:
            chat_history = state.messages + local_history

            reply = await self.agent.a_generate_reply(chat_history, None)

            if reply is None:
                break  # last reply empty

            is_valid, out_message = normilize_message_to_oai(reply, self.agent.name, role="assistant")

            if is_valid:
                local_history.append(out_message)

                if "tool_calls" in out_message:
                    _, tool_message = tool_executor.generate_tool_calls_reply([out_message])
                    is_valid, out_message = normilize_message_to_oai(tool_message, self.agent.name, role="tool")

                    if is_valid:
                        local_history.append(out_message)

                    else:
                        break

                    # TODO: catch update ContextVariables events
                    # TODO: catch handoffs
                    continue  # process response by agent itself

            # last reply broken or
            # final reply from LLM
            break

        if not local_history:
            return None

        # TODO: support output guardrails
        return AgentBusMessage(messages=local_history)
