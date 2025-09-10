# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import warnings
from collections.abc import Awaitable, Callable, Iterable, MutableMapping
from itertools import chain
from typing import Any, Literal, Protocol, cast
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Response, status

from autogen.agentchat import ConversableAgent
from autogen.agentchat.conversable_agent import normilize_message_to_oai
from autogen.agentchat.group.context_variables import ContextVariables
from autogen.agentchat.group.group_tool_executor import GroupToolExecutor
from autogen.agentchat.group.reply_result import ReplyResult
from autogen.agentchat.group.targets.transition_target import TransitionTarget
from autogen.doc_utils import export_module

from .protocol import AgentBusMessage, get_tool_names


class RemoteService(Protocol):
    """Interface to make AgentBus compatible with non AG2 systems."""

    name: str

    async def __call__(self, state: AgentBusMessage) -> AgentBusMessage | None:
        """Executable that consumes Conversation State and returns a new state."""
        ...


@export_module("autogen.remote")
class HTTPAgentBus:
    def __init__(
        self,
        agents: Iterable[ConversableAgent] = (),
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
        out_message: dict[str, Any] | None
        if guardrail_result := self.agent.run_input_guardrails(state.messages):
            # input guardrail activated by initial messages
            _, out_message = normilize_message_to_oai(guardrail_result.reply, self.agent.name, role="assistant")
            return AgentBusMessage(messages=[out_message], context=state.context)

        context_variables = ContextVariables(state.context)
        tool_executor = self._make_tool_executor(context_variables)

        local_history: list[dict[str, Any]] = []
        while True:
            messages = state.messages + local_history

            # TODO: catch ask user input event
            is_final, _ = await self.agent.a_check_termination_and_human_reply(messages)
            if is_final:
                break

            reply = await self.agent.a_generate_reply(
                messages,
                exclude=(
                    ConversableAgent.check_termination_and_human_reply,
                    ConversableAgent.a_check_termination_and_human_reply,
                    ConversableAgent.generate_oai_reply,
                    ConversableAgent.a_generate_oai_reply,
                ),
            )

            if not reply:
                _, reply = await self.agent.a_generate_oai_reply(
                    messages,
                    tools=state.client_tools,
                )

            should_continue, out_message = self._add_message_to_local_history(reply, role="assistant")
            if out_message:
                local_history.append(out_message)
            if not should_continue:
                break
            out_message = cast(dict[str, Any], out_message)

            called_tools = get_tool_names(out_message.get("tool_calls", []))
            if state.client_tool_names.intersection(called_tools):
                break  # return client tool execution command back to client

            tool_result, updated_context_variables = self._try_execute_local_tool(tool_executor, out_message)

            if updated_context_variables:
                context_variables.update(updated_context_variables.to_dict())

            should_continue, out_message = self._add_message_to_local_history(tool_result, role="tool")
            if out_message:
                local_history.append(out_message)
            if not should_continue:
                break

        if not local_history:
            return None

        return AgentBusMessage(messages=local_history, context=context_variables.data)

    def _add_message_to_local_history(
        self, message: str | dict[str, Any] | None, role: str
    ) -> tuple[Literal[True], dict[str, Any]] | tuple[Literal[False], dict[str, Any] | None]:
        if message is None:
            return False, None  # output message is empty, interrupt the loop

        if guardrail_result := self.agent.run_output_guardrails(message):
            _, out_message = normilize_message_to_oai(guardrail_result.reply, self.agent.name, role=role)
            return False, out_message  # output guardrail activated, interrupt the loop

        valid, out_message = normilize_message_to_oai(message, self.agent.name, role=role)
        if not valid:
            return False, None  # tool result is not valid OAI message, interrupt the loop

        return True, out_message

    def _make_tool_executor(self, context_variables: ContextVariables) -> GroupToolExecutor:
        tool_executor = GroupToolExecutor()
        for tool in self.agent.tools:
            # TODO: inject ChatContext to tool
            new_tool = tool_executor.make_tool_copy_with_context_variables(tool, context_variables) or tool
            tool_executor.register_for_execution(serialize=False, silent_override=True)(new_tool)
        return tool_executor

    def _try_execute_local_tool(
        self,
        tool_executor: GroupToolExecutor,
        tool_message: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, ContextVariables | None]:
        tool_result: dict[str, Any] | None = None
        updated_context_variables: ContextVariables | None = None

        if "tool_calls" in tool_message:
            _, tool_result = tool_executor.generate_tool_calls_reply([tool_message])
            if tool_result is None:
                return tool_result, updated_context_variables

            if "tool_responses" in tool_result:
                # TODO: catch handoffs
                for tool_response in tool_result["tool_responses"]:
                    content = tool_response["content"]

                    if isinstance(content, TransitionTarget):
                        warnings.warn(
                            f"Tool {self.agent.name} returned a target, which is not supported in remote mode"
                        )

                    elif isinstance(content, ReplyResult):
                        if content.target:
                            warnings.warn(
                                f"Tool {self.agent.name} returned a target, which is not supported in remote mode"
                            )

                        if content.context_variables:
                            updated_context_variables = content.context_variables
                            tool_response["content"] = content.message

        return tool_result, updated_context_variables
