# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Any, Literal, cast

from autogen.agentchat import ConversableAgent
from autogen.agentchat.conversable_agent import normilize_message_to_oai
from autogen.agentchat.group.context_variables import ContextVariables
from autogen.agentchat.group.group_tool_executor import GroupToolExecutor
from autogen.agentchat.group.reply_result import ReplyResult
from autogen.agentchat.group.targets.transition_target import AskUserTarget, TransitionTarget
from autogen.events.agent_events import TerminationAndHumanReplyNoInputEvent, TerminationEvent, UsingAutoReplyEvent
from autogen.events.base_event import BaseEvent
from autogen.io.base import AsyncIOStreamProtocol

from .protocol import RemoteService, RequestMessage, ResponseMessage, get_tool_names


class AgentService(RemoteService):
    def __init__(self, agent: ConversableAgent) -> None:
        self.name = agent.name
        self.agent = agent

    async def __call__(self, state: RequestMessage) -> ResponseMessage | None:
        out_message: dict[str, Any] | None
        if guardrail_result := self.agent.run_input_guardrails(state.messages):
            # input guardrail activated by initial messages
            _, out_message = normilize_message_to_oai(guardrail_result.reply, self.agent.name, role="assistant")
            return ResponseMessage(messages=[out_message], context=state.context)

        context_variables = ContextVariables(state.context)
        tool_executor = self._make_tool_executor(context_variables)

        local_history: list[dict[str, Any]] = []
        while True:
            messages = state.messages + local_history

            stream = HITLStream()
            await self.agent.a_check_termination_and_human_reply(messages, iostream=stream)
            if stream.is_input_required:
                return ResponseMessage(
                    messages=local_history,
                    context=context_variables.data or None,
                    input_required=stream.input_prompt,
                )

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

            tool_result, updated_context_variables, return_to_user = self._try_execute_local_tool(
                tool_executor, out_message
            )

            if updated_context_variables:
                context_variables.update(updated_context_variables.to_dict())

            should_continue, out_message = self._add_message_to_local_history(tool_result, role="tool")
            if out_message:
                local_history.append(out_message)

            if return_to_user:
                return ResponseMessage(
                    messages=local_history,
                    context=context_variables.data or None,
                    input_required="Please, provide additional information:\n",
                )

            if not should_continue:
                break

        if not local_history:
            return None

        return ResponseMessage(
            messages=local_history,
            context=context_variables.data or None,
        )

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
    ) -> tuple[dict[str, Any] | None, ContextVariables | None, bool]:
        tool_result: dict[str, Any] | None = None
        updated_context_variables: ContextVariables | None = None

        if "tool_calls" in tool_message:
            _, tool_result = tool_executor.generate_tool_calls_reply([tool_message])
            if tool_result is None:
                return tool_result, updated_context_variables, False

            if "tool_responses" in tool_result:
                # TODO: catch handoffs
                for tool_response in tool_result["tool_responses"]:
                    content = tool_response["content"]

                    if isinstance(content, AskUserTarget):
                        return tool_result, updated_context_variables, True

                    if isinstance(content, TransitionTarget):
                        warnings.warn(
                            f"Tool {self.agent.name} returned a target, which is not supported in remote mode"
                        )

                    elif isinstance(content, ReplyResult):
                        if content.context_variables:
                            updated_context_variables = content.context_variables
                            tool_response["content"] = content.message

                        if isinstance(content.target, AskUserTarget):
                            return tool_result, updated_context_variables, True

                        if content.target:
                            warnings.warn(
                                f"Tool {self.agent.name} returned a target, which is not supported in remote mode"
                            )

        return tool_result, updated_context_variables, False


class HITLStream(AsyncIOStreamProtocol):
    def __init__(self) -> None:
        self.input_prompt = ""

    @property
    def is_input_required(self) -> bool:
        return bool(self.input_prompt)

    async def input(self, prompt: str = "", *, password: bool = False) -> str:
        self.input_prompt = prompt
        return ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
        raise NotImplementedError("HITLStream does not support printing")

    def send(self, message: BaseEvent) -> None:
        if isinstance(
            message,
            (
                UsingAutoReplyEvent,
                TerminationAndHumanReplyNoInputEvent,
                TerminationEvent,
            ),
        ):
            return

        raise NotImplementedError("HITLStream does not support sending messages")
