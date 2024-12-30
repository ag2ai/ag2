# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Any, Callable, Literal, Optional, Union

from pydantic import BaseModel
from termcolor import colored

from .agentchat.agent import Agent
from .code_utils import content_str
from .oai.client import OpenAIWrapper

MessageRole = Literal["assistant", "function", "tool"]


class BaseMessage(BaseModel):
    content: Union[str, int, float, bool]
    sender_name: str
    recipient_name: str

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print
        f(f"{colored(self.sender_name, 'yellow')} (to {self.recipient_name}):\n", flush=True)


class FunctionResponseMessage(BaseMessage):
    name: Optional[str] = None
    role: MessageRole = "function"
    content: Union[str, int, float, bool]

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print
        super().print(f)

        id = self.name or "No id found"
        func_print = f"***** Response from calling {self.role} ({id}) *****"
        f(colored(func_print, "green"), flush=True)
        f(self.content, flush=True)
        f(colored("*" * len(func_print), "green"), flush=True)

        f("\n", "-" * 80, flush=True, sep="")


class ToolResponse(BaseModel):
    tool_call_id: Optional[str] = None
    role: MessageRole = "tool"
    content: Union[str, int, float, bool]

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print
        id = self.tool_call_id or "No id found"
        tool_print = f"***** Response from calling {self.role} ({id}) *****"
        f(colored(tool_print, "green"), flush=True)
        f(self.content, flush=True)
        f(colored("*" * len(tool_print), "green"), flush=True)


class ToolResponseMessage(BaseMessage):
    role: MessageRole = "tool"
    tool_responses: list[ToolResponse]
    content: Union[str, int, float, bool]

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print
        super().print(f)

        for tool_response in self.tool_responses:
            tool_response.print(f)
            f("\n", "-" * 80, flush=True, sep="")


class FunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        name = self.name or "(No function name found)"
        arguments = self.arguments or "(No arguments found)"

        func_print = f"***** Suggested function call: {name} *****"
        f(colored(func_print, "green"), flush=True)
        f(
            "Arguments: \n",
            arguments,
            flush=True,
            sep="",
        )
        f(colored("*" * len(func_print), "green"), flush=True)


class FunctionCallMessage(BaseMessage):
    content: Optional[Union[str, int, float, bool]] = None  # type: ignore [assignment]
    function_call: FunctionCall

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print
        super().print(f)

        if self.content is not None:
            f(self.content, flush=True)

        self.function_call.print(f)

        f("\n", "-" * 80, flush=True, sep="")


class ToolCall(BaseModel):
    id: Optional[str] = None
    function: FunctionCall
    type: str

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        id = self.id or "No tool call id found"

        name = self.function.name or "(No function name found)"
        arguments = self.function.arguments or "(No arguments found)"

        func_print = f"***** Suggested tool call ({id}): {name} *****"
        f(colored(func_print, "green"), flush=True)
        f(
            "Arguments: \n",
            arguments,
            flush=True,
            sep="",
        )
        f(colored("*" * len(func_print), "green"), flush=True)


class ToolCallMessage(BaseMessage):
    content: Optional[Union[str, int, float, bool]] = None  # type: ignore [assignment]
    refusal: Optional[str] = None
    role: Optional[MessageRole] = None
    audio: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: list[ToolCall]

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print
        super().print(f)

        if self.content is not None:
            f(self.content, flush=True)

        for tool_call in self.tool_calls:
            tool_call.print(f)

        f("\n", "-" * 80, flush=True, sep="")


class ContentMessage(BaseMessage):
    content: Optional[Union[str, int, float, bool, Callable[..., Any]]] = None  # type: ignore [assignment]
    context: Optional[dict[str, Any]] = None
    llm_config: Optional[Union[dict[str, Any], Literal[False]]] = None

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print
        super().print(f)

        if self.content is not None:
            allow_format_str_template = (
                self.llm_config.get("allow_format_str_template", False) if self.llm_config else False
            )
            content = OpenAIWrapper.instantiate(
                self.content,  # type: ignore [arg-type]
                self.context,
                allow_format_str_template,
            )
            f(content_str(content), flush=True)

        f("\n", "-" * 80, flush=True, sep="")


def create_received_message_model(message: dict[str, Any], sender: Agent, recipient: Agent) -> BaseMessage:
    # print(f"{message=}")
    # print(f"{sender=}")

    role = message.get("role")
    if role == "function":
        return FunctionResponseMessage(**message, sender_name=sender.name, recipient_name=recipient.name)
    if role == "tool":
        return ToolResponseMessage(**message, sender_name=sender.name, recipient_name=recipient.name)

    # Role is neither function nor tool

    if "function_call" in message and message["function_call"]:
        return FunctionCallMessage(
            **message,
            sender_name=sender.name,
            recipient_name=recipient.name,
        )

    if "tool_calls" in message and message["tool_calls"]:
        return ToolCallMessage(
            **message,
            sender_name=sender.name,
            recipient_name=recipient.name,
        )

    # Now message is a simple content message

    return ContentMessage(
        **message,
        sender_name=sender.name,
        recipient_name=recipient.name,
        llm_config=recipient.llm_config,  # type: ignore [attr-defined]
    )


class PostCarryoverProcessing(BaseModel):
    carryover: Union[str, list[Union[str, dict[str, Any], Any]]]
    message: Optional[Union[str, dict[str, Any], Callable[..., Any]]] = None
    verbose: bool = False

    sender_name: str
    recipient_name: str
    summary_method: Union[str, Callable[..., Any]]
    summary_args: Optional[dict[str, Any]] = None
    max_turns: Optional[int] = None

    def _process_carryover(self) -> str:
        if not isinstance(self.carryover, list):
            return self.carryover

        print_carryover = []
        for carryover_item in self.carryover:
            if isinstance(carryover_item, str):
                print_carryover.append(carryover_item)
            elif isinstance(carryover_item, dict) and "content" in carryover_item:
                print_carryover.append(str(carryover_item["content"]))
            else:
                print_carryover.append(str(carryover_item))

        return ("\n").join(print_carryover)

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        print_carryover = self._process_carryover()

        if isinstance(self.message, str):
            print_message = self.message
        elif callable(self.message):
            print_message = "Callable: " + self.message.__name__
        elif isinstance(self.message, dict):
            print_message = "Dict: " + str(self.message)
        elif self.message is None:
            print_message = "None"
        f(colored("\n" + "*" * 80, "blue"), flush=True, sep="")
        f(
            colored(
                "Starting a new chat....",
                "blue",
            ),
            flush=True,
        )
        if self.verbose:
            f(colored("Message:\n" + print_message, "blue"), flush=True)
            f(colored("Carryover:\n" + print_carryover, "blue"), flush=True)
        f(colored("\n" + "*" * 80, "blue"), flush=True, sep="")


def create_post_carryover_processing(chat_info: dict[str, Any]) -> PostCarryoverProcessing:
    if "message" not in chat_info:
        chat_info["message"] = None
    return PostCarryoverProcessing(
        **chat_info,
        sender_name=chat_info["sender"].name,
        recipient_name=chat_info["recipient"].name,
    )


class ClearAgentsHistory(BaseModel):
    agent_name: Optional[str] = None
    nr_messages_to_preserve: Optional[int] = None

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        if self.agent_name:
            if self.nr_messages_to_preserve:
                f(f"Clearing history for {self.agent_name} except last {self.nr_messages_to_preserve} messages.")
            else:
                f(f"Clearing history for {self.agent_name}.")
        else:
            if self.nr_messages_to_preserve:
                f(f"Clearing history for all agents except last {self.nr_messages_to_preserve} messages.")
            else:
                f("Clearing history for all agents.")


def create_clear_agents_history(
    agent: Optional[Agent] = None, nr_messages_to_preserve: Optional[int] = None
) -> ClearAgentsHistory:
    return ClearAgentsHistory(agent_name=agent.name if agent else None, nr_messages_to_preserve=nr_messages_to_preserve)


class SpeakerAttempt(BaseModel):
    mentions: dict[str, int]
    attempt: int
    attempts_left: int
    verbose: Optional[bool] = False

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        if not self.verbose:
            return

        if len(self.mentions) == 1:
            # Success on retry, we have just one name mentioned
            selected_agent_name = next(iter(self.mentions))
            f(
                colored(
                    f">>>>>>>> Select speaker attempt {self.attempt} of {self.attempt + self.attempts_left} successfully selected: {selected_agent_name}",
                    "green",
                ),
                flush=True,
            )
        elif len(self.mentions) > 1:
            f(
                colored(
                    f">>>>>>>> Select speaker attempt {self.attempt} of {self.attempt + self.attempts_left} failed as it included multiple agent names.",
                    "red",
                ),
                flush=True,
            )
        else:
            f(
                colored(
                    f">>>>>>>> Select speaker attempt #{self.attempt} failed as it did not include any agent names.",
                    "red",
                ),
                flush=True,
            )


def create_speaker_attempt(
    mentions: dict[str, int], attempt: int, attempts_left: int, select_speaker_auto_verbose: Optional[bool] = False
) -> SpeakerAttempt:
    return SpeakerAttempt(
        mentions=deepcopy(mentions), attempt=attempt, attempts_left=attempts_left, verbose=select_speaker_auto_verbose
    )


class GroupChatResume(BaseModel):
    last_speaker_name: str
    messages: list[dict[str, Any]]
    verbose: Optional[bool] = False

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        if self.verbose:
            f(
                f"Prepared group chat with {len(self.messages)} messages, the last speaker is",
                colored(self.last_speaker_name, "yellow"),
                flush=True,
            )


def create_group_chat_resume(
    last_speaker_name: str, messages: list[dict[str, Any]], silent: Optional[bool] = False
) -> GroupChatResume:
    return GroupChatResume(last_speaker_name=last_speaker_name, messages=messages, verbose=not silent)


class GroupChatRunChat(BaseModel):
    speaker_name: str
    verbose: Optional[bool] = False

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        if self.verbose:
            f(colored(f"\nNext speaker: {self.speaker_name}\n", "green"), flush=True)


def create_group_chat_run_chat(speaker: Agent, silent: Optional[bool] = False) -> GroupChatRunChat:
    return GroupChatRunChat(speaker_name=speaker.name, verbose=not silent)


class TerminationAndHumanReply(BaseModel):
    no_human_input_msg: str
    human_input_mode: str
    sender_name: str
    recipient_name: str

    def print_no_human_input_msg(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        if self.no_human_input_msg:
            f(colored(f"\n>>>>>>>> {self.no_human_input_msg}", "red"), flush=True)

    def print_human_input_mode(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        if self.human_input_mode != "NEVER":
            f(colored("\n>>>>>>>> USING AUTO REPLY...", "red"), flush=True)


def create_termination_and_human_reply(
    no_human_input_msg: str, human_input_mode: str, *, sender: Optional[Agent] = None, recipient: Agent
) -> TerminationAndHumanReply:
    return TerminationAndHumanReply(
        no_human_input_msg=no_human_input_msg,
        human_input_mode=human_input_mode,
        sender_name=sender.name if sender else "No sender",
        recipient_name=recipient.name,
    )


class ExecuteCodeBlock(BaseModel):
    code: str
    language: str
    code_block_count: int
    recipient_name: str

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f(
            colored(
                f"\n>>>>>>>> EXECUTING CODE BLOCK {self.code_block_count} (inferred language is {self.language})...",
                "red",
            ),
            flush=True,
        )


def create_execute_code_block(code: str, language: str, code_block_count: int, recipient: Agent) -> ExecuteCodeBlock:
    return ExecuteCodeBlock(
        code=code, language=language, code_block_count=code_block_count, recipient_name=recipient.name
    )
