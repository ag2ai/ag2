# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel, field_validator
from termcolor import colored

from ..agentchat.agent import LLMMessageType
from ..code_utils import content_str
from ..import_utils import optional_import_block, require_optional_import
from ..oai.client import OpenAIWrapper
from .base_event import BaseEvent, wrap_event

with optional_import_block() as result:
    from PIL.Image import Image

IS_PIL_AVAILABLE = result.is_successful

if TYPE_CHECKING:
    from ..agentchat.agent import Agent
    from ..coding.base import CodeBlock


__all__ = [
    "ClearAgentsHistoryEvent",
    "ClearConversableAgentHistoryEvent",
    "ConversableAgentUsageSummaryEvent",
    "ConversableAgentUsageSummaryNoCostIncurredEvent",
    "ExecuteCodeBlockEvent",
    "ExecuteFunctionEvent",
    "FunctionCallEvent",
    "FunctionResponseEvent",
    "GenerateCodeExecutionReplyEvent",
    "GroupChatResumeEvent",
    "GroupChatRunChatEvent",
    "PostCarryoverProcessingEvent",
    "SelectSpeakerEvent",
    "SpeakerAttemptFailedMultipleAgentsEvent",
    "SpeakerAttemptFailedNoAgentsEvent",
    "SpeakerAttemptSuccessfulEvent",
    "TerminationAndHumanReplyNoInputEvent",
    "TerminationEvent",
    "TextEvent",
    "ToolCallEvent",
    "ToolResponseEvent",
]

EventRole = Literal["assistant", "function", "tool"]


class BasePrintReceivedEvent(BaseEvent, ABC):
    content: Union[str, int, float, bool]
    sender_name: str
    recipient_name: str

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print
        f(f"{colored(self.sender_name, 'yellow')} (to {self.recipient_name}):\n", flush=True)


@wrap_event
class FunctionResponseEvent(BasePrintReceivedEvent):
    name: Optional[str] = None
    role: EventRole = "function"
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
    role: EventRole = "tool"
    content: Union[str, int, float, bool]

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print
        id = self.tool_call_id or "No id found"
        tool_print = f"***** Response from calling {self.role} ({id}) *****"
        f(colored(tool_print, "green"), flush=True)
        f(self.content, flush=True)
        f(colored("*" * len(tool_print), "green"), flush=True)


@wrap_event
class ToolResponseEvent(BasePrintReceivedEvent):
    role: EventRole = "tool"
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


@wrap_event
class FunctionCallEvent(BasePrintReceivedEvent):
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


@wrap_event
class ToolCallEvent(BasePrintReceivedEvent):
    content: Optional[Union[str, int, float, bool]] = None  # type: ignore [assignment]
    refusal: Optional[str] = None
    role: Optional[EventRole] = None
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


@wrap_event
class TextEvent(BasePrintReceivedEvent):
    content: Optional[Union[str, int, float, bool, list[dict[str, Union[str, dict[str, Any]]]]]] = None  # type: ignore [assignment]

    @classmethod
    @require_optional_import("PIL", "unknown")
    def _replace_pil_image_with_placeholder(cls, image_url: dict[str, Any]) -> None:
        if "url" in image_url and isinstance(image_url["url"], Image):
            image_url["url"] = "<image>"

    @field_validator("content", mode="before")
    @classmethod
    def validate_and_encode_content(
        cls, content: Optional[Union[str, int, float, bool, list[dict[str, Union[str, dict[str, Any]]]]]]
    ) -> Optional[Union[str, int, float, bool, list[dict[str, Union[str, dict[str, Any]]]]]]:
        if not IS_PIL_AVAILABLE:
            return content

        if not isinstance(content, list):
            return content

        for item in content:
            if isinstance(item, dict) and "image_url" in item and isinstance(item["image_url"], dict):
                cls._replace_pil_image_with_placeholder(item["image_url"])

        return content

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print
        super().print(f)

        if self.content is not None:
            f(content_str(self.content), flush=True)  # type: ignore [arg-type]

        f("\n", "-" * 80, flush=True, sep="")


def create_received_event_model(
    *, uuid: Optional[UUID] = None, event: dict[str, Any], sender: "Agent", recipient: "Agent"
) -> Union[FunctionResponseEvent, ToolResponseEvent, FunctionCallEvent, ToolCallEvent, TextEvent]:
    role = event.get("role")
    if role == "function":
        return FunctionResponseEvent(**event, sender_name=sender.name, recipient_name=recipient.name, uuid=uuid)
    if role == "tool":
        return ToolResponseEvent(**event, sender_name=sender.name, recipient_name=recipient.name, uuid=uuid)

    # Role is neither function nor tool

    if event.get("function_call"):
        return FunctionCallEvent(
            **event,
            sender_name=sender.name,
            recipient_name=recipient.name,
            uuid=uuid,
        )

    if event.get("tool_calls"):
        return ToolCallEvent(
            **event,
            sender_name=sender.name,
            recipient_name=recipient.name,
            uuid=uuid,
        )

    # Now message is a simple content message
    content = event.get("content")
    allow_format_str_template = (
        recipient.llm_config.get("allow_format_str_template", False) if recipient.llm_config else False  # type: ignore [attr-defined]
    )
    if content is not None and "context" in event:
        content = OpenAIWrapper.instantiate(
            content,  # type: ignore [arg-type]
            event["context"],
            allow_format_str_template,
        )

    return TextEvent(
        content=content,
        sender_name=sender.name,
        recipient_name=recipient.name,
        uuid=uuid,
    )


@wrap_event
class PostCarryoverProcessingEvent(BaseEvent):
    carryover: Union[str, list[Union[str, dict[str, Any], Any]]]
    message: str
    verbose: bool = False

    sender_name: str
    recipient_name: str
    summary_method: str
    summary_args: Optional[dict[str, Any]] = None
    max_turns: Optional[int] = None

    def __init__(self, *, uuid: Optional[UUID] = None, chat_info: dict[str, Any]):
        carryover = chat_info.get("carryover", "")
        message = chat_info.get("message")
        verbose = chat_info.get("verbose", False)

        sender_name = chat_info["sender"].name
        recipient_name = chat_info["recipient"].name
        summary_args = chat_info.get("summary_args")
        max_turns = chat_info.get("max_turns")

        # Fix Callable in chat_info
        summary_method = chat_info.get("summary_method", "")
        if callable(summary_method):
            summary_method = summary_method.__name__

        print_message = ""
        if isinstance(message, str):
            print_message = message
        elif callable(message):
            print_message = "Callable: " + message.__name__
        elif isinstance(message, dict):
            print_message = "Dict: " + str(message)
        elif message is None:
            print_message = "None"

        super().__init__(
            uuid=uuid,
            carryover=carryover,
            message=print_message,
            verbose=verbose,
            summary_method=summary_method,
            summary_args=summary_args,
            max_turns=max_turns,
            sender_name=sender_name,
            recipient_name=recipient_name,
        )

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

        f(colored("\n" + "*" * 80, "blue"), flush=True, sep="")
        f(
            colored(
                "Starting a new chat....",
                "blue",
            ),
            flush=True,
        )
        if self.verbose:
            f(colored("Event:\n" + self.message, "blue"), flush=True)
            f(colored("Carryover:\n" + print_carryover, "blue"), flush=True)
        f(colored("\n" + "*" * 80, "blue"), flush=True, sep="")


@wrap_event
class ClearAgentsHistoryEvent(BaseEvent):
    agent_name: Optional[str] = None
    nr_events_to_preserve: Optional[int] = None

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        agent: Optional["Agent"] = None,
        nr_events_to_preserve: Optional[int] = None,
    ):
        return super().__init__(
            uuid=uuid, agent_name=agent.name if agent else None, nr_events_to_preserve=nr_events_to_preserve
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        if self.agent_name:
            if self.nr_events_to_preserve:
                f(f"Clearing history for {self.agent_name} except last {self.nr_events_to_preserve} events.")
            else:
                f(f"Clearing history for {self.agent_name}.")
        else:
            if self.nr_events_to_preserve:
                f(f"Clearing history for all agents except last {self.nr_events_to_preserve} events.")
            else:
                f("Clearing history for all agents.")


# todo: break into multiple events
@wrap_event
class SpeakerAttemptSuccessfulEvent(BaseEvent):
    mentions: dict[str, int]
    attempt: int
    attempts_left: int
    verbose: Optional[bool] = False

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        mentions: dict[str, int],
        attempt: int,
        attempts_left: int,
        select_speaker_auto_verbose: Optional[bool] = False,
    ):
        super().__init__(
            uuid=uuid,
            mentions=deepcopy(mentions),
            attempt=attempt,
            attempts_left=attempts_left,
            verbose=select_speaker_auto_verbose,
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        selected_agent_name = next(iter(self.mentions))
        f(
            colored(
                f">>>>>>>> Select speaker attempt {self.attempt} of {self.attempt + self.attempts_left} successfully selected: {selected_agent_name}",
                "green",
            ),
            flush=True,
        )


@wrap_event
class SpeakerAttemptFailedMultipleAgentsEvent(BaseEvent):
    mentions: dict[str, int]
    attempt: int
    attempts_left: int
    verbose: Optional[bool] = False

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        mentions: dict[str, int],
        attempt: int,
        attempts_left: int,
        select_speaker_auto_verbose: Optional[bool] = False,
    ):
        super().__init__(
            uuid=uuid,
            mentions=deepcopy(mentions),
            attempt=attempt,
            attempts_left=attempts_left,
            verbose=select_speaker_auto_verbose,
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f(
            colored(
                f">>>>>>>> Select speaker attempt {self.attempt} of {self.attempt + self.attempts_left} failed as it included multiple agent names.",
                "red",
            ),
            flush=True,
        )


@wrap_event
class SpeakerAttemptFailedNoAgentsEvent(BaseEvent):
    mentions: dict[str, int]
    attempt: int
    attempts_left: int
    verbose: Optional[bool] = False

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        mentions: dict[str, int],
        attempt: int,
        attempts_left: int,
        select_speaker_auto_verbose: Optional[bool] = False,
    ):
        super().__init__(
            uuid=uuid,
            mentions=deepcopy(mentions),
            attempt=attempt,
            attempts_left=attempts_left,
            verbose=select_speaker_auto_verbose,
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f(
            colored(
                f">>>>>>>> Select speaker attempt #{self.attempt} failed as it did not include any agent names.",
                "red",
            ),
            flush=True,
        )


@wrap_event
class GroupChatResumeEvent(BaseEvent):
    last_speaker_name: str
    events: list[LLMMessageType]
    verbose: Optional[bool] = False

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        last_speaker_name: str,
        events: list["LLMMessageType"],
        silent: Optional[bool] = False,
    ):
        super().__init__(uuid=uuid, last_speaker_name=last_speaker_name, events=events, verbose=not silent)

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f(
            f"Prepared group chat with {len(self.events)} events, the last speaker is",
            colored(self.last_speaker_name, "yellow"),
            flush=True,
        )


@wrap_event
class GroupChatRunChatEvent(BaseEvent):
    speaker_name: str
    verbose: Optional[bool] = False

    def __init__(self, *, uuid: Optional[UUID] = None, speaker: "Agent", silent: Optional[bool] = False):
        super().__init__(uuid=uuid, speaker_name=speaker.name, verbose=not silent)

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f(colored(f"\nNext speaker: {self.speaker_name}\n", "green"), flush=True)


@wrap_event
class TerminationAndHumanReplyNoInputEvent(BaseEvent):
    """When the human-in-the-loop is prompted but provides no input."""

    no_human_input_msg: str
    sender_name: str
    recipient_name: str

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        no_human_input_msg: str,
        sender: Optional["Agent"] = None,
        recipient: "Agent",
    ):
        super().__init__(
            uuid=uuid,
            no_human_input_msg=no_human_input_msg,
            sender_name=sender.name if sender else "No sender",
            recipient_name=recipient.name,
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f(colored(f"\n>>>>>>>> {self.no_human_input_msg}", "red"), flush=True)


@wrap_event
class UsingAutoReplyEvent(BaseEvent):
    human_input_mode: str
    sender_name: str
    recipient_name: str

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        human_input_mode: str,
        sender: Optional["Agent"] = None,
        recipient: "Agent",
    ):
        super().__init__(
            uuid=uuid,
            human_input_mode=human_input_mode,
            sender_name=sender.name if sender else "No sender",
            recipient_name=recipient.name,
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f(colored("\n>>>>>>>> USING AUTO REPLY...", "red"), flush=True)


@wrap_event
class TerminationEvent(BaseEvent):
    """When a workflow termination condition is met"""

    termination_reason: str

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        termination_reason: str,
    ):
        super().__init__(
            uuid=uuid,
            termination_reason=termination_reason,
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f(colored(f"\n>>>>>>>> TERMINATING RUN ({str(self.uuid)}): {self.termination_reason}", "red"), flush=True)


@wrap_event
class ExecuteCodeBlockEvent(BaseEvent):
    code: str
    language: str
    code_block_count: int
    recipient_name: str

    def __init__(
        self, *, uuid: Optional[UUID] = None, code: str, language: str, code_block_count: int, recipient: "Agent"
    ):
        super().__init__(
            uuid=uuid, code=code, language=language, code_block_count=code_block_count, recipient_name=recipient.name
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f(
            colored(
                f"\n>>>>>>>> EXECUTING CODE BLOCK {self.code_block_count} (inferred language is {self.language})...",
                "red",
            ),
            flush=True,
        )


@wrap_event
class ExecuteFunctionEvent(BaseEvent):
    func_name: str
    call_id: Optional[str] = None
    arguments: dict[str, Any]
    recipient_name: str

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        func_name: str,
        call_id: Optional[str] = None,
        arguments: dict[str, Any],
        recipient: "Agent",
    ):
        super().__init__(
            uuid=uuid, func_name=func_name, call_id=call_id, arguments=arguments, recipient_name=recipient.name
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f(
            colored(
                f"\n>>>>>>>> EXECUTING FUNCTION {self.func_name}...\nCall ID: {self.call_id}\nInput arguments: {self.arguments}",
                "magenta",
            ),
            flush=True,
        )


@wrap_event
class ExecutedFunctionEvent(BaseEvent):
    func_name: str
    call_id: Optional[str] = None
    arguments: dict[str, Any]
    content: str
    recipient_name: str

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        func_name: str,
        call_id: Optional[str] = None,
        arguments: dict[str, Any],
        content: str,
        recipient: "Agent",
    ):
        super().__init__(
            uuid=uuid,
            func_name=func_name,
            call_id=call_id,
            arguments=arguments,
            content=content,
            recipient_name=recipient.name,
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f(
            colored(
                f"\n>>>>>>>> EXECUTED FUNCTION {self.func_name}...\nCall ID: {self.call_id}\nInput arguments: {self.arguments}\nOutput:\n{self.content}",
                "magenta",
            ),
            flush=True,
        )


@wrap_event
class SelectSpeakerEvent(BaseEvent):
    agent_names: Optional[list[str]] = None

    def __init__(self, *, uuid: Optional[UUID] = None, agents: Optional[list["Agent"]] = None):
        agent_names = [agent.name for agent in agents] if agents else None
        super().__init__(uuid=uuid, agent_names=agent_names)

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f("Please select the next speaker from the following list:")
        agent_names = self.agent_names or []
        for i, agent_name in enumerate(agent_names):
            f(f"{i + 1}: {agent_name}")


@wrap_event
class SelectSpeakerTryCountExceededEvent(BaseEvent):
    try_count: int
    agent_names: Optional[list[str]] = None

    def __init__(self, *, uuid: Optional[UUID] = None, try_count: int, agents: Optional[list["Agent"]] = None):
        agent_names = [agent.name for agent in agents] if agents else None
        super().__init__(uuid=uuid, try_count=try_count, agent_names=agent_names)

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f(f"You have tried {self.try_count} times. The next speaker will be selected automatically.")


@wrap_event
class SelectSpeakerInvalidInputEvent(BaseEvent):
    agent_names: Optional[list[str]] = None

    def __init__(self, *, uuid: Optional[UUID] = None, agents: Optional[list["Agent"]] = None):
        agent_names = [agent.name for agent in agents] if agents else None
        super().__init__(uuid=uuid, agent_names=agent_names)

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f(f"Invalid input. Please enter a number between 1 and {len(self.agent_names or [])}.")


@wrap_event
class ClearConversableAgentHistoryEvent(BaseEvent):
    agent_name: str
    recipient_name: str
    no_events_preserved: int

    def __init__(self, *, uuid: Optional[UUID] = None, agent: "Agent", no_events_preserved: Optional[int] = None):
        super().__init__(
            uuid=uuid,
            agent_name=agent.name,
            recipient_name=agent.name,
            no_events_preserved=no_events_preserved,
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        for _ in range(self.no_events_preserved):
            f(
                f"Preserving one more event for {self.agent_name} to not divide history between tool call and "
                f"tool response."
            )


@wrap_event
class ClearConversableAgentHistoryWarningEvent(BaseEvent):
    recipient_name: str

    def __init__(self, *, uuid: Optional[UUID] = None, recipient: "Agent"):
        super().__init__(
            uuid=uuid,
            recipient_name=recipient.name,
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f(
            colored(
                "WARNING: `nr_preserved_events` is ignored when clearing chat history with a specific agent.",
                "yellow",
            ),
            flush=True,
        )


@wrap_event
class GenerateCodeExecutionReplyEvent(BaseEvent):
    code_block_languages: list[str]
    sender_name: Optional[str] = None
    recipient_name: str

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        code_blocks: list["CodeBlock"],
        sender: Optional["Agent"] = None,
        recipient: "Agent",
    ):
        code_block_languages = [code_block.language for code_block in code_blocks]

        super().__init__(
            uuid=uuid,
            code_block_languages=code_block_languages,
            sender_name=sender.name if sender else None,
            recipient_name=recipient.name,
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        num_code_blocks = len(self.code_block_languages)
        if num_code_blocks == 1:
            f(
                colored(
                    f"\n>>>>>>>> EXECUTING CODE BLOCK (inferred language is {self.code_block_languages[0]})...",
                    "red",
                ),
                flush=True,
            )
        else:
            f(
                colored(
                    f"\n>>>>>>>> EXECUTING {num_code_blocks} CODE BLOCKS (inferred languages are [{', '.join([x for x in self.code_block_languages])}])...",
                    "red",
                ),
                flush=True,
            )


@wrap_event
class ConversableAgentUsageSummaryNoCostIncurredEvent(BaseEvent):
    recipient_name: str

    def __init__(self, *, uuid: Optional[UUID] = None, recipient: "Agent"):
        super().__init__(uuid=uuid, recipient_name=recipient.name)

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f(f"No cost incurred from agent '{self.recipient_name}'.")


@wrap_event
class ConversableAgentUsageSummaryEvent(BaseEvent):
    recipient_name: str

    def __init__(self, *, uuid: Optional[UUID] = None, recipient: "Agent"):
        super().__init__(uuid=uuid, recipient_name=recipient.name)

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f(f"Agent '{self.recipient_name}':")


@wrap_event
class InputRequestEvent(BaseEvent):
    prompt: str
    password: bool = False
    respond: Optional[Callable[[str], None]] = None

    type: str = "input_request"


@wrap_event
class AsyncInputRequestEvent(BaseEvent):
    prompt: str
    password: bool = False

    async def a_respond(self, response: "InputResponseEvent") -> None:
        pass


@wrap_event
class InputResponseEvent(BaseEvent):
    value: str


@wrap_event
class ErrorEvent(BaseEvent):
    error: Any
