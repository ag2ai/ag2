from enum import Enum
from pprint import pprint
from typing import Any, Callable, Literal, Optional, TypeVar, Union

from pydantic import BaseModel, Field
from termcolor import colored

from .agentchat.agent import Agent
from .code_utils import content_str
from .io.base import IOStream
from .oai.client import OpenAIWrapper

MessageRole = TypeVar("MessageRole", bound=Literal["assistant", "function", "tool"])


class BaseMessage(BaseModel):
    content: str
    sender_name: str
    receiver_name: str

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print
        f(f"{colored(self.sender_name, 'yellow')} (to {self.receiver_name}):\n", flush=True)


class FunctionResponseMessage(BaseMessage):
    name: Optional[str] = None
    role: MessageRole = "function"
    content: str

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
    content: str

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
    content: str

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
    content: Optional[str] = None
    function_call: FunctionCall
    context: Optional[dict[str, Any]] = None
    llm_config: Union[dict, Literal[False]]

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print
        super().print(f)

        if self.content is not None:
            content = self.content
            if self.context is not None:
                content = OpenAIWrapper.instantiate(
                    content,
                    self.context,
                    self.llm_config and self.llm_config.get("allow_format_str_template", False),
                )
            f(content_str(content), flush=True)

        self.function_call.print(f)

        f("\n", "-" * 80, flush=True, sep="")


class ToolCall(BaseModel):
    id: Optional[str] = None
    function: Optional[FunctionCall] = None
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
    content: Optional[str] = None
    refusal: Optional[str] = None
    role: MessageRole
    audio: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: list[ToolCall]
    context: Optional[dict[str, Any]] = None
    llm_config: Union[dict, Literal[False]]

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print
        super().print(f)

        if self.content is not None:
            content = self.content
            if self.context is not None:
                content = OpenAIWrapper.instantiate(
                    content,
                    self.context,
                    self.llm_config and self.llm_config.get("allow_format_str_template", False),
                )
            f(content_str(content), flush=True)

        for tool_call in self.tool_calls:
            tool_call.print(f)

        f("\n", "-" * 80, flush=True, sep="")


class ContentMessage(BaseMessage):
    content: Optional[Union[str, Callable[..., Any]]] = None
    context: Optional[dict[str, Any]] = None
    llm_config: Union[dict, Literal[False]]

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print
        super().print(f)

        if self.content is not None:
            content = self.content
            if self.context is not None:
                content = OpenAIWrapper.instantiate(
                    content,
                    self.context,
                    self.llm_config and self.llm_config.get("allow_format_str_template", False),
                )
            f(content_str(content), flush=True)

        f("\n", "-" * 80, flush=True, sep="")


def create_message_model(message: Union[dict[str, Any], str], sender: Agent, receiver: Agent) -> BaseMessage:
    print(f"{message=}")
    print(f"{sender=}")
    if isinstance(message, str):
        return

    role = message.get("role")
    if role == "function":
        return FunctionResponseMessage(**message, sender_name=sender.name, receiver_name=receiver.name)
    if role == "tool":
        return ToolResponseMessage(**message, sender_name=sender.name, receiver_name=receiver.name)

    # Role is neither function nor tool

    if "function_call" in message and message["function_call"]:
        return FunctionCallMessage(
            **message, sender_name=sender.name, receiver_name=receiver.name, llm_config=receiver.llm_config
        )

    if "tool_calls" in message and message["tool_calls"]:
        return ToolCallMessage(
            **message, sender_name=sender.name, receiver_name=receiver.name, llm_config=receiver.llm_config
        )

    # Now message is a simple content message

    return ContentMessage(
        **message, sender_name=sender.name, receiver_name=receiver.name, llm_config=receiver.llm_config
    )
