from enum import Enum
from pprint import pprint
from typing import Any, Callable, Literal, Optional, TypeVar, Union

from pydantic import BaseModel, Field
from termcolor import colored

from .agentchat.agent import Agent
from .io.base import IOStream

MessageRole = TypeVar("MessageRole", bound=Literal["assistant", "function", "tool"])


class BaseMessage(BaseModel):
    content: str
    sender_name: str
    receiver_name: str

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print
        f(f"{colored(self.sender_name, 'yellow')} (to {self.receiver_name}):\n", flush=True)


class FunctionMessage(BaseMessage):
    role: MessageRole
    id: str
    content: str

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        pass

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


def create_message_model(message: Union[dict[str, Any], str], sender: Agent, receiver: Agent) -> BaseMessage:
    pprint(f"{message=}")
    pprint(f"{sender=}")
    if isinstance(message, str):
        return

    if message.get("role") == "function":
        raise ValueError("FunctionMessage is not supported")
    
    if message.get("role") == "tool":
        return ToolResponseMessage(**message, sender_name=sender.name, receiver_name=receiver.name)

    return
    if isinstance(message, dict):
        if "role" in message:
            return FunctionMessage(**message)
        else:
            return BaseMessage(**message)
    elif isinstance(message, str):
        return BaseMessage(content=message)
    else:
        raise ValueError(f"Invalid message type: {type(message)}")
