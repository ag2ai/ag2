from enum import Enum
from termcolor import colored
from pydantic import BaseModel, Field
from typing import Any, Optional, Union
from io.base import IOStream


class MessageRole(str, Enum):
    ASSISTANT = "assistant"
    FUNCTION = "function" 
    TOOL = "tool"


class BaseMessage(BaseModel):
    content: str
    color: Optional[str] = None
    flush: bool = False
    end: str = "\n"
    sep: str = " "
    
    def print(self):
        iostream = IOStream.get_default()
        text = colored(self.content, self.color) if self.color else self.content
        iostream.print(text, flush=self.flush, end=self.end, sep=self.sep)


class FunctionMessage(BaseMessage):
    role: MessageRole
    id: str
    content: str
    
    def print(self):
        iostream = IOStream.get_default()
        header = f"***** Response from calling {self.role} ({self.id}) *****"
        iostream.print(colored(header, "green"), flush=True)
        iostream.print(self.content, flush=True) 
        iostream.print(colored("*" * len(header), "green"), flush=True)


def create_message_model(message: Union[dict[str, Any], str]) -> BaseMessage:
    if isinstance(message, dict):
        if "role" in message:
            return FunctionMessage(**message)
        else:
            return BaseMessage(**message)
    elif isinstance(message, str):
        return BaseMessage(content=message)
    else:
        raise ValueError(f"Invalid message type: {type(message)}")
