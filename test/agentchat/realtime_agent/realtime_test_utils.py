# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import base64
from functools import wraps
from typing import Any, Callable, Literal, Optional, TypeVar, Union
from unittest.mock import MagicMock

from anyio import Event
from openai import NotGiven, OpenAI

__all__ = ["Credentials", "text_to_speech", "trace"]


def text_to_speech(
    *,
    text: str,
    openai_api_key: str,
    model: str = "tts-1",
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "alloy",
    response_format: Union[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"], NotGiven] = "pcm",
) -> str:
    """Convert text to voice using OpenAI API.

    Args:
        text (str): Text to convert to voice.
        openai_api_key (str): OpenAI API key.
        model (str, optional): Model to use for the conversion. Defaults to "tts-1".
        voice (Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"], optional): Voice to use for the conversion. Defaults to "alloy".
        response_format (Union[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"], NotGiven], optional): Response format. Defaults to "pcm".

    Returns:
        str: Base64 encoded audio.
    """
    tts_client = OpenAI(api_key=openai_api_key)
    response = tts_client.audio.speech.create(model=model, voice=voice, input=text, response_format=response_format)
    return base64.b64encode(response.content).decode("utf-8")


F = TypeVar("F", bound=Callable[..., Any])


def trace(
    mock: MagicMock, *, precall_event: Optional[Event] = None, postcall_event: Optional[Event] = None
) -> Callable[[F], F]:
    """Decorator to trace a function

    Mock will be called before the function.
    If defined, precall_event will be set before the function call and postcall_event will be set after the function call.

    Args:
        mock (MagicMock): Mock object.
        precall_event (Optional[Event], optional): Event to set before the function call. Defaults to None.
        postcall_event (Optional[Event], optional): Event to set after the function call. Defaults to None.

    Returns:
        Callable[[F], F]: Function decorator.
    """

    def decorator(f: F) -> F:
        @wraps(f)
        def _inner(*args: Any, **kwargs: Any) -> Any:
            mock(*args, **kwargs)
            if precall_event is not None:
                precall_event.set()
            retval = f(*args, **kwargs)
            if postcall_event is not None:
                postcall_event.set()

            return retval

        return _inner  # type: ignore[return-value]

    return decorator


class Credentials:
    """Credentials for the OpenAI API."""

    def __init__(self, llm_config: dict[str, Any]) -> None:
        self.llm_config = llm_config

    def sanitize(self) -> dict[str, Any]:
        llm_config = self.llm_config.copy()
        for config in llm_config["config_list"]:
            if "api_key" in config:
                config["api_key"] = "********"
        return llm_config

    def __repr__(self) -> str:
        return repr(self.sanitize())

    def __str___(self) -> str:
        return str(self.sanitize())

    @property
    def openai_api_key(self) -> str:
        return self.llm_config["config_list"][0]["api_key"]  # type: ignore[no-any-return]
