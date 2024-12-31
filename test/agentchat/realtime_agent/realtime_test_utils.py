# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import base64
import os
from functools import wraps
from typing import Any, Callable
from unittest.mock import MagicMock

from anyio import Event
from openai import OpenAI


def generate_voice_input(text: str) -> str:
    tts_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = tts_client.audio.speech.create(model="tts-1", voice="alloy", input=text, response_format="pcm")
    return base64.b64encode(response.content).decode("utf-8")


def trace(mock: MagicMock, event: Event) -> Callable[..., Any]:
    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(f)
        def _inner(*args: Any, **kwargs: Any) -> Any:
            mock(*args, **kwargs)
            event.set()
            return f(*args, **kwargs)

        return _inner

    return decorator
