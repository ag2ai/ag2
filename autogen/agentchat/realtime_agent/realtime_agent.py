# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional, TypeVar, Union

import websockets

from autogen.agentchat.agent import Agent, LLMAgent
from autogen.function_utils import get_function_schema

from .client import Client
from .function_observer import FunctionObserver
from .realtime_observer import RealtimeObserver

F = TypeVar("F", bound=Callable[..., Any])


class RealtimeAgent(LLMAgent):
    def __init__(
        self,
        name: str,
        audio_adapter: RealtimeObserver,
        system_message: Optional[Union[str, List]] = "You are a helpful AI Assistant.",
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = "TERMINATE",
        function_map: Optional[Dict[str, Callable]] = None,
        code_execution_config: Union[Dict, Literal[False]] = False,
        llm_config: Optional[Union[Dict, Literal[False]]] = None,
        default_auto_reply: Union[str, Dict] = "",
        description: Optional[str] = None,
        chat_messages: Optional[Dict[Agent, List[Dict]]] = None,
        silent: Optional[bool] = None,
        context_variables: Optional[Dict[str, Any]] = None,
        voice: str = "alloy",
    ):

        self._client = Client(self, audio_adapter, FunctionObserver(self))
        self.llm_config = llm_config
        self.voice = voice
        self.registered_functions = {}

        self._oai_system_message = [{"content": system_message, "role": "system"}]  # todo still needed?

    async def run(self):
        await self._client.run()

    def register_handover(
        self,
        *,
        description: str,
        name: Optional[str] = None,
    ) -> Callable[[F], F]:
        def _decorator(func: F, name=name) -> F:
            """Decorator for registering a function to be used by an agent.

            Args:
                func: the function to be registered.

            Returns:
                The function to be registered, with the _description attribute set to the function description.

            Raises:
                ValueError: if the function description is not provided and not propagated by a previous decorator.
                RuntimeError: if the LLM config is not set up before registering a function.

            """
            # get JSON schema for the function
            name = name or func.__name__

            schema = get_function_schema(func, name=name, description=description)["function"]
            schema["type"] = "function"

            self.registered_functions[name] = (schema, func)

            return func

        return _decorator
