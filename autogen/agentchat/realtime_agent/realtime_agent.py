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
        self.llm_config = llm_config
        self._oai_system_message = [{"content": system_message, "role": "system"}]
        self.voice = voice
        self.observers = []
        self.openai_ws = None
        self.registered_functions = {}

        self.register(audio_adapter)

    def register(self, observer):
        observer.register_client(self)
        self.observers.append(observer)

    async def notify_observers(self, message):
        for observer in self.observers:
            await observer.update(message)

    async def function_result(self, call_id, result):
        result_item = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": result,
            },
        }
        await self.openai_ws.send(json.dumps(result_item))
        await self.openai_ws.send(json.dumps({"type": "response.create"}))

    async def _read_from_client(self):
        try:
            async for openai_message in self.openai_ws:
                response = json.loads(openai_message)
                await self.notify_observers(response)
        except Exception as e:
            print(f"Error in _read_from_client: {e}")

    async def run(self):
        self.register(FunctionObserver(registered_functions=self.registered_functions))
        async with websockets.connect(
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
            additional_headers={
                "Authorization": f"Bearer {self.llm_config['config_list'][0]['api_key']}",
                "OpenAI-Beta": "realtime=v1",
            },
        ) as openai_ws:
            self.openai_ws = openai_ws
            await self.initialize_session()
            await asyncio.gather(self._read_from_client(), *[observer.run() for observer in self.observers])

    async def initialize_session(self):
        """Control initial session with OpenAI."""
        session_update = {
            "turn_detection": {"type": "server_vad"},
            "voice": self.voice,
            "instructions": self.system_message,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
        }
        await self.session_update(session_update)

    async def session_update(self, session_options):
        update = {"type": "session.update", "session": session_options}
        print("Sending session update:", json.dumps(update))
        await self.openai_ws.send(json.dumps(update))

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

            schema = get_function_schema(func, name=name, description=description)

            self.registered_functions["name"] = (schema, func)

            return func

        return _decorator
