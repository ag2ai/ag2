# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict

from autogen.agentchat.conversable_agent import ConversableAgent

from .tool import Tool

__all__ = ["PydanticAITool"]


class PydanticAITool(Tool):
    def __init__(
        self, name: str, description: str, func: Callable[..., Any], parameters_json_schema: Dict[str, Any]
    ) -> None:
        super().__init__(name, description, func)
        self._func_schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters_json_schema,
            },
        }

    def register_for_llm(self, agent: ConversableAgent) -> None:
        agent.update_tool_signature(self._func_schema, is_remove=False)
