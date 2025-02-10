# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Union

from .... import ConversableAgent
from ....doc_utils import export_module
from ....tools import Tool
from .deep_research_tool import DeepResearchTool

__all__ = ["DeepResearchAgent"]


@export_module("autogen.agents.experimental")
class DeepResearchAgent(ConversableAgent):
    DEFAULT_PROMPT = "You are a deep research agent. You have the ability to get information from the web and perform research tasks."

    def __init__(
        self,
        name: str,
        llm_config: dict[str, Any],
        system_message: Optional[Union[str, list[str]]] = DEFAULT_PROMPT,
        max_web_steps: int = 30,
        **kwargs,
    ):
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            **kwargs,
        )

        self.tool = DeepResearchTool(
            llm_config=llm_config,
            max_web_steps=max_web_steps,
        )

        self.register_for_llm()(self.tool)

    @property
    def tools(self) -> list[Tool]:
        return [self.tool]
