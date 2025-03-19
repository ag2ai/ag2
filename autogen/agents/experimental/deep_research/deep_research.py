# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Union

from .... import ConversableAgent
from ....doc_utils import export_module
from ....llm_config import LLMConfig
from ....tools.experimental import DeepResearchTool

__all__ = ["DeepResearchAgent"]


@export_module("autogen.agents.experimental")
class DeepResearchAgent(ConversableAgent):
    """An agent that performs deep research tasks."""

    DEFAULT_PROMPT = "You are a deep research agent. You have the ability to get information from the web and perform research tasks."

    def __init__(
        self,
        name: str,
        llm_config: Optional[Union[LLMConfig, dict[str, Any]]] = None,
        system_message: Optional[Union[str, list[str]]] = DEFAULT_PROMPT,
        max_web_steps: int = 30,
        **kwargs: Any,
    ) -> None:
        """Initialize the DeepResearchAgent.

        Args:
            name: The name of the agent.
            llm_config: The LLM configuration.
            system_message: The system message. Defaults to DEFAULT_PROMPT.
            max_web_steps: The maximum number of web steps. Defaults to 30.
            **kwargs: Additional keyword arguments to pass to the ConversableAgent.
        """
        llm_config = LLMConfig.get_current_llm_config(llm_config)  # type: ignore[arg-type]

        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            **kwargs,
        )

        self.tool = DeepResearchTool(
            llm_config=llm_config,  # type: ignore[arg-type]
            max_web_steps=max_web_steps,
        )

        self.register_for_llm()(self.tool)
