# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, Optional

from .... import ConversableAgent
from ....doc_utils import export_module
from ....tools import Tool
from ....tools.experimental import BrowserUseTool, Crawl4AITool

__all__ = ["WebSurferAgent"]


@export_module("autogen.agents.experimental")
class WebSurferAgent(ConversableAgent):
    """An agent that uses web tools to interact with the web."""

    def __init__(
        self,
        *,
        llm_config: dict[str, Any],
        web_tool_llm_config: Optional[dict[str, Any]] = None,
        web_tool: Literal["browser_use", "crawl4ai"] = "browser_use",
        web_tool_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the WebSurferAgent.

        Args:
            llm_config: The LLM configuration.
            web_tool_llm_config: The LLM configuration for the web tool. I not provided, the llm_config will be used.
            web_tool: The web tool to use. Defaults to "browser_use".
            web_tool_kwargs: The keyword arguments for the web tool. Defaults to None.
            **kwargs: Additional keyword arguments passed to the parent ConversableAgent class.
        """
        web_tool_kwargs = web_tool_kwargs if web_tool_kwargs else {}
        web_tool_llm_config = web_tool_llm_config if web_tool_llm_config else llm_config
        if web_tool == "browser_use":
            self.tool: Tool = BrowserUseTool(llm_config=web_tool_llm_config, **web_tool_kwargs)
        elif web_tool == "crawl4ai":
            self.tool = Crawl4AITool(llm_config=web_tool_llm_config, **web_tool_kwargs)
        else:
            raise ValueError(f"Unsupported {web_tool=}.")

        super().__init__(llm_config=llm_config, **kwargs)

        self.register_for_llm()(self.tool)
