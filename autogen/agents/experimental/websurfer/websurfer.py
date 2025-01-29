# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from .... import ConversableAgent
from ....tools import Tool
from ....tools.experimental import BrowserUseTool, Crawl4AITool

__all__ = ["WebSurfer"]


class WebSurfer(ConversableAgent):
    def __init__(
        self,
        llm_config: dict[str, Any],
        *args,
        web_tool: Literal["browser_use", "crawl4ai"] = "browser_use",
        web_tool_kwargs: dict[str, Any] = None,
        **kwargs,
    ) -> None:
        web_tool_kwargs = web_tool_kwargs if web_tool_kwargs else {}
        if web_tool == "browser_use":
            self.tool: Tool = BrowserUseTool(llm_config=llm_config, **web_tool_kwargs)
        elif web_tool == "crawl4ai":
            self.tool = Crawl4AITool(llm_config=llm_config, **web_tool_kwargs)
        else:
            raise ValueError(f"Unsupported {web_tool=}.")

        super().__init__(*args, llm_config=llm_config, **kwargs)

        self.register_for_llm()(self.tool)

    @property
    def tools(self) -> list[Tool]:
        return [self.tool]
