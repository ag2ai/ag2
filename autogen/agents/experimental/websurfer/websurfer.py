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
        *args,
        web_tool: Literal["browser-use", "crawl4ai"] = "browser-use",
        web_tool_kwargs: dict[str, Any] = None,
        **kwargs,
    ) -> None:
        if web_tool == "browser-use":
            self.tool: Tool = BrowserUseTool(**(web_tool_kwargs if web_tool_kwargs else {}))
        elif web_tool == "crawl4ai":
            self.tool = Crawl4AITool(**(web_tool_kwargs if web_tool_kwargs else {}))
        else:
            raise ValueError(f"Unsupported {web_tool=}.")

        super().__init__(*args, **kwargs)

        self.register_for_llm(self.tool)

    @property
    def tools(self) -> list[Tool]:
        return [self.tool]
