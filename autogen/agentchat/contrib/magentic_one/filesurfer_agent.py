# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

from typing import Callable, Dict, Literal, Optional, Union

from autogen.agentchat import ConversableAgent


def create_file_surfer(
    name: str = "file_surfer",
    system_message: str = "You are a helpful AI Assistant that can navigate and read local files.",
    llm_config: Optional[Union[Dict, Literal[False]]] = None,
) -> ConversableAgent:
    """Create a ConversableAgent configured for file surfing capabilities."""

    agent = ConversableAgent(
        name=name,
        system_message=system_message,
        human_input_mode="NEVER",
        code_execution_config=False,
        llm_config=llm_config,
    )

    def open_local_file(path: str) -> Dict:
        """Open a local file at a path in the text-based browser."""
        pass

    def page_up() -> Dict:
        """Scroll the viewport UP one page-length."""
        pass

    def page_down() -> Dict:
        """Scroll the viewport DOWN one page-length."""
        pass

    def find_on_page(search_string: str) -> Dict:
        """Find first occurrence of search string on page."""
        pass

    def find_next() -> Dict:
        """Find next occurrence of search string."""
        pass

    def get_browser_state() -> Dict:
        """Get current browser state including header and content."""
        pass

    agent.register_function(
        {
            "open_local_file": open_local_file,
            "page_up": page_up,
            "page_down": page_down,
            "find_on_page": find_on_page,
            "find_next": find_next,
            "get_browser_state": get_browser_state,
        }
    )

    if llm_config:
        for schema in FILE_TOOL_SCHEMA:
            agent.update_tool_signature(schema, is_remove=False)

    return agent


FILE_TOOL_SCHEMA = []
