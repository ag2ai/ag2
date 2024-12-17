# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

from typing import Dict, Literal, Optional, Union

from autogen.agentchat import ConversableAgent

from .markdown_browser import RequestsMarkdownBrowser


def create_file_surfer(
    name: str = "file_surfer",
    # TODO: adjust system massage
    system_message: str = "You are a helpful AI Assistant that can navigate and read local files.",
    llm_config: Optional[Union[Dict, Literal[False]]] = None,
    browser: Optional[RequestsMarkdownBrowser] = None,
) -> ConversableAgent:
    """Create a ConversableAgent configured for file surfing capabilities."""

    # Initialize browser
    browser = browser or RequestsMarkdownBrowser(viewport_size=1024 * 5, downloads_folder="coding")

    # Create agent
    agent = ConversableAgent(
        name=name,
        system_message=system_message,
        human_input_mode="NEVER",
        code_execution_config=False,
        llm_config=llm_config,
    )

    # Define file navigation functions
    def open_local_file(path: str) -> Dict:
        """Open a local file at a path in the text-based browser."""
        browser.open_local_file(path)
        return {"content": f"Opened file: {path}"}

    def page_up() -> Dict:
        """Scroll the viewport UP one page-length."""
        browser.page_up()
        return {"content": "Scrolled up one page"}

    def page_down() -> Dict:
        """Scroll the viewport DOWN one page-length."""
        browser.page_down()
        return {"content": "Scrolled down one page"}

    def find_on_page(search_string: str) -> Dict:
        """Find first occurrence of search string on page."""
        browser.find_on_page(search_string)
        return {"content": f"Found first occurrence of: {search_string}"}

    def find_next() -> Dict:
        """Find next occurrence of search string."""
        browser.find_next()
        return {"content": "Found next occurrence"}

    def get_browser_state() -> Dict:
        """Get current browser state including header and content."""
        header = f"Address: {browser.address}\n"

        if browser.page_title:
            header += f"Title: {browser.page_title}\n"

        current_page = browser.viewport_current_page
        total_pages = len(browser.viewport_pages)
        header += f"Viewport position: Showing page {current_page+1} of {total_pages}.\n"

        content = browser.viewport
        return {"content": header.strip() + "\n=======================\n" + content}

    # Register functions with agent
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

    # Register function schemas if LLM config is provided
    if llm_config:
        for schema in FILE_TOOL_SCHEMA:
            agent.update_tool_signature(schema, is_remove=False)

    return agent


# Tool schemas for LLM
FILE_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "open_local_file",
            "description": "Open a local file at a path in the text-based browser",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The relative or absolute path of a local file to visit"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "page_up",
            "description": "Scroll the viewport UP one page-length",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "page_down",
            "description": "Scroll the viewport DOWN one page-length",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_on_page",
            "description": "Find first occurrence of search string on page",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_string": {"type": "string", "description": "The string to search for on the page"}
                },
                "required": ["search_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_next",
            "description": "Find next occurrence of search string",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_browser_state",
            "description": "Get current browser state including header and content",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
