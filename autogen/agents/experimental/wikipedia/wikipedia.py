# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Optional, Union

from .... import ConversableAgent
from ....doc_utils import export_module
from ....tools.experimental import WikipediaPageLoadTool, WikipediaQueryRunTool
from ..document_agent.parser_utils import logger


@export_module("autogen.agents.experimental")
class WikipediaAgent(ConversableAgent):
    """
    An AI agent that leverages Wikipedia tools to provide accurate and concise information
    in response to user queries.

    This agent utilizes two primary tools:
    - WikipediaQueryRunTool: For querying Wikipedia articles based on user input.
    - WikipediaPageLoadTool: For loading full Wikipedia pages for detailed information.

    The agent can be customized with specific system messages and formatting instructions
    to guide its responses.

    Attributes:
        _query_run_tool (WikipediaQueryRunTool): Tool for querying Wikipedia.
        _page_load_tool (WikipediaPageLoadTool): Tool for loading Wikipedia pages.
    """

    DEFAULT_SYSTEM_MESSAGE = """You are a knowledgeable AI assistant with access to Wikipedia.
    Use the your tools when necessary. Respond to user queries by providing accurate and concise information.
    If a question requires external data, utilize the appropriate tool to retrieve it.
    """

    def __init__(
        self,
        system_message: Optional[Union[str, list[str]]] = None,
        format_instructions: Optional[str] = None,
        language: str = "en",
        top_k: int = 2,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the WikipediaAgent with optional customization.

        Args:
            system_message (Optional[Union[str, list[str]]]): Custom system message(s) to guide the agent's behavior.
                If None, the DEFAULT_SYSTEM_MESSAGE is used.
            format_instructions (Optional[str]): Specific instructions on how the agent should format its responses.
            language (str): Language code for Wikipedia (default is "en" for English).
            top_k (int): Number of top search results to retrieve from Wikipedia (default is 2).
            **kwargs: Additional keyword arguments passed to the base ConversableAgent.
        """

        # Use provided system_message or default
        system_message = kwargs.pop("system_message", self.DEFAULT_SYSTEM_MESSAGE)

        if format_instructions is not None:
            instructions = f"\n\nFollow this format:\n\n{format_instructions}"
            if system_message is None:
                system_message = self.DEFAULT_SYSTEM_MESSAGE
            if isinstance(system_message, list):
                system_message.append(instructions)
            elif isinstance(system_message, str):
                system_message = system_message + instructions
            else:
                logger.error(f"system_message must be str or list[str], got {type(system_message).__name__}")

        # Initialize Wikipedia tools
        self._query_run_tool = WikipediaQueryRunTool(language=language, top_k=top_k)
        self._page_load_tool = WikipediaPageLoadTool(language=language, top_k=top_k)

        # Initialize the base ConversableAgent
        super().__init__(system_message=system_message, **kwargs)

        # Register tools for LLM recommendations
        self.register_for_llm()(self._query_run_tool)
        self.register_for_llm()(self._page_load_tool)
