# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, Optional, Union

from .... import ConversableAgent
from ....doc_utils import export_module
from ....tools.experimental import WikipediaPageLoadTool, WikipediaQueryRunTool


@export_module("autogen.agents.experimental")
class WikipediaAgent(ConversableAgent):
    """
    An AI agent that leverages Wikipedia tools to provide accurate, concise answers
    to user queries.

    Tools:
        - WikipediaQueryRunTool: searches Wikipedia for relevant article titles.
        - WikipediaPageLoadTool: loads full Wikipedia pages (and metadata) for in‑depth content.

    Attributes:
        _query_run_tool (WikipediaQueryRunTool): for running title/keyword searches.
        _page_load_tool (WikipediaPageLoadTool): for fetching full page content.

    Parameters:
        system_message (Optional[Union[str, List[str]]]):
            Custom system prompt(s). If None, DEFAULT_SYSTEM_MESSAGE is used.
            Must be a str or a list of strings, otherwise raises ValueError.
        format_instructions (Optional[str]): Extra formatting instructions to append.
        language (str): ISO code for the Wikipedia language edition (default: "en").
        top_k (int): Number of top search results to return (default: 2).
        **kwargs: Passed through to the base ConversableAgent.

    Raises:
        ValueError: If `system_message` is not a str or list of str.
    """

    DEFAULT_SYSTEM_MESSAGE = (
        "You are a knowledgeable AI assistant with access to Wikipedia.\n"
        "Use your tools when necessary. Respond to user queries by providing accurate and concise information.\n"
        "If a question requires external data, utilize the appropriate tool to retrieve it."
    )

    def __init__(
        self,
        system_message: Optional[Union[str, List[str]]] = None,
        format_instructions: Optional[str] = None,
        language: str = "en",
        top_k: int = 2,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the WikipediaAgent with optional custom prompts and tools.

        Args:
            system_message (Optional[Union[str, List[str]]]):
                Custom system prompt(s). If None, DEFAULT_SYSTEM_MESSAGE is used.
                Must be a str or list of strings.
            format_instructions (Optional[str]): Extra formatting instructions to append.
            language (str): Wikipedia language code (default: "en").
            top_k (int): How many top search results to fetch (default: 2).
            **kwargs: Other parameters for ConversableAgent.

        Raises:
            ValueError: If `system_message` is not a str or list of str.
        """
        # Use explicit system_message or fall back to default
        system_message = system_message or self.DEFAULT_SYSTEM_MESSAGE

        # Append formatting instructions if provided
        if format_instructions is not None:
            instructions = f"\n\nFollow this format:\n\n{format_instructions}"
            if isinstance(system_message, list):
                system_message.append(instructions)
            elif isinstance(system_message, str):
                system_message = system_message + instructions
            else:
                raise ValueError(f"system_message must be str or list[str], got {type(system_message).__name__}")

        # Initialize Wikipedia tools
        self._query_run_tool = WikipediaQueryRunTool(language=language, top_k=top_k)
        self._page_load_tool = WikipediaPageLoadTool(language=language, top_k=top_k)

        # Initialize the base ConversableAgent
        super().__init__(system_message=system_message, **kwargs)

        # Register tools for LLM recommendations
        self.register_for_llm()(self._query_run_tool)
        self.register_for_llm()(self._page_load_tool)
