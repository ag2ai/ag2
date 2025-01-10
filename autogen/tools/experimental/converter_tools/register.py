# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, Optional

from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.tools import Tool

from .... import register_function
from .md_converter import (
    DoclingConverter,
    MarkItDownConverter,
    MarkdownConverter,
)


def create_converter(
    converter_type: str = "markitdown",
) -> Callable[[str], str]:
    """
    Creates a markdown converter function based on the specified type.

    Args:
        converter_type (str): Type of converter ("markitdown", "markitdown-llm", or "docling")
        mlm_client (Optional[OpenAIWrapper]): Multimodal LLM client (only for markitdown-llm)
        mlm_model (Optional[str]): Multimodal LLM model (only for markitdown-llm)

    Returns:
        Callable[[str], str]: Converter function that takes a source and returns markdown
    """

    def converter_func(source: str) -> str:
        if converter_type == "markitdown":
            return MarkItDownConverter().convert(source)
        elif converter_type == "docling":
            return DoclingConverter().convert(source)
        else:
            raise ValueError(f"Unsupported converter type: {converter_type}")

    return converter_func


markdown_convert_tool = Tool(
    name="markdown_convert",
    description="Converts a local file or a URL to markdown.",
    func_or_tool=create_converter("markitdown"),
)


def register_converter(
    caller: ConversableAgent,
    executor: ConversableAgent,
    converter: str = "markitdown",
    name: str = "markdown_convert",
    description: str = "Converts a local file or a URL to markdown. Including images with text.",
    args: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Registers a markdown converter tool with the specified caller and executor agents.

    Args:
        caller (ConversableAgent): The agent that will propose the conversion.
        executor (ConversableAgent): The agent that will execute the conversion.
        converter (str): The converter to use ("markitdown" or "docling"). Defaults to "markitdown".
        name (str): The name of the tool. Defaults to "markdown_convert".
        description (str): The description of the tool. Defaults to "Converts a local file or a URL to markdown.".
        args (Optional[Dict[str, Any]]): Optional arguments for the converter.
    """
    args = args or {}

    if converter == "markitdown":
        convert_func = create_converter("markitdown")
    elif converter == "docling":
        convert_func = create_converter("docling")
    else:
        raise ValueError(
            f"Unsupported converter: {converter}. Supported converters are 'markitdown','markitdown-llm' and 'docling'"
        )

    register_function(
        convert_func,
        caller=caller,
        executor=executor,
        name=name,
        description=description,
    )
