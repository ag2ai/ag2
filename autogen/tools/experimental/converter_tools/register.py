# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, Optional, Type

from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.tools import Tool

from .... import register_function
from .md_converter_protocol import MarkdownConverterProtocol

# from .md_docling import DoclingConverter
from .md_markitdown import MarkItDownConverter


def create_converter(converter: Type[MarkdownConverterProtocol] = MarkItDownConverter) -> Callable[..., Any]:
    def converter_func(source: str) -> str:
        return converter().convert(source)

    return converter_func


def register_converter(
    *,
    caller: ConversableAgent,
    executor: ConversableAgent,
    converter: Type[MarkdownConverterProtocol] = MarkItDownConverter,
    name: str = "markdown_convert",
    description: str = "Converts a local file or a URL to markdown. Including images with text.",
    args: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Registers a markdown converter tool with the specified caller and executor agents.

    Args:
        caller: The agent that will propose the conversion.
        executor: The agent that will execute the conversion.
        converter: The converter class implementing MarkdownConverterProtocol to use.
                   Defaults to MarkItDownConverter.
        name: The name of the tool. Defaults to "markdown_convert".
        description: The description of the tool.
                     Defaults to "Converts a local file or a URL to markdown. Including images with text.".
        args: Optional additional arguments for the converter (currently unused).

    Note:
        The converter must implement the MarkdownConverterProtocol interface with a convert() method.
        The registered function will take a source string (file path or URL) and return the converted markdown.
    """

    register_function(
        create_converter(converter),
        caller=caller,
        executor=executor,
        name=name,
        description=description,
    )
