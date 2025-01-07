# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

from typing import Annotated, Callable, Optional

from autogen.oai.client import OpenAIWrapper
from autogen.tools import Tool
from autogen.tools.dependency_injection import Depends
from autogen.tools.md_converter import (
    DoclingConverter,
    MarkdownConverter,
    MarkItDownConverter,
)


def markdown_convert(
    source: Annotated[str, "Path to a local file or a URL"],
    converter: Annotated[MarkdownConverter, "The converter to use"] = Depends(lambda: MarkItDownConverter()),
    mlm_client: Annotated[Optional[OpenAIWrapper], "Multimodal LLM client"] = Depends(lambda: None),
    mlm_model: Annotated[Optional[str], "Multimodal LLM model"] = Depends(lambda: None),
) -> str:
    """
    Converts a local file or a URL to markdown.

    Args:
        source (str): Path to a local file or a URL.
        converter (MarkdownConverter): The converter to use.
        mlm_client (OpenAIWrapper, optional): Multimodal LLM client. Defaults to None.
        mlm_model (str, optional): Multimodal LLM model. Defaults to None.

    Returns:
        str: The result of the conversion.
    """

    if hasattr(converter, "dependency"):
        converter = converter.dependency()
    elif callable(converter):
        converter = converter()

    if isinstance(converter, MarkItDownConverter):
        converter.mlm_client = mlm_client
        converter.mlm_model = mlm_model
    return converter.convert(source)


def create_markitdown_converter(
    mlm_client: Optional[OpenAIWrapper] = None, mlm_model: Optional[str] = None
) -> Callable[[str], str]:
    """Creates a markdown_convert function with MarkItDownConverter."""

    def _markdown_convert(
        source: Annotated[str, "Path to a local file or a URL"],
        converter: Annotated[MarkdownConverter, "The converter to use"] = Depends(lambda: MarkItDownConverter()),
        mlm_client: Annotated[Optional[OpenAIWrapper], "Multimodal LLM client"] = Depends(lambda: mlm_client),
        mlm_model: Annotated[Optional[str], "Multimodal LLM model"] = Depends(lambda: mlm_model),
    ) -> str:
        return markdown_convert(source, converter, mlm_client, mlm_model)

    return _markdown_convert


def create_docling_converter() -> Callable[[str], str]:
    """Creates a markdown_convert function with DoclingConverter."""

    def _markdown_convert(
        source: Annotated[str, "Path to a local file or a URL"],
        converter: Annotated[MarkdownConverter, "The converter to use"] = Depends(lambda: DoclingConverter()),
        mlm_client: Annotated[Optional[OpenAIWrapper], "Multimodal LLM client"] = Depends(lambda: None),
        mlm_model: Annotated[Optional[str], "Multimodal LLM model"] = Depends(lambda: None),
    ) -> str:
        return markdown_convert(source, converter, mlm_client, mlm_model)

    return _markdown_convert


markdown_convert_tool = Tool(
    name="markdown_convert",
    description="Converts a local file or a URL to markdown.",
    func_or_tool=markdown_convert,
)
