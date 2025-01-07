# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

from typing import Annotated, Optional

from autogen.oai.client import OpenAIWrapper
from autogen.tools import Tool
from autogen.tools.md_converter import (
    MarkdownConverter,
    MarkItDownConverter,
    DoclingConverter,
)


def markdown_convert(
    source: Annotated[str, "Path to a local file or a URL"],
    converter: Annotated[
        str, "The converter to use: 'markitdown' or 'docling'"
    ] = "markitdown",
    mlm_client: Annotated[Optional[OpenAIWrapper], "Multimodal LLM client"] = None,
    mlm_model: Annotated[Optional[str], "Multimodal LLM model"] = None,
) -> str:
    """
    Converts a local file or a URL to markdown.

    Args:
        source (str): Path to a local file or a URL.
        converter (str): The converter to use: 'markitdown' or 'docling'. Defaults to 'markitdown'.
        mlm_client (OpenAIWrapper, optional): Multimodal LLM client. Defaults to None.
        mlm_model (str, optional): Multimodal LLM model. Defaults to None.

    Returns:
        str: The result of the conversion.
    """
    if converter == "markitdown":
        md_converter: MarkdownConverter = MarkItDownConverter(
            mlm_client=mlm_client, mlm_model=mlm_model
        )
    elif converter == "docling":
        md_converter = DoclingConverter()
    else:
        raise ValueError(f"Unknown converter: {converter}")

    return md_converter.convert(source)


markdown_convert_tool = Tool(
    name="markdown_convert",
    description="Converts a local file or a URL to markdown.",
    func_or_tool=markdown_convert,
)
