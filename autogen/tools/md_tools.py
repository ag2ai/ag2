# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

from typing import Annotated, Optional

from markitdown import MarkItDown

from autogen.oai.client import OpenAIWrapper
from autogen.tools import Tool


def markdown_convert(
    source: Annotated[str, "Path to a local file or a URL"],
    # mlm_client: Annotated[Optional[OpenAIWrapper], "Multimodal LLM client"] = None,
    mlm_model: Annotated[Optional[str], "Multimodal LLM model"] = None,
) -> str:
    """
    Converts a local file or a URL to markdown.

    Args:
        source (str): Path to a local file or a URL.
        mlm_client (OpenAIWrapper, optional): Multimodal LLM client. Defaults to None.
        mlm_model (str, optional): Multimodal LLM model. Defaults to None.

    Returns:
        str: The result of the conversion.
    """
    mlm_client = None
    md = MarkItDown(llm_client=mlm_client, llm_model=mlm_model)
    result = md.convert(source)
    return str(result.text_content)


markdown_convert_tool = Tool(
    name="markdown_convert",
    description="Converts a local file or a URL to markdown.",
    func_or_tool=markdown_convert,
)
