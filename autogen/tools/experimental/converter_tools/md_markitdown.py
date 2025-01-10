# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from markitdown import MarkItDown

from autogen.oai.client import OpenAIWrapper

from .md_converter_protocol import MarkdownConverterProtocol


class MarkItDownConverter(MarkdownConverterProtocol):
    """
    Markdown converter using the MarkItDown library.
    """

    def __init__(
        self,
        mlm_client: Optional[OpenAIWrapper] = None,
        mlm_model: Optional[str] = None,
    ):
        self.mlm_client = mlm_client
        self.mlm_model = mlm_model

    def convert(self, source: str) -> str:
        md = MarkItDown(llm_client=self.mlm_client, llm_model=self.mlm_model)
        result = md.convert(source)
        return str(result.text_content)
