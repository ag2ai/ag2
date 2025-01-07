# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from typing import Optional

from autogen.oai.client import OpenAIWrapper


class MarkdownConverter(ABC):
    """
    Abstract base class for markdown converters.
    """

    @abstractmethod
    def convert(self, source: str) -> str:
        """
        Converts a source (local file path or URL) to markdown.

        Args:
            source (str): Path to a local file or a URL.

        Returns:
            str: The result of the conversion.
        """
        pass


class MarkItDownConverter(MarkdownConverter):
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
        from markitdown import MarkItDown

        md = MarkItDown(llm_client=self.mlm_client, llm_model=self.mlm_model)
        result = md.convert(source)
        return str(result.text_content)


class DoclingConverter(MarkdownConverter):
    """
    Markdown converter using the Docling library.
    """

    def convert(self, source: str) -> str:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(source)
        return str(result.document.export_to_markdown())
