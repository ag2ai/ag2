# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from docling.document_converter import DocumentConverter

from .md_converter_protocol import MarkdownConvertProtocol


class DoclingConverter(MarkdownConvertProtocol):
    """
    Markdown converter using the Docling library.
    """

    def convert(self, source: str) -> str:
        converter = DocumentConverter()
        result = converter.convert(source)
        return str(result.document.export_to_markdown())
