# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol


class MarkdownConverterProtocol(Protocol):
    """
    Protocol for markdown converters.
    """

    def convert(self, source: str) -> str:
        """
        Converts a source (local file path or URL) to markdown.

        Args:
            source (str): Path to a local file or a URL.

        Returns:
            str: The result of the conversion.
        """
        ...
