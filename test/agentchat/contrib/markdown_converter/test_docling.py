# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
import logging
import sys
from typing import List

import pytest

from autogen.tools.experimental.converter_tools.register import create_converter

# from autogen.tools.experimental.converter_tools.md_docling import DoclingConverter


@pytest.fixture(scope="module")
def converter():
    return create_converter()  # TODO: add DoclingConverter


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class TestDoclingConverter:
    def validate_content(self, content: str, test_strings: List[str], excludes: List[str] = None) -> None:
        """Helper method to validate content against test strings and excludes"""
        text_content = content.replace("\\", "")

        if excludes:
            for exclude in excludes:
                if exclude in text_content:
                    logger.error(f"Unexpectedly found '{exclude}' in:\n{text_content}")
                assert exclude not in text_content

        for test_string in test_strings:
            if test_string not in text_content:
                logger.error(f"Failed to find '{test_string}' in:\n{text_content}")
            assert test_string in text_content

    def test_pdf_conversion(self, converter, pdf_test_data):
        result = converter(pdf_test_data["file_path"])
        self.validate_content(result, pdf_test_data["test_strings"])

    def test_xlsx_conversion(self, converter, xlsx_test_data):
        result = converter(xlsx_test_data["file_path"])
        self.validate_content(result, xlsx_test_data["test_strings"])

    def test_docx_conversion(self, converter, docx_test_data):
        result = converter(docx_test_data["file_path"])
        self.validate_content(result, docx_test_data["test_strings"])

    def test_pptx_conversion(self, converter, pptx_test_data):
        result = converter(pptx_test_data["file_path"])
        self.validate_content(result, pptx_test_data["test_strings"])

    def test_wikipedia_conversion(self, converter, wikipedia_test_data):
        result = converter(wikipedia_test_data["file_path"])
        self.validate_content(result, wikipedia_test_data["test_strings"], wikipedia_test_data["excludes"])

    def test_serp_conversion(self, converter, serp_test_data):
        result = converter(serp_test_data["file_path"])
        self.validate_content(result, serp_test_data["test_strings"], serp_test_data["excludes"])

    """
    #Image not supported yet
    def test_jpg_conversion(self, converter, jpg_test_data):
        result = converter(jpg_test_data["file_path"])

        # Validate metadata
        for key, value in jpg_test_data["test_metadata"].items():
            target = f"{key}: {value}"
            assert target in result, f"Failed to find '{target}' in:\n{result}"
    """
