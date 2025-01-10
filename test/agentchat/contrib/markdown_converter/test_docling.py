# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
import io
import logging
import os
import shutil
import sys

import pytest

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

from autogen.tools.experimental.converter_tools.register import create_converter

from .test_config import (
    DOCX_TEST_STRINGS,
    JPG_TEST_EXIFTOOL,
    PDF_TEST_STRINGS,
    PDF_TEST_URL,
    PPTX_TEST_STRINGS,
    SERP_TEST_EXCLUDES,
    SERP_TEST_STRINGS,
    SERP_TEST_URL,
    TEST_FILES_DIR,
    WIKIPEDIA_TEST_EXCLUDES,
    WIKIPEDIA_TEST_STRINGS,
    WIKIPEDIA_TEST_URL,
    XLSX_TEST_STRINGS,
)


def test_mdconvert_local_docling() -> None:
    converter = create_converter("docling")
    # Test XLSX processing

    result = converter(os.path.join(TEST_FILES_DIR, "test.xlsx"))
    for test_string in XLSX_TEST_STRINGS:
        text_content = result.replace("\\", "")
        assert test_string in text_content

    # Test DOCX processing
    result = converter(os.path.join(TEST_FILES_DIR, "test.docx"))
    for test_string in DOCX_TEST_STRINGS:
        text_content = result.replace("\\", "")
        assert test_string in text_content

    # Test PPTX processing
    result = converter(os.path.join(TEST_FILES_DIR, "test.pptx"))
    for test_string in PPTX_TEST_STRINGS:
        text_content = result.replace("\\", "")
        assert test_string in text_content

    # Test Wikipedia processing
    result = converter(WIKIPEDIA_TEST_URL)
    logger.debug(f"Wikipedia conversion result: {result}")
    text_content = result.replace("\\", "")
    for test_string in WIKIPEDIA_TEST_EXCLUDES:
        if test_string in text_content:
            logger.error(f"Unexpectedly found '{test_string}' in:\n{text_content}")
        assert test_string not in text_content
    for test_string in WIKIPEDIA_TEST_STRINGS:
        if test_string not in text_content:
            logger.error(f"Failed to find '{test_string}' in:\n{text_content}")
        assert test_string in text_content

    # Test Bing processing
    result = converter(SERP_TEST_URL)
    logger.debug(f"SERP conversion result: {result}")
    text_content = result.replace("\\", "")
    for test_string in SERP_TEST_EXCLUDES:
        if test_string in text_content:
            logger.error(f"Unexpectedly found '{test_string}' in:\n{text_content}")
        assert test_string not in text_content
    for test_string in SERP_TEST_STRINGS:
        if test_string not in text_content:
            logger.error(f"Failed to find '{test_string}' in:\n{text_content}")
        assert test_string in text_content


def test_mdconvert_exiftool_markitdown() -> None:
    converter = create_converter()
    # Test JPG metadata processing
    result = converter(os.path.join(TEST_FILES_DIR, "test_image.jpg"))
    logger.debug(f"Image metadata result: {result}")
    for key in JPG_TEST_EXIFTOOL:
        target = f"{key}: {JPG_TEST_EXIFTOOL[key]}"
        if target not in result:
            logger.error(f"Failed to find '{target}' in:\n{result}")
        assert target in result
