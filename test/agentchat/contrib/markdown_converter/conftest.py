# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest


@pytest.fixture
def test_files_dir():
    return Path(__file__).parent / "test_files"


@pytest.fixture
def pdf_test_data():
    return {
        "file_path": "https://arxiv.org/pdf/2308.08155v2.pdf",
        "test_strings": ["While there is contemporaneous exploration of multi-agent approaches"],
    }


@pytest.fixture
def xlsx_test_data(test_files_dir):
    file_path = (test_files_dir / "test.xlsx").resolve()
    return {
        "file_path": str(file_path),
        "test_strings": [
            "## 09060124-b5e7-4717-9d07-3c046eb",
            "6ff4173b-42a5-4784-9b19-f49caff4d93d",
            "affc7dad-52dc-4b98-9b5d-51e65d8a8ad0",
        ],
    }


@pytest.fixture
def docx_test_data(test_files_dir):
    file_path = (test_files_dir / "test.docx").resolve()
    return {
        "file_path": str(file_path),
        "test_strings": [
            "314b0a30-5b04-470b-b9f7-eed2c2bec74a",
            "49e168b7-d2ae-407f-a055-2167576f39a1",
            "## d666f1f7-46cb-42bd-9a39-9a39cf2a509f",
            "# Abstract",
            "# Introduction",
            "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation",
        ],
    }


@pytest.fixture
def pptx_test_data(test_files_dir):
    file_path = (test_files_dir / "test.pptx").resolve()
    return {
        "file_path": str(file_path),
        "test_strings": [
            "2cdda5c8-e50e-4db4-b5f0-9722a649f455",
            "04191ea8-5c73-4215-a1d3-1cfb43aaaf12",
            "44bf7d06-5e7a-4a40-a2e1-a2e42ef28c8a",
            "1b92870d-e3b5-4e65-8153-919f4ff45592",
            "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation",
        ],
    }


@pytest.fixture
def wikipedia_test_data():
    return {
        "file_path": "https://en.wikipedia.org/wiki/Microsoft",
        "test_strings": [
            "Microsoft entered the operating system (OS) business in 1980 with its own version of [Unix]",
            'Microsoft was founded by [Bill Gates](/wiki/Bill_Gates "Bill Gates")',
        ],
        "excludes": [
            "You are encouraged to create an account and log in",
            "154 languages",
            "move to sidebar",
        ],
    }


@pytest.fixture
def serp_test_data():
    return {
        "file_path": "https://www.bing.com/search?q=microsoft+wikipedia",
        "test_strings": [
            "](https://en.wikipedia.org/wiki/Microsoft",
            "Microsoft Corporation is an American multinational technology conglomerate headquartered in Redmond, Washington.",
        ],
        "excludes": [
            "https://www.bing.com/ck/a?!&&p=",
            "data:image/svg+xml,%3Csvg%20width%3D",
        ],
    }


@pytest.fixture
def jpg_test_data(test_files_dir):
    file_path = (test_files_dir / "test_image.jpg").resolve()
    return {
        "file_path": str(file_path),
        "test_metadata": {
            "Title": "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation",
            "Description": "AutoGen enables diverse LLM-based applications",
            "ImageSize": "1615x1967",
            "DateTimeOriginal": "2024:03:14 22:10:00",
        },
    }
