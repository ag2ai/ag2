# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add the ../../website directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "website"))
from process_notebooks import (
    ensure_mint_json_exists,
    extract_example_group,
    generate_nav_group,
    get_sorted_files,
)


def test_ensure_mint_json():
    # Test with empty temp directory - should raise SystemExit
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with pytest.raises(SystemExit):
            ensure_mint_json_exists(tmp_path)

        # Now create mint.json - should not raise error
        (tmp_path / "mint.json").touch()
        ensure_mint_json_exists(tmp_path)  # Should not raise any exception


class TestAddBlogsToNavigation:
    @pytest.fixture
    def test_dir(self):
        """Create a temporary directory with test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create test directory structure
            files = [
                "2023-04-21-LLM-tuning-math/index.mdx",
                "2023-05-18-GPT-adaptive-humaneval/index.mdx",
                "2023-10-26-TeachableAgent/index.mdx",
                "2023-10-18-RetrieveChat/index.mdx",
                "2024-12-20-RetrieveChat/index.mdx",
                "2024-12-20-Tools-interoperability/index.mdx",
                "2024-12-20-RealtimeAgent/index.mdx",
                "2024-08-26/index.mdx",
                "2024-11-11/index.mdx",
                "2024-11-12/index.mdx",
                "2024-12-20-Another-RealtimeAgent/index.mdx",
            ]

            for file_path in files:
                full_path = tmp_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.touch()

            yield tmp_path

    @pytest.fixture
    def expected(self):
        return [
            "blog/2024-12-20-Tools-interoperability/index",
            "blog/2024-12-20-RetrieveChat/index",
            "blog/2024-12-20-RealtimeAgent/index",
            "blog/2024-12-20-Another-RealtimeAgent/index",
            "blog/2024-11-12/index",
            "blog/2024-11-11/index",
            "blog/2024-08-26/index",
            "blog/2023-10-26-TeachableAgent/index",
            "blog/2023-10-18-RetrieveChat/index",
            "blog/2023-05-18-GPT-adaptive-humaneval/index",
            "blog/2023-04-21-LLM-tuning-math/index",
        ]

    def test_get_sorted_files(self, test_dir, expected):
        actual = get_sorted_files(test_dir, "blog")
        assert actual == expected, actual

    def test_add_blogs_to_navigation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            website_dir = Path(tmp_dir)
            blog_dir = website_dir / "blog"

            # Create test blog structure
            test_files = [
                "2023-04-21-LLM-tuning-math/index.mdx",
                "2024-12-20-RealtimeAgent/index.mdx",
                "2024-12-20-RetrieveChat/index.mdx",
            ]

            for file_path in test_files:
                full_path = blog_dir / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.touch()

            # Expected result after processing
            expected = {
                "group": "Blog",
                "pages": [
                    "blog/2024-12-20-RetrieveChat/index",
                    "blog/2024-12-20-RealtimeAgent/index",
                    "blog/2023-04-21-LLM-tuning-math/index",
                ],
            }

            # Run function and check result
            actual = generate_nav_group(blog_dir, "Blog", "blog")
            assert actual == expected, actual

            # Expected result after processing
            expected = {
                "group": "Talks",
                "pages": [
                    "talks/2024-12-20-RetrieveChat/index",
                    "talks/2024-12-20-RealtimeAgent/index",
                    "talks/2023-04-21-LLM-tuning-math/index",
                ],
            }

            # Run function and check result
            actual = generate_nav_group(blog_dir, "Talks", "talks")
            assert actual == expected, actual


class TestUpdateNavigation:
    def setup(self, temp_dir: Path) -> None:
        """Set up test files in the temporary directory."""

        # Create directories
        snippets_dir = temp_dir / "snippets" / "data"
        snippets_dir.mkdir(parents=True, exist_ok=True)

        # Create mint.json content
        mint_json_content = {
            "name": "AG2",
            "logo": {"dark": "/logo/ag2-white.svg", "light": "/logo/ag2.svg"},
            "navigation": [
                {"group": "", "pages": ["docs/Home", "docs/Getting-Started"]},
                {
                    "group": "Installation",
                    "pages": [
                        "docs/installation/Installation",
                        "docs/installation/Docker",
                        "docs/installation/Optional-Dependencies",
                    ],
                },
                {"group": "API Reference", "pages": ["PLACEHOLDER"]},
                {
                    "group": "AutoGen Studio",
                    "pages": [
                        "docs/autogen-studio/getting-started",
                        "docs/autogen-studio/usage",
                        "docs/autogen-studio/faqs",
                    ],
                },
            ],
        }

        # Create NotebooksMetadata.mdx content
        notebooks_metadata_content = """{/*
    Auto-generated file - DO NOT EDIT
    Please edit the add_front_matter_to_metadata_mdx function in process_notebooks.py
    */}

    export const notebooksMetadata = [
        {
            "title": "Using RetrieveChat Powered by MongoDB Atlas for Retrieve Augmented Code Generation and Question Answering",
            "link": "/notebooks/agentchat_RetrieveChat_mongodb",
            "description": "Explore the use of AutoGen's RetrieveChat for tasks like code generation from docstrings, answering complex questions with human feedback, and exploiting features like Update Context, custom prompts, and few-shot learning.",
            "image": null,
            "tags": [
                "MongoDB",
                "integration",
                "RAG"
            ],
            "source": "/notebook/agentchat_RetrieveChat_mongodb.ipynb"
        },
        {
            "title": "Mitigating Prompt hacking with JSON Mode in Autogen",
            "link": "/notebooks/JSON_mode_example",
            "description": "Use JSON mode and Agent Descriptions to mitigate prompt manipulation and control speaker transition.",
            "image": null,
            "tags": [
                "JSON",
                "description",
                "prompt hacking",
                "group chat",
                "orchestration"
            ],
            "source": "/notebook/JSON_mode_example.ipynb"
        }];"""

        # Write files
        mint_json_path = temp_dir / "mint.json"
        with open(mint_json_path, "w", encoding="utf-8") as f:
            json.dump(mint_json_content, f, indent=2)
            f.write("\n")

        metadata_path = snippets_dir / "NotebooksMetadata.mdx"
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(notebooks_metadata_content)

    def test_extract_example_group(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            self.setup(tmp_path)

            # Run the function
            metadata_path = tmp_path / "snippets" / "data" / "NotebooksMetadata.mdx"
            actual = extract_example_group(metadata_path)

            expected = {
                "group": "Examples",
                "pages": [
                    "notebooks/Examples",
                    {
                        "group": "Examples by Notebook",
                        "pages": [
                            "notebooks/Notebooks",
                            "notebooks/agentchat_RetrieveChat_mongodb",
                            "notebooks/JSON_mode_example",
                        ],
                    },
                    "notebooks/Gallery",
                ],
            }

            assert actual == expected, actual
