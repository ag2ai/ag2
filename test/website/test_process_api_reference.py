# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Add the ../../website directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "website"))
from process_api_reference import generate_mint_json_from_template


@pytest.fixture
def template_content():
    """Fixture providing the template JSON content."""
    return {
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


@pytest.fixture
def temp_dir():
    """Fixture providing a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def template_file(temp_dir, template_content):
    """Fixture creating a template file in a temporary directory."""
    template_path = temp_dir / "mint-json-template.json"
    with open(template_path, "w") as f:
        json.dump(template_content, f, indent=2)
    return template_path


@pytest.fixture
def target_file(temp_dir):
    """Fixture providing the target mint.json path."""
    return temp_dir / "mint.json"


def test_generate_mint_json_from_template(template_file, target_file, template_content):
    """Test that mint.json is generated correctly from template."""
    # Run the function
    generate_mint_json_from_template(template_file, target_file)

    # Verify the file exists
    assert target_file.exists()

    # Verify the contents
    with open(target_file) as f:
        generated_content = json.load(f)

    assert generated_content == template_content


def test_generate_mint_json_existing_file(template_file, target_file, template_content):
    """Test that function works when mint.json already exists."""
    # Create an existing mint.json with different content
    existing_content = {"name": "existing"}
    with open(target_file, "w") as f:
        json.dump(existing_content, f)

    # Run the function
    generate_mint_json_from_template(template_file, target_file)

    # Verify the contents were overwritten
    with open(target_file) as f:
        generated_content = json.load(f)

    assert generated_content == template_content


def test_generate_mint_json_missing_template(target_file):
    """Test handling of missing template file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        nonexistent_template = Path(tmp_dir) / "nonexistent.template"
        with pytest.raises(FileNotFoundError):
            generate_mint_json_from_template(nonexistent_template, target_file)
