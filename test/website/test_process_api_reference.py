# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from autogen.import_utils import optional_import_block, skip_on_missing_imports

with optional_import_block():
    # Add the ../../website directory to sys.path
    website_path = Path(__file__).resolve().parents[2] / "website"
    assert website_path.exists()
    assert website_path.is_dir()
    sys.path.append(str(website_path))

    from process_api_reference import generate_mint_json_from_template, move_files_excluding_index


@pytest.fixture
def api_dir(tmp_path: Path) -> Path:
    """Helper function to create test directory structure"""
    # Create autogen directory
    autogen_dir = tmp_path / "autogen"
    autogen_dir.mkdir()

    # Create some files in autogen
    (autogen_dir / "browser_utils.md").write_text("browser_utils")
    (autogen_dir / "code_utils.md").write_text("code_utils")
    (autogen_dir / "version.md").write_text("version")

    # Create subdirectories with files
    subdir1 = autogen_dir / "agentchat"
    subdir1.mkdir()
    (subdir1 / "assistant_agent.md").write_text("assistant_agent")
    (subdir1 / "conversable_agent.md").write_text("conversable_agent")
    (subdir1 / "agentchat.md").write_text("agentchat")
    (subdir1 / "index.md").write_text("index")

    # nested subdirectory
    nested_subdir = subdir1 / "contrib"
    nested_subdir.mkdir()
    (nested_subdir / "agent_builder.md").write_text("agent_builder")
    (nested_subdir / "index.md").write_text("index")

    subdir2 = autogen_dir / "cache"
    subdir2.mkdir()
    (subdir2 / "cache_factory.md").write_text("cache_factory")
    (subdir2 / "disk_cache.md").write_text("disk_cache")
    (subdir2 / "index.md").write_text("index")

    return tmp_path


@skip_on_missing_imports("jinja2", "docs")
def test_move_files_excluding_index(api_dir: Path) -> None:
    """Test that files are moved correctly excluding index.md"""
    # Call the function under test
    move_files_excluding_index(api_dir)

    # Verify that autogen directory no longer exists
    assert not (api_dir / "autogen").exists()

    # Verify the version.md file was not moved
    assert not (api_dir / "version.md").exists()

    # Verify files were moved correctly
    assert (api_dir / "browser_utils.md").exists()
    assert (api_dir / "code_utils.md").exists()
    assert (api_dir / "agentchat" / "assistant_agent.md").exists()
    assert (api_dir / "agentchat" / "conversable_agent.md").exists()
    assert (api_dir / "agentchat" / "agentchat.md").exists()
    assert (api_dir / "agentchat" / "contrib" / "agent_builder.md").exists()
    assert (api_dir / "cache" / "cache_factory.md").exists()
    assert (api_dir / "cache" / "disk_cache.md").exists()

    # Verify index.md was not moved
    assert not (api_dir / "agentchat" / "index.md").exists()
    assert not (api_dir / "agentchat" / "contrib" / "index.md").exists()
    assert not (api_dir / "cache" / "index.md").exists()


@pytest.fixture
def template_content() -> str:
    """Fixture providing the template JSON content."""
    template = """
    {
        "name": "AG2",
        "logo": {"dark": "/logo/ag2-white.svg", "light": "/logo/ag2.svg"},
        "navigation": [
            {"group": "", "pages": ["docs/Home", "docs/Getting-Started"]},
            {
                "group": "Installation",
                "pages": [
                    "docs/installation/Installation",
                    "docs/installation/Docker",
                    "docs/installation/Optional-Dependencies"
                ]
            },
            {"group": "API Reference", "pages": ["PLACEHOLDER"]}
        ]
    }
    """
    return template


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Fixture providing a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def template_file(temp_dir: Path, template_content: str) -> Path:
    """Fixture creating a template file in a temporary directory."""
    template_path = temp_dir / "mint-json-template.json.jinja"
    with open(template_path, "w") as f:
        f.write(template_content)
    return template_path


@pytest.fixture
def target_file(temp_dir: Path) -> Path:
    """Fixture providing the target mint.json path."""
    return temp_dir / "mint.json"


@skip_on_missing_imports("jinja2", "docs")
def test_generate_mint_json_from_template(template_file: Path, target_file: Path, template_content: str) -> None:
    """Test that mint.json is generated correctly from template."""
    # Run the function
    generate_mint_json_from_template(template_file, target_file)

    # Verify the file exists
    assert target_file.exists()

    # Verify the contents
    with open(target_file) as f:
        actual = json.load(f)

    expected = json.loads(template_content)
    assert actual == expected


@skip_on_missing_imports("jinja2", "docs")
def test_generate_mint_json_existing_file(template_file: Path, target_file: Path, template_content: str) -> None:
    """Test that function works when mint.json already exists."""
    # Create an existing mint.json with different content
    existing_content = {"name": "existing"}
    with open(target_file, "w") as f:
        json.dump(existing_content, f)

    # Run the function
    generate_mint_json_from_template(template_file, target_file)

    # Verify the contents were overwritten
    with open(target_file) as f:
        actual = json.load(f)

    expected = json.loads(template_content)
    assert actual == expected


@skip_on_missing_imports("jinja2", "docs")
def test_generate_mint_json_missing_template(target_file: Path) -> None:
    """Test handling of missing template file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        nonexistent_template = Path(tmp_dir) / "nonexistent.template"
        with pytest.raises(FileNotFoundError):
            generate_mint_json_from_template(nonexistent_template, target_file)
