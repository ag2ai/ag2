# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from jinja2 import Template


def move_files_excluding_index(api_dir: Path) -> None:
    """Move files from api_dir/autogen to api_dir, excluding index.md files.

    Args:
        api_dir (Path): Path to the API directory
    """
    autogen_dir = api_dir / "autogen"
    for file_path in autogen_dir.rglob("*"):
        if file_path.is_file() and file_path.name != "index.md" and file_path.name != "version.md":
            dest = api_dir / file_path.relative_to(autogen_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file_path), str(dest))
    shutil.rmtree(autogen_dir)


def run_pdoc3(api_dir: Path) -> None:
    """Run pydoc3 to generate the API documentation."""
    try:
        print(f"Generating API documentation and saving to {str(api_dir)}...")
        subprocess.run(
            ["pdoc", "--output-dir", str(api_dir), "--template-dir", "mako_templates", "--force", "autogen"],
            check=True,
            capture_output=True,
            text=True,
        )

        # the generated files are saved in a directory named '{api_dir}/autogen'. move all files to the parent directory
        move_files_excluding_index(api_dir)

        print("Successfully generated API documentation")
    except subprocess.CalledProcessError as e:
        print(f"Error running pdoc3: {e.stderr}")
        sys.exit(1)


def read_file_content(file_path: Path) -> str:
    """Read content from a file.

    Args:
        file_path (str): Path to the file

    Returns:
        str: Content of the file
    """
    with open(file_path, encoding="utf-8") as f:
        return f.read()


def write_file_content(file_path: str, content: str) -> None:
    """Write content to a file.

    Args:
        file_path (str): Path to the file
        content (str): Content to write
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def convert_md_to_mdx(input_dir: Path) -> None:
    """Convert all .md files in directory to .mdx while preserving structure.

    Args:
        input_dir (Path): Directory containing .md files to convert
    """
    if not input_dir.exists():
        print(f"Directory not found: {input_dir}")
        sys.exit(1)

    for md_file in input_dir.rglob("*.md"):
        mdx_file = md_file.with_suffix(".mdx")

        # Read content from .md file
        content = md_file.read_text(encoding="utf-8")

        # Write content to .mdx file
        mdx_file.write_text(content, encoding="utf-8")

        # Remove original .md file
        md_file.unlink()
        print(f"Converted: {md_file} -> {mdx_file}")


def get_mdx_files(directory: Path) -> list[str]:
    """Get all MDX files in directory and subdirectories."""
    return [f"{p.relative_to(directory).with_suffix('')!s}".replace("\\", "/") for p in directory.rglob("*.mdx")]


def add_prefix(path: str, parent_groups: Optional[list[str]] = None) -> str:
    """Create full path with prefix and parent groups."""
    groups = parent_groups or []
    return f"docs/reference/{'/'.join(groups + [path])}"


def create_nav_structure(paths: list[str], parent_groups: Optional[list[str]] = None) -> list[Any]:
    """Convert list of file paths into nested navigation structure."""
    groups: dict[str, list[str]] = {}
    pages = []
    parent_groups = parent_groups or []

    for path in paths:
        parts = path.split("/")
        if len(parts) == 1:
            pages.append(add_prefix(path, parent_groups))
        else:
            group = parts[0]
            subpath = "/".join(parts[1:])
            groups.setdefault(group, []).append(subpath)

    # Sort directories and create their structures
    sorted_groups = [
        {
            "group": ".".join(parent_groups + [group]) if parent_groups else group,
            "pages": create_nav_structure(subpaths, parent_groups + [group]),
        }
        for group, subpaths in sorted(groups.items())
    ]

    # Sort pages
    sorted_pages = sorted(pages)

    # Return directories first, then files
    return sorted_groups + sorted_pages


def update_nav(mint_json_path: Path, new_nav_pages: list[Any]) -> None:
    """Update the 'API Reference' section in mint.json navigation with new pages.

    Args:
        mint_json_path: Path to mint.json file
        new_nav_pages: New navigation structure to replace in API Reference pages
    """
    try:
        # Read the current mint.json
        with open(mint_json_path) as f:
            mint_config = json.load(f)

        # Find and update the API Reference section
        for section in mint_config["navigation"]:
            if section.get("group") == "API Reference":
                section["pages"] = new_nav_pages
                break

        # Write back to mint.json with proper formatting
        with open(mint_json_path, "w") as f:
            json.dump(mint_config, f, indent=2)
            f.write("\n")

    except json.JSONDecodeError:
        print(f"Error: {mint_json_path} is not valid JSON")
    except Exception as e:
        print(f"Error updating mint.json: {e}")


def update_mint_json_with_api_nav(script_dir: Path, api_dir: Path) -> None:
    """Update mint.json with MDX files in the API directory."""
    mint_json_path = script_dir / "mint.json"
    if not mint_json_path.exists():
        print(f"File not found: {mint_json_path}")
        sys.exit(1)

    # Get all MDX files in the API directory
    mdx_files = get_mdx_files(api_dir)

    # Create navigation structure
    nav_structure = create_nav_structure(mdx_files)

    # Update mint.json with new navigation
    update_nav(mint_json_path, nav_structure)


def generate_mint_json_from_template(mint_json_template_path: Path, mint_json_path: Path) -> None:
    # if mint.json already exists, delete it
    if mint_json_path.exists():
        os.remove(mint_json_path)

    # Copy the template file to mint.json
    contents = read_file_content(mint_json_template_path)
    mint_json_template_content = Template(contents).render()

    # Parse the rendered template content as JSON
    mint_json_data = json.loads(mint_json_template_content)

    # Write content to mint.json
    with open(mint_json_path, "w") as f:
        json.dump(mint_json_data, f, indent=2)


def main() -> None:
    script_dir = Path(__file__).parent.absolute()

    parser = argparse.ArgumentParser(description="Process API reference documentation")
    parser.add_argument(
        "--api-dir",
        type=Path,
        help="Directory containing API documentation to process",
        default=script_dir / "docs" / "reference",
    )

    args = parser.parse_args()

    if args.api_dir.exists():
        # Force delete the directory and its contents
        shutil.rmtree(args.api_dir, ignore_errors=True)

    api_dir_rel_path = args.api_dir.resolve().relative_to(script_dir)

    # Run pdoc3
    print("Running pdoc3...")
    run_pdoc3(api_dir_rel_path)

    # Convert MD to MDX
    print("Converting MD files to MDX...")
    convert_md_to_mdx(args.api_dir)

    # Create mint.json from the template file
    mint_json_template_path = script_dir / "mint-json-template.json.jinja"
    mint_json_path = script_dir / "mint.json"

    print("Generating mint.json from template...")
    generate_mint_json_from_template(mint_json_template_path, mint_json_path)

    # Update mint.json
    update_mint_json_with_api_nav(script_dir, args.api_dir)

    print("API reference processing complete!")


if __name__ == "__main__":
    main()
