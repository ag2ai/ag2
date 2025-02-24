# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


import json
import re
import shutil
from pathlib import Path

from ..import_utils import optional_import_block, require_optional_import
from .utils import NavigationGroup, copy_files, get_git_tracked_and_untracked_files_in_directory

with optional_import_block():
    from jinja2 import Template


def filter_excluded_files(files: list[Path], exclusion_list: list[str], website_dir: Path) -> list[Path]:
    return [
        file
        for file in files
        if not any(str(file.relative_to(website_dir)).startswith(excl) for excl in exclusion_list)
    ]


def copy_file(file: Path, mkdocs_output_dir: Path) -> None:
    dest = mkdocs_output_dir / file.relative_to(file.parents[1])
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file, dest)


def transform_content_for_mkdocs(content: str) -> str:
    # Transform admonitions (Tip, Warning, Note)
    tag_mappings = {
        "Tip": "tip",
        "Warning": "warning",
        "Note": "note",
        "Danger": "danger",
    }
    for html_tag, mkdocs_type in tag_mappings.items():
        pattern = f"<{html_tag}>(.*?)</{html_tag}>"

        def replacement(match):
            inner_content = match.group(1).strip()
            return f"!!! {mkdocs_type}\n    {inner_content}"

        content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    # Clean up style tags with double curly braces
    style_pattern = r"style\s*=\s*{{\s*([^}]+)\s*}}"

    def style_replacement(match):
        style_content = match.group(1).strip()
        return f"style={{ {style_content} }}"

    content = re.sub(style_pattern, style_replacement, content)

    return content


def process_and_copy_files(input_dir: Path, output_dir: Path, files: list[Path]) -> None:
    for file in files:
        if file.suffix == ".mdx":
            content = file.read_text()
            processed_content = transform_content_for_mkdocs(content)
            dest = output_dir / file.relative_to(input_dir).with_suffix(".md")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(processed_content)
        else:
            copy_files(input_dir, output_dir, [file])
            # copy_file(file, output_dir)


def format_title(title: str, keywords: dict[str, str]) -> str:
    """Format a page title with proper capitalization for special keywords."""
    words = title.replace("-", " ").title().split()
    return " ".join(keywords.get(word, word) for word in words)


def format_page_entry(page_path: str, indent: str, keywords: dict[str, str]) -> str:
    """Format a single page entry as either a parenthesized path or a markdown link."""
    path = f"{page_path}.md"
    title = format_title(Path(page_path).name, keywords)
    return f"{indent}    - [{title}]({path})"


def format_navigation(nav: list[NavigationGroup], depth: int = 0, keywords: dict[str, str] = None) -> str:
    """
    Recursively format navigation structure into markdown-style nested list.

    Args:
        nav: List of navigation items with groups and pages
        depth: Current indentation depth
        keywords: Dictionary of special case word capitalizations

    Returns:
        Formatted navigation as a string
    """
    if keywords is None:
        keywords = {
            "Ag2": "AG2",
            "Rag": "RAG",
            "Llm": "LLM",
        }

    indent = "    " * depth
    result = []

    for item in nav:
        # Add group header
        result.append(f"{indent}- {item['group']}")

        # Process each page
        for page in item["pages"]:
            if isinstance(page, dict):
                # Handle nested navigation groups
                result.append(format_navigation([page], depth + 1, keywords))
            else:
                # Handle individual pages
                result.append(format_page_entry(page, indent, keywords))

    return "\n".join(result)


@require_optional_import("jinja2", "docs")
def generate_mkdocs_navigation(website_dir: Path, mkdocs_root_dir: Path, nav_exclusions: list[str]) -> None:
    mintlify_nav_template_path = website_dir / "mint-json-template.json.jinja"
    mkdocs_nav_path = mkdocs_root_dir / "docs" / "navigation_template.txt"
    summary_md_path = mkdocs_root_dir / "docs" / "SUMMARY.md"

    mintlify_json = json.loads(Template(mintlify_nav_template_path.read_text(encoding="utf-8")).render())
    mintlify_nav = mintlify_json["navigation"]
    filtered_nav = [item for item in mintlify_nav if item["group"] not in nav_exclusions]

    mkdocs_nav_content = "---\nsearch:\n  exclude: true\n---\n" + format_navigation(filtered_nav) + "\n"
    mkdocs_nav_path.write_text(mkdocs_nav_content)
    summary_md_path.write_text(mkdocs_nav_content)


def main() -> None:
    root_dir = Path(__file__).resolve().parents[2]
    website_dir = root_dir / "website"

    mint_inpur_dir = website_dir / "docs"

    mkdocs_root_dir = website_dir / "mkdocs"
    mkdocs_output_dir = mkdocs_root_dir / "docs" / "docs"

    exclusion_list = ["docs/_blogs", "docs/home", "docs/.gitignore", "docs/use-cases"]
    nav_exclusions = ["Use Cases"]

    files_to_copy = get_git_tracked_and_untracked_files_in_directory(mint_inpur_dir)
    filtered_files = filter_excluded_files(files_to_copy, exclusion_list, website_dir)

    process_and_copy_files(mint_inpur_dir, mkdocs_output_dir, filtered_files)

    generate_mkdocs_navigation(website_dir, mkdocs_root_dir, nav_exclusions)
