# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from textwrap import dedent, indent
from typing import Literal, Optional, TypedDict, Union

from ..import_utils import optional_import_block, require_optional_import

with optional_import_block():
    import yaml


EDIT_URL_HTML = """
<div className="edit-url-container">
    <a className="edit-url" href="https://github.com/ag2ai/ag2/edit/main/{file_path}" target='_blank'><Icon icon="pen" iconType="solid" size="13px"/> Edit this page</a>
</div>
"""

build_system = Literal["mkdocs", "mintlify"]


class NavigationGroup(TypedDict):
    group: str
    pages: list[Union[str, "NavigationGroup"]]


def get_git_tracked_and_untracked_files_in_directory(directory: Path) -> list[Path]:
    """Get all files in the directory that are tracked by git or newly added."""
    proc = subprocess.run(
        ["git", "-C", str(directory), "ls-files", "--others", "--exclude-standard", "--cached"],
        capture_output=True,
        text=True,
        check=True,
    )
    return list({directory / p for p in proc.stdout.splitlines()})


def copy_files(src_dir: Path, dst_dir: Path, files_to_copy: list[Path]) -> None:
    """Copy files from src_dir to dst_dir."""
    for file in files_to_copy:
        if file.is_file():
            dst = dst_dir / file.relative_to(src_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dst)


def copy_only_git_tracked_and_untracked_files(src_dir: Path, dst_dir: Path, ignore_dir: Optional[str] = None) -> None:
    """Copy only the files that are tracked by git or newly added from src_dir to dst_dir."""
    tracked_and_new_files = get_git_tracked_and_untracked_files_in_directory(src_dir)

    if ignore_dir:
        ignore_dir_rel_path = src_dir / ignore_dir

        tracked_and_new_files = list({
            file for file in tracked_and_new_files if not any(parent == ignore_dir_rel_path for parent in file.parents)
        })

    copy_files(src_dir, dst_dir, tracked_and_new_files)


def remove_marker_blocks(content: str, marker_prefix: str) -> str:
    """Remove marker blocks from the content.

    Args:
        content: The source content to process
        marker_prefix: The marker prefix to identify blocks to remove (without the START/END suffix)

    Returns:
        Processed content with appropriate blocks handled
    """
    # First, remove blocks with the specified marker completely
    if f"{{/* {marker_prefix}-START */}}" in content:
        pattern = rf"\{{/\* {re.escape(marker_prefix)}-START \*/\}}.*?\{{/\* {re.escape(marker_prefix)}-END \*/\}}"
        content = re.sub(pattern, "", content, flags=re.DOTALL)

    # Now, remove markers but keep content for the other marker type
    other_prefix = (
        "DELETE-ME-WHILE-BUILDING-MKDOCS"
        if marker_prefix == "DELETE-ME-WHILE-BUILDING-MINTLIFY"
        else "DELETE-ME-WHILE-BUILDING-MINTLIFY"
    )

    # Remove start markers
    start_pattern = rf"\{{/\* {re.escape(other_prefix)}-START \*/\}}\s*"
    content = re.sub(start_pattern, "", content)

    # Remove end markers
    end_pattern = rf"\s*\{{/\* {re.escape(other_prefix)}-END \*/\}}"
    content = re.sub(end_pattern, "", content)

    # Fix any double newlines that might have been created
    content = re.sub(r"\n\s*\n\s*\n", "\n\n", content)

    return content


# Sort files by parent directory date (if exists) and name
def sort_files_by_date(file_path: Path) -> tuple[datetime, str]:
    dirname = file_path.parent.name
    try:
        # Extract date from directory name (first 3 parts)
        date_str = "-".join(dirname.split("-")[:3])
        date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        date = datetime.min
    return (date, dirname)


def construct_authors_html(authors_list: list[str], authors_dict: dict[str, dict[str, str]], build_system: str) -> str:
    """Constructs HTML for displaying author cards in a blog.

    Args:
        authors_list: list of author identifiers
        authors_dict: Dictionary containing author information keyed by author identifier
        build_system: The build system being used (mkdocs or mintlify)
    Returns:
        str: Formatted HTML string containing author cards
    """
    if not authors_list:
        return ""

    card_template_mintlify = """
        <Card href="{url}">
            <div class="col card">
              <div class="img-placeholder">
                <img noZoom src="{avatar}" />
              </div>
              <div>
                <p class="name">{name}</p>
                <p>{description}</p>
              </div>
            </div>
        </Card>"""

    card_template_mkdocs = """
        <div class="card">
            <div class="col card">
              <div class="img-placeholder">
                <img noZoom src="{avatar}" />
              </div>
              <div>
                <p class="name">{name}</p>
                <p>{description}</p>
              </div>
            </div>
        </div>
    """

    card_template = card_template_mintlify if build_system == "mintlify" else card_template_mkdocs

    authors_html = [card_template.format(**authors_dict[author]) for author in authors_list]

    author_label = "Author:" if len(authors_list) == 1 else "Authors:"
    authors_html_str = indent("".join(authors_html), "        ")

    retval = ""
    if build_system == "mintlify":
        retval = dedent(
            f"""
                <div class="blog-authors">
                <p class="authors">{author_label}</p>
                <CardGroup cols={{2}}>{authors_html_str}
                </CardGroup>
                </div>
            """
        )
    else:
        retval = dedent(
            f"""
                <div class="blog-authors">
                <p class="authors">{author_label}</p>
                <div class="card-group">
                    {authors_html_str}
                </div>
                </div>
            """
        )
    return retval


def separate_front_matter_and_content(file_path: Path) -> tuple[str, str]:
    """Separate front matter and content from a markdown file.

    Args:
        file_path (Path): Path to the mdx file
    """
    content = file_path.read_text(encoding="utf-8")

    if content.startswith("---"):
        front_matter_end = content.find("---", 3)
        front_matter = content[0 : front_matter_end + 3]
        content = content[front_matter_end + 3 :].strip()
        return front_matter, content

    return "", content


def ensure_edit_url(content: str, file_path: Path) -> str:
    """Ensure editUrl is present in the content.
    Args:
        content (str): Content of the file
        file_path (Path): Path to the file
    """
    html_placeholder = [line for line in EDIT_URL_HTML.splitlines() if line.strip() != ""][0]
    if html_placeholder in content:
        return content

    return content + EDIT_URL_HTML.format(file_path=file_path)


@require_optional_import("yaml", "docs")
def add_authors_and_social_preview(
    website_build_dir: Path,
    target_dir: Path,
    all_authors_info: dict[str, dict[str, str]],
    build_system: build_system = "mintlify",
) -> None:
    """Add authors info and social share image to mdx files in the target directory."""

    social_img_html = (
        """\n<div>
<img noZoom className="social-share-img"
  src="https://media.githubusercontent.com/media/ag2ai/ag2/refs/heads/main/website/static/img/cover.png"
  alt="social preview"
  style={{ position: 'absolute', left: '-9999px' }}
/>
</div>"""
        if build_system == "mintlify"
        else ""
    )

    target_file_extension = "mdx" if build_system == "mintlify" else "md"
    for file_path in target_dir.glob(f"**/*.{target_file_extension}"):
        try:
            front_matter_string, content = separate_front_matter_and_content(file_path)

            # Convert single author to list and handle authors
            front_matter = yaml.safe_load(front_matter_string[4:-3])
            authors = front_matter.get("authors", [])
            authors_list = [authors] if isinstance(authors, str) else authors

            # Generate authors HTML
            authors_html = (
                construct_authors_html(authors_list, all_authors_info, build_system)
                if '<div class="blog-authors">' not in content
                else ""
            )

            # Combine content
            new_content = f"{front_matter_string}\n{social_img_html}\n{authors_html}\n{content}"

            # ensure editUrl is present

            if build_system == "mintlify":
                rel_file_path = (
                    str(file_path.relative_to(website_build_dir.parent))
                    .replace("build/docs/", "website/docs/")
                    .replace("website/docs/blog/", "website/docs/_blogs/")
                )
                content_with_edit_url = ensure_edit_url(new_content, Path(rel_file_path))

                # replace the mkdocs excerpt marker
                content_with_edit_url = content_with_edit_url.replace(r"\<!-- more -->", "")

                file_path.write_text(f"{content_with_edit_url}\n", encoding="utf-8")

            else:
                file_path.write_text(f"{new_content}\n", encoding="utf-8")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue


@require_optional_import("yaml", "docs")
def get_authors_info(authors_yml: Path) -> dict[str, dict[str, str]]:
    try:
        all_authors_info = yaml.safe_load(authors_yml.read_text(encoding="utf-8"))["authors"]
    except (yaml.YAMLError, OSError) as e:
        print(f"Error reading authors file: {e}")
        sys.exit(1)

    return all_authors_info  # type: ignore [no-any-return]


@require_optional_import("yaml", "docs")
def render_gallery_html(gallery_file_path: Path) -> str:
    """Renders a gallery of items with tag filtering

    Args:
        gallery_file_path: Path to the YAML file containing gallery items

    Returns:
        HTML string for the gallery
    """
    try:
        # Load gallery items from YAML file
        with open(gallery_file_path, "r") as file:
            gallery_items = yaml.safe_load(file)

        # Ensure gallery_items is a list
        if not isinstance(gallery_items, list):
            return f"<div class='error'>Error: YAML file did not contain a list, but a {type(gallery_items)}</div>"

        # Extract all unique tags from gallery items
        all_tags = []
        for item in gallery_items:
            if not isinstance(item, dict):
                continue

            if "tags" in item and item["tags"]:
                all_tags.extend(item["tags"])
        all_tags = sorted(list(set(all_tags)))

        # Generate HTML directly
        html = '<div class="examples-gallery-container">'

        # Generate tag filter select
        html += '<select multiple class="tag-filter" data-placeholder="Filter by tags">'
        for tag in all_tags:
            html += f'<option value="{tag}">{tag}</option>'
        html += "</select>"

        # Generate gallery cards
        html += '<div class="gallery-cards">'

        for item in gallery_items:
            # Skip if item is not a dictionary
            if not isinstance(item, dict):
                continue

            image_url = item.get("image", "default.png")
            if image_url and not isinstance(image_url, str):
                # Handle case where image is not a string
                image_url = "default.png"

            if image_url and not image_url.startswith("http"):
                image_url = f"../../../../assets/img/gallery/{image_url}"

            # Handle default image
            if not image_url:
                image_url = "../../../../assets/img/gallery/default.png"

            # Tags HTML
            tags_html = ""
            if "tags" in item and item["tags"]:
                tags_html = '<div class="tags-container">'
                for tag in item["tags"]:
                    tags_html += f'<span class="tag" data-tag="{tag}">{tag}</span>'
                tags_html += "</div>"

            # Badges HTML
            badges_html = ""
            notebook_src = item.get("source", None)

            if notebook_src:
                colab_href = f"https://colab.research.google.com/github/ag2ai/ag2/blob/main/{notebook_src}"
                github_href = f"https://github.com/ag2ai/ag2/blob/main/{notebook_src}"
                badges_html = f"""
                <div class="badges">
                    <a style="margin-right: 5px" href="{colab_href}" target="_blank">
                        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
                    </a>
                    <p class="hidden">{item.get("title", "")}</p>
                    <a href="{github_href}" target="_blank">
                        <img alt="GitHub" src="https://img.shields.io/badge/Open%20on%20GitHub-grey?logo=github"/>
                    </a>
                </div>
                """

            # Generate card HTML with safer access to attributes
            tags_str = ",".join(item.get("tags", [])) if isinstance(item.get("tags"), list) else ""

            # Generate HTML for image tag
            img_tag = (
                f'<img src="{image_url}" alt="{item.get("title", "")}" class="card-image">' if not notebook_src else ""
            )

            data_link_target = "_self" if notebook_src else "_blank"
            html += f"""
            <div class="card" data-link="{item.get("link", "#")}" data-rel-link="{item.get("rel_link", "#")}" data-tags="{tags_str}" data-link-target="{data_link_target}">
                <div class="card-container">
                    {img_tag}
                    <p class="card-title">{item.get("title", "")}</p>
                    {badges_html}
                    <p class="card-description">{item.get("description", item.get("title", ""))}</p>
                    {tags_html}
                </div>
            </div>
            """

        # Close containers
        html += """
            </div>
        </div>
        """

        return html

    except FileNotFoundError:
        return f"<div class='error'>Error: YAML file not found at path: {gallery_file_path}</div>"
    except yaml.YAMLError:
        return f"<div class='error'>Error: Invalid YAML format in file: {gallery_file_path}</div>"
    except Exception as e:
        return f"<div class='error'>Error: {str(e)}</div>"
