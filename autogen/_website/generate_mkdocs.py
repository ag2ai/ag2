# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


import json
import re
import shutil
from pathlib import Path
from typing import Optional, Union

from ..import_utils import optional_import_block, require_optional_import
from .notebook_processor import (
    create_base_argument_parser,
    process_notebooks_core,
)
from .utils import (
    NavigationGroup,
    add_authors_and_social_preview,
    copy_files,
    get_authors_info,
    get_git_tracked_and_untracked_files_in_directory,
    remove_marker_blocks,
    render_gallery_html,
    sort_files_by_date,
)

with optional_import_block():
    import yaml
    from jinja2 import Template


def filter_excluded_files(files: list[Path], exclusion_list: list[str], website_dir: Path) -> list[Path]:
    return [
        file
        for file in files
        if not any(Path(str(file.relative_to(website_dir))).as_posix().startswith(excl) for excl in exclusion_list)
    ]


def copy_file(file: Path, mkdocs_output_dir: Path) -> None:
    dest = mkdocs_output_dir / file.relative_to(file.parents[1])
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file, dest)


def transform_tab_component(content: str) -> str:
    """Transform React-style tab components to MkDocs tab components.

    Args:
        content: String containing React-style tab components.
            Expected format is:
            <Tabs>
                <Tab title="Title 1">
                    content 1
                </Tab>
                <Tab title="Title 2">
                    content 2
                </Tab>
            </Tabs>

    Returns:
        String with MkDocs tab components:
            === "Title 1"
                content 1

            === "Title 2"
                content 2
    """
    if "<Tabs>" not in content:
        return content

    # Find and replace each Tabs section
    pattern = re.compile(r"<Tabs>(.*?)</Tabs>", re.DOTALL)

    def replace_tabs(match: re.Match[str]) -> str:
        tabs_content = match.group(1)

        # Extract all Tab elements
        tab_pattern = re.compile(r'<Tab title="([^"]+)">(.*?)</Tab>', re.DOTALL)
        tabs = tab_pattern.findall(tabs_content)

        if not tabs:
            return ""

        result = []

        for i, (title, tab_content) in enumerate(tabs):
            # Add tab header
            result.append(f'=== "{title}"')

            # Process content by maintaining indentation structure
            lines = tab_content.strip().split("\n")

            # Find minimum common indentation for non-empty lines
            non_empty_lines = [line for line in lines if line.strip()]
            min_indent = min([len(line) - len(line.lstrip()) for line in non_empty_lines]) if non_empty_lines else 0

            # Remove common indentation and add 4-space indent
            processed_lines = []
            for line in lines:
                if line.strip():
                    # Remove the common indentation but preserve relative indentation
                    if len(line) >= min_indent:
                        processed_lines.append("    " + line[min_indent:])
                    else:
                        processed_lines.append("    " + line.lstrip())
                else:
                    processed_lines.append("")

            result.append("\n".join(processed_lines))

            # Add a blank line between tabs (but not after the last one)
            if i < len(tabs) - 1:
                result.append("")

        return "\n".join(result)

    # Replace each Tabs section
    result = pattern.sub(replace_tabs, content)

    return result


def transform_card_grp_component(content: str) -> str:
    # Replace CardGroup tags
    modified_content = re.sub(r"<CardGroup\s+cols=\{(\d+)\}>\s*", "", content)
    modified_content = re.sub(r"\s*</CardGroup>", "", modified_content)

    # Replace Card tags with title and href attributes
    pattern = r'<Card\s+title="([^"]*)"\s+href="([^"]*)">(.*?)</Card>'
    replacement = r'<a class="card" href="\2">\n<h2>\1</h2>\3</a>'
    modified_content = re.sub(pattern, replacement, modified_content, flags=re.DOTALL)

    # Replace simple Card tags
    modified_content = re.sub(r"<Card>", '<div class="card">', modified_content)
    modified_content = re.sub(r"</Card>", "</div>", modified_content)

    return modified_content


def fix_asset_path(content: str) -> str:
    # Replace static/img paths with ag2/assets/img
    modified_content = re.sub(r'src="/static/img/([^"]+)"', r'src="/assets/img/\1"', content)

    # Replace docs paths with ag2/docs
    modified_content = re.sub(r'href="/docs/([^"]+)"', r'href="/docs/\1"', modified_content)

    return modified_content


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

        def replacement(match: re.Match[str]) -> str:
            inner_content = match.group(1).strip()

            lines = inner_content.split("\n")

            non_empty_lines = [line for line in lines if line.strip()]
            min_indent = min([len(line) - len(line.lstrip()) for line in non_empty_lines]) if non_empty_lines else 0

            # Process each line
            processed_lines = []
            for line in lines:
                if line.strip():
                    # Remove common indentation and add 4-space indent
                    if len(line) >= min_indent:
                        processed_lines.append("    " + line[min_indent:])
                    else:
                        processed_lines.append("    " + line.lstrip())
                else:
                    processed_lines.append("")

            # Format the admonition with properly indented content
            return f"!!! {mkdocs_type.lstrip()}\n" + "\n".join(processed_lines)

        content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    # Clean up style tags with double curly braces
    style_pattern = r"style\s*=\s*{{\s*([^}]+)\s*}}"

    def style_replacement(match: re.Match[str]) -> str:
        style_content = match.group(1).strip()
        return f"style={{ {style_content} }}"

    content = re.sub(style_pattern, style_replacement, content)

    # Transform tab components
    content = transform_tab_component(content)

    # Transform CardGroup components
    content = transform_card_grp_component(content)

    # Fix assets path
    content = fix_asset_path(content)

    # Remove the mintlify specific markers
    content = remove_marker_blocks(content, "DELETE-ME-WHILE-BUILDING-MKDOCS")

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


def format_navigation(nav: list[NavigationGroup], depth: int = 0, keywords: Optional[dict[str, str]] = None) -> str:
    """Recursively format navigation structure into markdown-style nested list.

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

    ret_val = "\n".join(result)
    ret_val = ret_val.replace("- Home\n", "- [Home](index.md)\n")
    return ret_val


def add_api_ref_to_mkdocs_template(mkdocs_nav: str, section_to_follow: str) -> str:
    """Add API Reference section to the navigation template."""
    api_reference_section = """- API References
{api}
"""
    section_to_follow_marker = f"- {section_to_follow}"

    replacement_content = f"{api_reference_section}{section_to_follow_marker}"
    ret_val = mkdocs_nav.replace(section_to_follow_marker, replacement_content)
    return ret_val


@require_optional_import("jinja2", "docs")
def generate_mkdocs_navigation(website_dir: Path, mkdocs_root_dir: Path, nav_exclusions: list[str]) -> None:
    mintlify_nav_template_path = website_dir / "mint-json-template.json.jinja"
    mkdocs_nav_path = mkdocs_root_dir / "docs" / "navigation_template.txt"
    summary_md_path = mkdocs_root_dir / "docs" / "SUMMARY.md"

    mintlify_json = json.loads(Template(mintlify_nav_template_path.read_text(encoding="utf-8")).render())
    mintlify_nav = mintlify_json["navigation"]
    filtered_nav = [item for item in mintlify_nav if item["group"] not in nav_exclusions]

    mkdocs_nav = format_navigation(filtered_nav)
    mkdocs_nav_with_api_ref = add_api_ref_to_mkdocs_template(mkdocs_nav, "Contributor Guide")

    blog_nav = "- Blog\n    - [Blog](docs/blog)"

    mkdocs_nav_content = "---\nsearch:\n  exclude: true\n---\n" + mkdocs_nav_with_api_ref + "\n" + blog_nav + "\n"
    mkdocs_nav_path.write_text(mkdocs_nav_content)
    summary_md_path.write_text(mkdocs_nav_content)


def copy_assets(website_dir: Path) -> None:
    src_dir = website_dir / "static" / "img"
    dest_dir = website_dir / "mkdocs" / "docs" / "assets" / "img"

    git_tracket_img_files = get_git_tracked_and_untracked_files_in_directory(website_dir / "static" / "img")
    copy_files(src_dir, dest_dir, git_tracket_img_files)


def add_excerpt_marker(content: str) -> str:
    """Add <!-- more --> marker before the second heading in markdown body content.

    Args:
        content (str): Body content of the markdown file (without frontmatter)

    Returns:
        str: Modified body content with <!-- more --> added
    """

    if "<!-- more -->" in content:
        return content.replace(r"\<!-- more -->", "<!-- more -->")

    # Find all headings
    heading_pattern = re.compile(r"^(#{1,6}\s+.+?)$", re.MULTILINE)
    headings = list(heading_pattern.finditer(content))

    # If there are fewer than 2 headings, add the marker at the end
    if len(headings) < 2:
        # If there's content, add the marker at the end
        return content.rstrip() + "\n\n<!-- more -->\n"

    # Get position of the second heading
    second_heading = headings[1]
    position = second_heading.start()

    # Insert the more marker before the second heading
    return content[:position] + "\n<!-- more -->\n\n" + content[position:]


def generate_url_slug(file: Path) -> str:
    parent_dir = file.parts[-2]
    slug = "-".join(parent_dir.split("-")[3:])
    return f"\nslug: {slug}"


def process_blog_contents(contents: str, file: Path) -> str:
    # Split the content into parts
    parts = contents.split("---", 2)
    if len(parts) < 3:
        return contents

    frontmatter = parts[1]
    content = parts[2]

    # Extract tags
    tags_match = re.search(r"tags:\s*\[(.*?)\]", frontmatter)
    if not tags_match:
        return contents

    tags_str = tags_match.group(1)
    tags = [tag.strip() for tag in tags_str.split(",")]

    # Extract date from second-to-last part of file path
    date_match = re.match(r"(\d{4}-\d{2}-\d{2})", file.parts[-2])
    date = date_match.group(1) if date_match else None

    # Remove original tags
    frontmatter = re.sub(r"tags:\s*\[.*?\]", "", frontmatter).strip()

    # Format tags and categories as YAML lists
    tags_yaml = "tags:\n    - " + "\n    - ".join(tags)
    categories_yaml = "categories:\n    - " + "\n    - ".join(tags)

    # Add date to metadata
    date_yaml = f"\ndate: {date}" if date else ""

    # Add URL slug metadata
    url_slug = generate_url_slug(file)

    # add the excerpt marker in the content
    content_with_excerpt_marker = add_excerpt_marker(content)

    return f"---\n{frontmatter}\n{tags_yaml}\n{categories_yaml}{date_yaml}{url_slug}\n---{content_with_excerpt_marker}"


def fix_snippet_imports(content: str, snippets_dir: Path) -> str:
    """Replace import statements for MDX files from snippets directory with the target format.

    Args:
        content (str): Content containing import statements
        snippets_dir (Path): Path to the snippets directory

    Returns:
        str: Content with import statements replaced
    """
    # Regular expression to find import statements for MDX files from /snippets/
    import_pattern = re.compile(r'import\s+(\w+)\s+from\s+"(/snippets/[^"]+\.mdx)"\s*;')

    # Function to replace the matched import statement
    def replace_import(match: re.Match[str]) -> str:
        relative_path = match.group(2).lstrip("/")

        # Remove "snippets/" prefix from the relative path if it exists
        if relative_path.startswith("snippets/"):
            relative_path = relative_path[len("snippets/") :]

        # Create the new format: {!<full_path_to_snippets>/<relative_path> !}
        new_path = "{!" + (snippets_dir / relative_path).as_posix() + " !}"
        return new_path + "\n"

    # Replace all matching import statements
    return import_pattern.sub(replace_import, content)


def process_blog_files(mkdocs_output_dir: Path, authors_yml_path: Path, snippets_src_path: Path) -> None:
    src_blog_dir = mkdocs_output_dir / "_blogs"
    target_blog_dir = mkdocs_output_dir / "blog"
    target_posts_dir = target_blog_dir / "posts"
    snippets_dir = mkdocs_output_dir.parent / "snippets"

    # Create the target posts directory
    target_posts_dir.mkdir(parents=True, exist_ok=True)

    # Create the index file in the target blog directory
    index_file = target_blog_dir / "index.md"
    index_file.write_text("# Blog\n\n")

    # Get all files to copy
    files_to_copy = list(src_blog_dir.rglob("*"))

    # process blog metadata
    for file in files_to_copy:
        if file.suffix == ".md":
            contents = file.read_text()
            processed_contents = process_blog_contents(contents, file)
            processed_contents = fix_snippet_imports(processed_contents, snippets_dir)
            file.write_text(processed_contents)

    # Copy files from source to target
    copy_files(src_blog_dir, target_posts_dir, files_to_copy)

    # Copy snippets directory
    snippets_files_to_copy = list(snippets_src_path.rglob("*"))
    copy_files(snippets_src_path, snippets_dir, snippets_files_to_copy)

    # Copy authors_yml_path to the target_blog_dir and rename it as .authors.yml
    target_authors_yml_path = target_blog_dir / ".authors.yml"
    shutil.copy2(authors_yml_path, target_authors_yml_path)


_is_first_notebook = True


def add_front_matter_to_metadata_yml(
    front_matter: dict[str, Union[str, list[str], None]], website_build_directory: Path, rendered_mdx: Path
) -> None:
    """Add notebook metadata to a YAML file containing metadata for all notebooks."""
    global _is_first_notebook

    source = front_matter.get("source_notebook")
    if isinstance(source, str) and source.startswith("/website/docs/"):
        return

    # Get the metadata file path
    metadata_yml_path = website_build_directory / "../../data/notebooks_metadata.yml"

    # Create parent directories if they don't exist
    metadata_yml_path.parent.mkdir(parents=True, exist_ok=True)

    # If this is the first notebook, delete the existing file
    if _is_first_notebook and metadata_yml_path.exists():
        metadata_yml_path.unlink()
        _is_first_notebook = False

    # Create new entry for current notebook
    title = front_matter.get("title", "")
    link = f"/docs/use-cases/notebooks/notebooks/{rendered_mdx.stem}"
    description = front_matter.get("description", "")
    tags = front_matter.get("tags", []) or []

    # Escape quotes in strings
    title = str(title).replace('"', '\\"')
    description = str(description).replace('"', '\\"')
    source_str = str(source or "").replace('"', '\\"')

    # Open file in append mode
    with open(metadata_yml_path, "a", encoding="utf-8") as f:
        # Write the entry
        f.write(f'- title: "{title}"\n')
        f.write(f'  link: "{link}"\n')
        f.write(f'  description: "{description}"\n')
        f.write('  image: ""\n')

        # Write tags
        if tags:
            f.write("  tags:\n")
            for tag in tags:
                if tag:  # Only write non-empty tags
                    tag_str = str(tag).replace('"', '\\"')
                    f.write(f'    - "{tag_str}"\n')
        else:
            f.write("  tags: []\n")

        # Write source
        f.write(f'  source: "{source_str}"\n')
        f.write("\n")


def transform_admonition_blocks(content: str) -> str:
    """Transform admonition blocks from ::: syntax to Material for MkDocs syntax.

    Converts blocks like:
    :::info Requirements
    content here
    :::

    To:
    !!! info "Requirements"
        content here

    Args:
        content: String containing ::: syntax admonition blocks

    Returns:
        String with Material for MkDocs admonition blocks
    """

    tag_mappings = {
        "Tip": "tip",
        "Warning": "warning",
        "Note": "note",
        "Danger": "danger",
    }

    # Simplified approach: first detect admonition blocks boundaries
    lines = content.split("\n")
    admonition_start = None
    admonition_type = None
    admonition_title = None
    admonition_content: list[str] = []
    result_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for admonition start
        if line.strip().startswith(":::") and admonition_start is None:
            admonition_start = i
            # Extract admonition type and optional title
            match = re.match(r":::(\w+)(?:\s+(.+))?", line.strip())
            if match:
                admonition_type = match.group(1)
                admonition_title = match.group(2) if match.group(2) else ""
            else:
                # No match for admonition type means we couldn't parse the format
                admonition_type = None
            i += 1
            continue

        # Check for admonition end
        elif line.strip() == ":::" and admonition_start is not None:
            # If admonition_type is None, preserve the original content
            if admonition_type is None:
                # Add back the original admonition block without transformation
                original_lines = []
                original_lines.append(lines[admonition_start])  # Opening :::
                original_lines.extend(admonition_content)  # Content
                original_lines.append(line)  # Closing :::
                result_lines.extend(original_lines)
            else:
                # Process as before for valid admonition types
                # Map the admonition type
                if admonition_type in tag_mappings:
                    mapped_type = tag_mappings[admonition_type]
                else:
                    # Try case-insensitive match
                    for tag, mapped in tag_mappings.items():
                        if tag.lower() == admonition_type.lower():
                            mapped_type = mapped
                            break
                    else:
                        # Default to lowercase of original if no mapping found
                        mapped_type = admonition_type.lower()

                # Process indentation
                if admonition_content:
                    # Find minimum common indentation
                    non_empty_lines = [line for line in admonition_content if line.strip()]
                    min_indent = min((len(line) - len(line.lstrip()) for line in non_empty_lines), default=0)

                    # Remove common indentation and add 4-space indent
                    processed_content = []
                    for line in admonition_content:
                        if line.strip():
                            if len(line) >= min_indent:
                                processed_content.append("    " + line[min_indent:])
                            else:
                                processed_content.append("    " + line.lstrip())
                        else:
                            processed_content.append("")
                else:
                    processed_content = []

                # Create the MkDocs admonition
                if admonition_title:
                    mkdocs_admonition = [f'!!! {mapped_type} "{admonition_title}"'] + processed_content
                else:
                    mkdocs_admonition = [f"!!! {mapped_type}"] + processed_content

                # Add the processed admonition
                result_lines.extend(mkdocs_admonition)

            # Reset admonition tracking
            admonition_start = None
            admonition_type = None
            admonition_title = None
            admonition_content = []
            i += 1
            continue

        elif admonition_start is not None:
            admonition_content.append(line)
            i += 1
            continue

        else:
            result_lines.append(line)
            i += 1

    if admonition_start is not None:
        for j in range(admonition_start, len(lines)):
            result_lines.append(lines[j])

    return "\n".join(result_lines)


def remove_mdx_code_blocks(content: str) -> str:
    """Remove ````mdx-code-block and ```` markers from the content.

    This function removes the mdx-code-block markers while preserving the content inside.

    Args:
        content: String containing mdx-code-block markers

    Returns:
        String with mdx-code-block markers removed
    """

    # Pattern to match mdx-code-block sections
    # Captures everything between ````mdx-code-block and ````
    pattern = re.compile(r"````mdx-code-block\n(.*?)\n````", re.DOTALL)

    # Replace with just the content (group 1)
    result = pattern.sub(r"\1", content)

    return result


@require_optional_import("yaml", "docs")
def post_process_func(
    rendered_mdx: Path,
    source_notebooks: Path,
    front_matter: dict[str, Union[str, list[str], None]],
    website_build_directory: Path,
) -> None:
    with open(rendered_mdx, encoding="utf-8") as f:
        content = f.read()

    # If there is front matter in the mdx file, we need to remove it
    if content.startswith("---"):
        front_matter_end = content.find("---", 3)
        mdx_front_matter = yaml.safe_load(content[4:front_matter_end])
        # Merge while preserving original values
        front_matter = {**front_matter, **mdx_front_matter}
        content = content[front_matter_end + 3 :]

    # Clean heading IDs using regex - matches from # to the end of ID block
    content = re.sub(r"(#{1,6}[^{]+){#[^}]+}", r"\1", content)

    # Each intermediate path needs to be resolved for this to work reliably
    repo_root = Path(__file__).resolve().parents[2]
    repo_relative_notebook = source_notebooks.resolve().relative_to(repo_root)
    front_matter["source_notebook"] = f"/{repo_relative_notebook}"
    front_matter["custom_edit_url"] = f"https://github.com/ag2ai/ag2/edit/main/{repo_relative_notebook}"

    github_link = f"https://github.com/ag2ai/ag2/blob/main/{repo_relative_notebook}"
    content = (
        f'\n<a href="{github_link}" class="github-badge" target="_blank">'
        + """<img noZoom src="https://img.shields.io/badge/Open%20on%20GitHub-grey?logo=github" alt="Open on GitHub" />"""
        + "</a>"
        + content
    )

    # If no colab link is present, insert one
    if "colab-badge.svg" not in content:
        colab_link = f"https://colab.research.google.com/github/ag2ai/ag2/blob/main/{repo_relative_notebook}"
        content = (
            f'\n<a href="{colab_link}" class="colab-badge" target="_blank">'
            + """<img noZoom src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" />"""
            + "</a>"
            + content
        )

    # Create the front matter metadata js file for examples by notebook section
    add_front_matter_to_metadata_yml(front_matter, website_build_directory, rendered_mdx)

    # Dump front_matter to yaml
    front_matter_str = yaml.dump(front_matter, default_flow_style=False)

    # Add render_macros: false to the front matter
    front_matter_str += "render_macros: false\n"

    # Convert callout blocks
    # content = convert_callout_blocks(content)

    # Convert mdx image syntax to mintly image syntax
    # content = convert_mdx_image_blocks(content, rendered_mdx, website_build_directory)

    # ensure editUrl is present
    # content = ensure_edit_url(content, repo_relative_notebook)

    # Remove admonition blocks
    content = transform_admonition_blocks(content)

    # Remove mdx-code-block markers
    content = remove_mdx_code_blocks(content)

    # Generate the page title
    page_header = front_matter.get("title")
    page_title = f"# {page_header}\n\n" if page_header else ""

    # Rewrite the content as
    # ---
    # front_matter_str
    # ---
    # content
    new_content = f"---\n{front_matter_str}---\n\n{page_title}\n{content}"

    # Change the file extension to .md
    rendered_md = rendered_mdx.with_suffix(".md")

    with open(rendered_md, "w", encoding="utf-8") as f:
        f.write(new_content)

    # Optionally, remove the original .mdx file
    rendered_mdx.unlink()


def target_dir_func(website_build_directory: Path) -> Path:
    """Return the target directory for notebooks."""
    return website_build_directory / "use-cases" / "notebooks" / "notebooks"


def inject_gallery_html(notebooks_md_path: Path, metadata_yml_path: Path) -> None:
    """Generate the index.html file for the notebooks section."""
    with open(notebooks_md_path, encoding="utf-8") as f:
        content = f.read()

    gallery_html = render_gallery_html(metadata_yml_path)

    updated_content = content.replace("{{ render_gallery(gallery_items) }}", gallery_html)
    with open(notebooks_md_path, "w", encoding="utf-8") as f:
        f.write(updated_content)


@require_optional_import("yaml", "docs")
def add_notebooks_nav(mkdocs_nav_path: Path, metadata_yml_path: Path) -> None:
    """Add notebooks navigation to the summary markdown file.

    Args:
        mkdocs_nav_path: Path to the mkdocs navigation template file
        metadata_yml_path: Path to the notebooks metadata YAML file
    """
    # Read the metadata file to get notebook items
    with open(metadata_yml_path, "r") as file:
        items = yaml.safe_load(file)

    # Create navigation list entries for each notebook
    nav_list = []
    for item in items:
        _link = item["link"][1:] if item["link"].startswith("/") else item["link"]
        nav_list.append(f"        - [{item['title']}]({_link})\n")

    # Read the summary file
    with open(mkdocs_nav_path, "r") as file:
        lines = file.readlines()

    # Find where to insert the notebook entries
    for i, line in enumerate(lines):
        if line.strip() == "- [Notebooks](docs/use-cases/notebooks/Notebooks.md)":
            # Insert all notebook items after the Notebooks line
            # No need to insert extra blank lines, just the notebook entries
            for j, nav_item in enumerate(nav_list):
                lines.insert(i + 1 + j, nav_item)
            break

    # Write the updated content back to the summary file
    with open(mkdocs_nav_path, "w") as file:
        file.writelines(lines)


def generate_user_stories_nav(mkdocs_output_dir: Path, mkdocs_nav_path: Path) -> None:
    user_stories_dir = mkdocs_output_dir / "docs" / "user-stories"

    # Read all user story files and sort them by date (newest first)
    files = sorted(user_stories_dir.glob("**/index.md"), key=sort_files_by_date, reverse=True)

    # Prepare user stories navigation entries
    user_stories_entries = []
    for file in files:
        # Extract the title from the frontmatter using a simpler split approach
        content = file.read_text()

        # Split content at the "---" markers
        parts = content.split("---", 2)
        if len(parts) < 3:
            # No valid frontmatter found, use directory name as title
            title = file.parent.name
        else:
            # Parse the frontmatter
            frontmatter_text = parts[1].strip()
            frontmatter = yaml.safe_load(frontmatter_text)
            title = frontmatter.get("title", file.parent.name)

        # Generate relative path from the docs root directory
        relative_path = file.parent.relative_to(mkdocs_output_dir)
        path_for_link = str(relative_path).replace("\\", "/")

        # Format navigation entry
        user_stories_entries.append(f"    - [{title}]({path_for_link})")

    # Read existing navigation template
    nav_content = mkdocs_nav_path.read_text()
    user_stories_section = "- User Stories\n" + "\n".join(user_stories_entries)

    section_to_follow_marker = "- Contributor Guide"

    replacement_content = f"{user_stories_section}\n{section_to_follow_marker}"
    updated_nav_content = nav_content.replace(section_to_follow_marker, replacement_content)

    # Write updated navigation to file
    mkdocs_nav_path.write_text(updated_nav_content)


def add_authors_info_to_user_stories(website_dir: Path) -> None:
    mkdocs_output_dir = website_dir / "mkdocs" / "docs" / "docs"
    user_stories_dir = mkdocs_output_dir / "user-stories"
    authors_yml = website_dir / "blogs_and_user_stories_authors.yml"

    all_authors_info = get_authors_info(authors_yml)

    add_authors_and_social_preview(website_dir, user_stories_dir, all_authors_info, "mkdocs")

    for file_path in user_stories_dir.glob("**/*.md"):
        content = file_path.read_text(encoding="utf-8")
        updated_content = transform_content_for_mkdocs(content)
        file_path.write_text(updated_content, encoding="utf-8")


def main(force: bool) -> None:
    root_dir = Path(__file__).resolve().parents[2]
    website_dir = root_dir / "website"

    mint_input_dir = website_dir / "docs"

    mkdocs_root_dir = website_dir / "mkdocs"
    mkdocs_output_dir = mkdocs_root_dir / "docs" / "docs"

    parser = create_base_argument_parser()
    args = parser.parse_args(["render"])
    args.dry_run = False
    args.quarto_bin = "quarto"
    args.notebooks = None

    # check if args.force is set
    if force and mkdocs_output_dir.exists():
        shutil.rmtree(mkdocs_output_dir)

    exclusion_list = [
        "docs/.gitignore",
        "docs/installation",
        "docs/user-guide/getting-started",
        "docs/user-guide/models/litellm-with-watsonx.md",
        "docs/contributor-guide/Migration-Guide.md",
    ]
    nav_exclusions = [""]

    files_to_copy = get_git_tracked_and_untracked_files_in_directory(mint_input_dir)
    filtered_files = filter_excluded_files(files_to_copy, exclusion_list, website_dir)

    copy_assets(website_dir)
    process_and_copy_files(mint_input_dir, mkdocs_output_dir, filtered_files)

    snippets_dir_path = website_dir / "snippets"
    authors_yml_path = website_dir / "blogs_and_user_stories_authors.yml"

    process_blog_files(mkdocs_output_dir, authors_yml_path, snippets_dir_path)
    generate_mkdocs_navigation(website_dir, mkdocs_root_dir, nav_exclusions)

    if args.website_build_directory is None:
        args.website_build_directory = mkdocs_output_dir

    if args.notebook_directory is None:
        args.notebook_directory = mkdocs_root_dir / "../../notebook"

    if force and mkdocs_output_dir.exists():
        process_notebooks_core(args, post_process_func, target_dir_func)

    # Render Notebooks Gallery HTML
    notebooks_md_path = mkdocs_output_dir / "use-cases" / "notebooks" / "Notebooks.md"
    metadata_yml_path = Path(args.website_build_directory) / "../../data/notebooks_metadata.yml"
    inject_gallery_html(notebooks_md_path, metadata_yml_path)

    # Add Notebooks Navigation to Summary.md
    mkdocs_nav_path = mkdocs_root_dir / "docs" / "navigation_template.txt"
    add_notebooks_nav(mkdocs_nav_path, metadata_yml_path)

    # Render Community Gallery HTML
    community_md_path = mkdocs_output_dir / "use-cases" / "community-gallery" / "community-gallery.md"
    metadata_yml_path = Path(args.website_build_directory) / "../../data/gallery_items.yml"
    inject_gallery_html(community_md_path, metadata_yml_path)

    # Generate Navigation for User Stories
    docs_dir = mkdocs_root_dir / "docs"
    generate_user_stories_nav(docs_dir, mkdocs_nav_path)

    # Add Authors info to User Stories
    add_authors_info_to_user_stories(website_dir)
