# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
#!/usr/bin/env python

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from textwrap import dedent, indent
from typing import Any, Callable, Dict, List, Tuple, TypeVar, Union

from ..import_utils import optional_import_block, require_optional_import

with optional_import_block():
    import nbformat
    import yaml
    from nbclient.client import NotebookClient
    from nbclient.exceptions import (
        CellExecutionError,
        CellTimeoutError,
    )
    from nbformat import NotebookNode
    from termcolor import colored


@lru_cache
def check_quarto_bin(quarto_bin: str = "quarto") -> bool:
    """Check if quarto is installed."""
    try:
        version_str = subprocess.check_output([quarto_bin, "--version"], text=True).strip()
        version = tuple(map(int, version_str.split(".")))
        return version >= (1, 5, 23)

    except FileNotFoundError:
        return False


C = TypeVar("C", bound=Callable[..., Any])


def require_quarto_bin(f: C) -> C:
    """Decorator to skip a function if quarto is not installed."""

    if check_quarto_bin():
        return f
    else:

        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return ImportError("Quarto is not installed")

        return wrapper


class Result:
    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def notebooks_target_dir(website_directory: Path) -> Path:
    """Return the target directory for notebooks."""
    return website_directory / "notebooks"


def load_metadata(notebook: Path) -> dict[str, dict[str, Union[str, list[str], None]]]:
    content = json.load(notebook.open(encoding="utf-8"))
    metadata: dict[str, dict[str, Union[str, list[str], None]]] = content.get("metadata", {})
    return metadata


def skip_reason_or_none_if_ok(notebook: Path) -> Union[str, None, dict[str, Any]]:
    """Return a reason to skip the notebook, or None if it should not be skipped."""
    if notebook.suffix != ".ipynb":
        return "not a notebook"

    if not notebook.exists():
        return "file does not exist"

    # Extra checks for notebooks in the notebook directory
    if "notebook" not in notebook.parts:
        return None

    with open(notebook, encoding="utf-8") as f:
        content = f.read()

    # Load the json and get the first cell
    json_content = json.loads(content)
    first_cell = json_content["cells"][0]

    # <!-- and --> must exists on lines on their own
    if first_cell["cell_type"] == "markdown" and first_cell["source"][0].strip() == "<!--":
        raise ValueError(
            f"Error in {notebook.resolve()!s} - Front matter should be defined in the notebook metadata now."
        )

    metadata = load_metadata(notebook)

    if "skip_render" in metadata:
        return metadata["skip_render"]

    if "front_matter" not in metadata:
        return "front matter missing from notebook metadata ⚠️"

    front_matter = metadata["front_matter"]

    if "tags" not in front_matter:
        return "tags is not in front matter"

    if "description" not in front_matter:
        return "description is not in front matter"

    # Make sure tags is a list of strings
    if front_matter["tags"] is not None and not all([isinstance(tag, str) for tag in front_matter["tags"]]):
        return "tags must be a list of strings"

    # Make sure description is a string
    if not isinstance(front_matter["description"], str):
        return "description must be a string"

    return None


def extract_title(notebook: Path) -> str | None:
    """Extract the title of the notebook."""
    with open(notebook, encoding="utf-8") as f:
        content = f.read()

    # Load the json and get the first cell
    json_content = json.loads(content)
    first_cell = json_content["cells"][0]

    # find the # title
    for line in first_cell["source"]:
        if line.startswith("# "):
            title: str = line[2:].strip()
            # Strip off the { if it exists
            if "{" in title:
                title = title[: title.find("{")].strip()
            return title

    return None


@require_quarto_bin
@require_optional_import(["nbclient", "termcolor"], "docs")
def process_notebook(src_notebook: Path, website_dir: Path, notebook_dir: Path, quarto_bin: str, dry_run: bool) -> str:
    """Process a single notebook."""
    in_notebook_dir = "notebook" in src_notebook.parts

    metadata = load_metadata(src_notebook)

    title = extract_title(src_notebook)
    if title is None:
        return fmt_error(src_notebook, "Title not found in notebook")

    front_matter = {}
    if "front_matter" in metadata:
        front_matter = metadata["front_matter"]

    front_matter["title"] = title

    if in_notebook_dir:
        relative_notebook = src_notebook.resolve().relative_to(notebook_dir.resolve())
        dest_dir = notebooks_target_dir(website_directory=website_dir)
        target_file = dest_dir / relative_notebook.with_suffix(".mdx")
        intermediate_notebook = dest_dir / relative_notebook

        # If the intermediate_notebook already exists, check if it is newer than the source file
        if target_file.exists() and target_file.stat().st_mtime > src_notebook.stat().st_mtime:
            return fmt_skip(src_notebook, f"target file ({target_file.name}) is newer ☑️")

        if dry_run:
            return colored(f"Would process {src_notebook.name}", "green")

        # Copy notebook to target dir
        # The reason we copy the notebook is that quarto does not support rendering from a different directory
        shutil.copy(src_notebook, intermediate_notebook)

        # Check if another file has to be copied too
        # Solely added for the purpose of agent_library_example.json
        if "extra_files_to_copy" in metadata:
            for file in metadata["extra_files_to_copy"]:
                shutil.copy(src_notebook.parent / file, dest_dir / file)

        # Capture output
        result = subprocess.run([quarto_bin, "render", intermediate_notebook], capture_output=True, text=True)
        if result.returncode != 0:
            return fmt_error(
                src_notebook, f"Failed to render {src_notebook}\n\nstderr:\n{result.stderr}\nstdout:\n{result.stdout}"
            )

        # Unlink intermediate files
        intermediate_notebook.unlink()
    else:
        target_file = src_notebook.with_suffix(".mdx")

        # If the intermediate_notebook already exists, check if it is newer than the source file
        if target_file.exists() and target_file.stat().st_mtime > src_notebook.stat().st_mtime:
            return fmt_skip(src_notebook, f"target file ({target_file.name}) is newer ☑️")

        if dry_run:
            return colored(f"Would process {src_notebook.name}", "green")

        result = subprocess.run([quarto_bin, "render", src_notebook], capture_output=True, text=True)
        if result.returncode != 0:
            return fmt_error(
                src_notebook, f"Failed to render {src_notebook}\n\nstderr:\n{result.stderr}\nstdout:\n{result.stdout}"
            )

    post_process_mdx(target_file, src_notebook, front_matter, website_dir)

    return fmt_ok(src_notebook)


# Notebook execution based on nbmake: https://github.com/treebeardtech/nbmakes
@dataclass
class NotebookError:
    error_name: str
    error_value: str | None
    traceback: str
    cell_source: str


@dataclass
class NotebookSkip:
    reason: str


NB_VERSION = 4


@require_quarto_bin
@require_optional_import("nbclient", "docs")
def test_notebook(notebook_path: Path, timeout: int = 300) -> tuple[Path, NotebookError | NotebookSkip | None]:
    nb = nbformat.read(str(notebook_path), NB_VERSION)  # type: ignore

    if "skip_test" in nb.metadata:
        return notebook_path, NotebookSkip(reason=nb.metadata.skip_test)

    try:
        c = NotebookClient(
            nb,
            timeout=timeout,
            allow_errors=False,
            record_timing=True,
        )
        os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        with tempfile.TemporaryDirectory() as tempdir:
            c.execute(cwd=tempdir)
    except CellExecutionError:
        error = get_error_info(nb)
        assert error is not None
        return notebook_path, error
    except CellTimeoutError:
        error = get_timeout_info(nb)
        assert error is not None
        return notebook_path, error

    return notebook_path, None


# Find the first code cell which did not complete.
@require_optional_import("nbclient", "docs")
def get_timeout_info(
    nb: NotebookNode,
) -> NotebookError | None:
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        if "shell.execute_reply" not in cell.metadata.execution:
            return NotebookError(
                error_name="timeout",
                error_value="",
                traceback="",
                cell_source="".join(cell["source"]),
            )

    return None


@require_optional_import("nbclient", "docs")
def get_error_info(nb: NotebookNode) -> NotebookError | None:
    for cell in nb["cells"]:  # get LAST error
        if cell["cell_type"] != "code":
            continue
        errors = [output for output in cell["outputs"] if output["output_type"] == "error" or "ename" in output]

        if errors:
            traceback = "\n".join(errors[0].get("traceback", ""))
            return NotebookError(
                error_name=errors[0].get("ename", ""),
                error_value=errors[0].get("evalue", ""),
                traceback=traceback,
                cell_source="".join(cell["source"]),
            )
    return None


def add_front_matter_to_metadata_mdx(
    front_matter: dict[str, Union[str, list[str], None]], website_dir: Path, rendered_mdx: Path
) -> None:
    source = front_matter.get("source_notebook")
    if isinstance(source, str) and source.startswith("/website/docs/"):
        return

    metadata_mdx = website_dir / "snippets" / "data" / "NotebooksMetadata.mdx"

    if not metadata_mdx.exists():
        with open(metadata_mdx, "w", encoding="utf-8") as f:
            f.write(
                "{/*\nAuto-generated file - DO NOT EDIT\nPlease edit the add_front_matter_to_metadata_mdx function in process_notebooks.py\n*/}\n\n"
            )
            f.write("export const notebooksMetadata = [];\n")

    metadata = []
    with open(metadata_mdx, encoding="utf-8") as f:
        content = f.read()
        if content:
            start = content.find("export const notebooksMetadata = [")
            end = content.rfind("]")
            if start != -1 and end != -1:
                metadata = json.loads(content[start + 32 : end + 1])

    # Create new entry for current notebook
    entry = {
        "title": front_matter.get("title", ""),
        "link": f"/notebooks/{rendered_mdx.stem}",
        "description": front_matter.get("description", ""),
        "image": front_matter.get("image"),
        "tags": front_matter.get("tags", []),
        "source": source,
    }
    # Update metadata list
    existing_entry = next((item for item in metadata if item["title"] == entry["title"]), None)
    if existing_entry:
        metadata[metadata.index(existing_entry)] = entry
    else:
        metadata.append(entry)

    # Write metadata back to file
    with open(metadata_mdx, "w", encoding="utf-8") as f:
        f.write(
            "{/*\nAuto-generated file - DO NOT EDIT\nPlease edit the add_front_matter_to_metadata_mdx function in process_notebooks.py\n*/}\n\n"
        )
        f.write("export const notebooksMetadata = ")
        f.write(json.dumps(metadata, indent=4))
        f.write(";\n")


def convert_callout_blocks(content: str) -> str:
    """Converts callout blocks in the following formats:
    1) Plain callout blocks using ::: syntax.
    2) Blocks using 3-4 backticks + (mdx-code-block or {=mdx}) + ::: syntax.
    Transforms them into custom HTML/component syntax.
    """
    callout_types = {
        "tip": "Tip",
        "note": "Note",
        "warning": "Warning",
        "info": "Info",
        "info Requirements": "Info",
        "check": "Check",
        "danger": "Warning",
        "tabs": "Tabs",
    }

    # Regex explanation (using alternation):
    #
    # -- Alternative #1: Backticks + mdx-code-block/{=mdx} --
    #
    #   ^(?P<backticks>`{3,4})(?:mdx-code-block|\{=mdx\})[ \t]*\n
    #     - Matches opening backticks and optional mdx markers.
    #   :::(?P<callout_type_backtick>...)
    #     - Captures the callout type.
    #   (.*?)
    #     - Captures the content inside the callout.
    #   ^:::[ \t]*\n
    #     - Matches the closing ::: line.
    #   (?P=backticks)
    #     - Ensures the same number of backticks close the block.
    #
    # -- Alternative #2: Plain ::: callout --
    #
    #   ^:::(?P<callout_type_no_backtick>...)
    #     - Captures the callout type after :::.
    #   (.*?)
    #     - Captures the content inside the callout.
    #   ^:::
    #     - Matches the closing ::: line.
    #
    #  (?s)(?m): DOTALL + MULTILINE flags.
    #    - DOTALL (`.` matches everything, including newlines).
    #    - MULTILINE (`^` and `$` work at the start/end of each line).

    pattern = re.compile(
        r"(?s)(?m)"
        r"(?:"
        # Alternative #1: Backticks + mdx-code-block/{=mdx}
        r"^(?P<backticks>`{3,4})(?:mdx-code-block|\{=mdx\})[ \t]*\n"
        r":::(?P<callout_type_backtick>\w+(?:\s+\w+)?)[ \t]*\n"
        r"(?P<inner_backtick>.*?)"
        r"^:::[ \t]*\n"
        r"(?P=backticks)"  # Closing backticks must match the opening count.
        r")"
        r"|"
        # Alternative #2: Plain ::: callout
        r"(?:"
        r"^:::(?P<callout_type_no_backtick>\w+(?:\s+\w+)?)[ \t]*\n"
        r"(?P<inner_no_backtick>.*?)"
        r"^:::[ \t]*(?:\n|$)"
        r")"
    )

    def replace_callout(m: re.Match[str]) -> str:
        # Determine the matched alternative and extract the corresponding groups.
        ctype = m.group("callout_type_backtick") or m.group("callout_type_no_backtick")
        inner = m.group("inner_backtick") or m.group("inner_no_backtick") or ""

        # Map the callout type to its standard representation or fallback to the original type.
        mapped_type = callout_types.get(ctype, ctype)

        # Return the formatted HTML block.
        return f"""
<div class="{ctype}">
<{mapped_type}>
{inner.strip()}
</{mapped_type}>
</div>
"""

    # Apply the regex pattern and replace matched callouts with the custom HTML structure.
    return pattern.sub(replace_callout, content)


def convert_mdx_image_blocks(content: str, rendered_mdx: Path, website_dir: Path) -> str:
    """Converts MDX code block image syntax to regular markdown image syntax.

    Args:
        content (str): The markdown content containing mdx-code-block image syntax

    Returns:
        str: The converted markdown content with standard image syntax
    """

    def resolve_path(match: re.Match[str]) -> str:
        img_pattern = r"!\[(.*?)\]\((.*?)\)"
        img_match = re.search(img_pattern, match.group(1))
        if not img_match:
            return match.group(0)

        alt, rel_path = img_match.groups()
        abs_path = (rendered_mdx.parent / Path(rel_path)).resolve().relative_to(website_dir)
        return f"![{alt}](/{abs_path})"

    pattern = r"````mdx-code-block\n(!\[.*?\]\(.*?\))\n````"
    return re.sub(pattern, resolve_path, content)


# rendered_notebook is the final mdx file
def post_process_mdx(
    rendered_mdx: Path, source_notebooks: Path, front_matter: dict[str, Union[str, list[str], None]], website_dir: Path
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

    # Is there a title on the content? Only search up until the first code cell
    # first_code_cell = content.find("```")
    # if first_code_cell != -1:
    #     title_search_content = content[:first_code_cell]
    # else:
    #     title_search_content = content

    # title_exists = title_search_content.find("\n# ") != -1
    # if not title_exists:
    #     content = f"# {front_matter['title']}\n{content}"
    # inject in content directly after the markdown title the word done
    # Find the end of the line with the title
    # title_end = content.find("\n", content.find("#"))

    # Extract page title
    # title = content[content.find("#") + 1 : content.find("\n", content.find("#"))].strip()
    # If there is a { in the title we trim off the { and everything after it
    # if "{" in title:
    #     title = title[: title.find("{")].strip()

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
    add_front_matter_to_metadata_mdx(front_matter, website_dir, rendered_mdx)

    # Dump front_matter to ysaml
    front_matter_str = yaml.dump(front_matter, default_flow_style=False)

    # Convert callout blocks
    content = convert_callout_blocks(content)

    # Convert mdx image syntax to mintly image syntax
    content = convert_mdx_image_blocks(content, rendered_mdx, website_dir)

    # Rewrite the content as
    # ---
    # front_matter_str
    # ---
    # content
    new_content = f"---\n{front_matter_str}---\n{content}"
    with open(rendered_mdx, "w", encoding="utf-8") as f:
        f.write(new_content)


def path(path_str: str) -> Path:
    """Return a Path object."""
    return Path(path_str)


def collect_notebooks(notebook_directory: Path, website_directory: Path) -> list[Path]:
    notebooks = list(notebook_directory.glob("*.ipynb"))
    notebooks.extend(list(website_directory.glob("docs/**/*.ipynb")))
    return notebooks


@require_optional_import("termcolor", "docs")
def fmt_skip(notebook: Path, reason: str) -> str:
    return f"{colored('[Skip]', 'yellow')} {colored(notebook.name, 'blue')}: {reason}"


@require_optional_import("termcolor", "docs")
def fmt_ok(notebook: Path) -> str:
    return f"{colored('[OK]', 'green')} {colored(notebook.name, 'blue')} ✅"


@require_optional_import("termcolor", "docs")
def fmt_error(notebook: Path, error: NotebookError | str) -> str:
    if isinstance(error, str):
        return f"{colored('[Error]', 'red')} {colored(notebook.name, 'blue')}: {error}"
    elif isinstance(error, NotebookError):
        return f"{colored('[Error]', 'red')} {colored(notebook.name, 'blue')}: {error.error_name} - {error.error_value}"
    else:
        raise ValueError("error must be a string or a NotebookError")


def start_thread_to_terminate_when_parent_process_dies(ppid: int) -> None:
    pid = os.getpid()

    def f() -> None:
        while True:
            try:
                os.kill(ppid, 0)
            except OSError:
                os.kill(pid, signal.SIGTERM)
            time.sleep(1)

    thread = threading.Thread(target=f, daemon=True)
    thread.start()


def copy_examples_mdx_files(website_dir: str) -> None:
    # The mdx files to copy to the notebooks directory
    example_section_mdx_files = ["Gallery", "Notebooks"]

    # Create notebooks directory if it doesn't exist
    website_dir_path = Path(website_dir)
    notebooks_dir = website_dir_path / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    for mdx_file in example_section_mdx_files:
        src_mdx_file_path = (website_dir_path / "docs" / f"{mdx_file}.mdx").resolve()
        dest_mdx_file_path = (notebooks_dir / f"{mdx_file}.mdx").resolve()
        # Copy mdx file to notebooks directory
        shutil.copy(src_mdx_file_path, dest_mdx_file_path)


def get_sorted_files(input_dir: Path, prefix: str) -> list[str]:
    """Get sorted list of files with prefix prepended."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    # Sort files by parent directory date (if exists) and name
    def sort_key(file_path: Path) -> Tuple[datetime, str]:
        dirname = file_path.parent.name
        try:
            # Extract date from directory name (first 3 parts)
            date_str = "-".join(dirname.split("-")[:3])
            date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            date = datetime.min
        return (date, dirname)

    files = sorted(input_dir.glob("**/index.mdx"), key=sort_key)
    reversed_files = files[::-1]

    return [f"{prefix}/{f.parent.relative_to(input_dir)}/index".replace("\\", "/") for f in reversed_files]


def generate_nav_group(input_dir: Path, group_header: str, prefix: str) -> Dict[str, Union[str, List[str]]]:
    """Generate navigation group for a directory.

    Args:
        input_dir (Path): Directory to process
        group_header (str): Group header
    """
    sorted_dir_files = get_sorted_files(input_dir, prefix)

    return {"group": group_header, "pages": sorted_dir_files}


def extract_example_group(metadata_path: Path) -> dict[str, Sequence[Collection[str]]]:
    # Read NotebooksMetadata.mdx and extract metadata links
    with open(metadata_path, encoding="utf-8") as f:
        content = f.read()
        # Extract the array between the brackets
        start = content.find("export const notebooksMetadata = [")
        end = content.rfind("]")
        if start == -1 or end == -1:
            print("Could not find notebooksMetadata in the file")
            return {}
        metadata_str = content[start + 32 : end + 1]
        notebooks_metadata = json.loads(metadata_str)

    # Create notebooks entry
    notebooks_entry = {
        "group": "Examples by Notebook",
        "pages": ["notebooks/Notebooks"]
        + [
            Path(item["source"])
            .with_suffix("")
            .as_posix()
            .replace("/website/", "/")
            .replace("/notebook/", "notebooks/")
            for item in notebooks_metadata
            if not item["source"].startswith("/website/docs/")
        ],
    }

    example_group = {"group": "Examples", "pages": [notebooks_entry, "notebooks/Gallery"]}

    return example_group


def update_navigation_with_notebooks(website_dir: Path) -> None:
    """Updates mint.json navigation to include notebook entries from NotebooksMetadata.mdx.

    Args:
        website_dir (Path): Root directory of the website
    """
    mint_json_path = (website_dir / "mint.json").resolve()
    metadata_path = (website_dir / "snippets" / "data" / "NotebooksMetadata.mdx").resolve()

    if not mint_json_path.exists():
        print(f"mint.json not found at {mint_json_path}")
        return

    if not metadata_path.exists():
        print(f"NotebooksMetadata.mdx not found at {metadata_path}")
        return

    # Read mint.json
    with open(mint_json_path, encoding="utf-8") as f:
        mint_config = json.load(f)

    # add talks to navigation
    talks_dir = website_dir / "talks"
    talks_section = generate_nav_group(talks_dir, "Talks", "talks")
    talks_section_pages = (
        [talks_section["pages"]] if isinstance(talks_section["pages"], str) else talks_section["pages"]
    )

    # Add "talks/future_talks/index" item at the beginning of the list
    future_talks_index = talks_section_pages.pop()
    talks_section_pages.insert(0, future_talks_index)
    mint_config["navigation"].append(talks_section)

    # add blogs to navigation
    blogs_dir = website_dir / "_blogs"
    blog_section = {"group": "Blog", "pages": [generate_nav_group(blogs_dir, "Recent posts", "blog")]}
    mint_config["navigation"].append(blog_section)

    # Add examples to navigation
    example_group = extract_example_group(metadata_path)
    mint_config["navigation"].append(example_group)

    # Write back to mint.json
    with open(mint_json_path, "w", encoding="utf-8") as f:
        json.dump(mint_config, f, indent=2)
        f.write("\n")

    print(f"Updated navigation in {mint_json_path}")


def fix_internal_references(content: str, root_path: Path, current_file_path: Path) -> str:
    """Resolves internal markdown references relative to root_dir and returns fixed content.

    Args:
        content: Markdown content to fix
        root_path: Root directory for resolving paths
        current_file_path: Path of the current file being processed
    """

    def resolve_link(match: re.Match[str]) -> str:
        display_text, raw_path = match.groups()
        try:
            path_parts = raw_path.split("#")
            rel_path = path_parts[0]
            anchor = f"#{path_parts[1]}" if len(path_parts) > 1 else ""

            resolved = (current_file_path.parent / rel_path).resolve()
            final_path = (resolved.relative_to(root_path.resolve())).with_suffix("")

            return f"[{display_text}](/{final_path}{anchor})"
        except Exception:
            return match.group(0)

    pattern = r"\[([^\]]+)\]\(((?:\.\./|\./)?\w+(?:/[\w-]+)*\.md(?:#[\w-]+)?)\)"
    return re.sub(pattern, resolve_link, content)


def fix_internal_references_in_mdx_files(website_dir: Path) -> None:
    """Process all MDX files in directory to fix internal references."""
    for file_path in website_dir.glob("**/*.mdx"):
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            fixed_content = fix_internal_references(content, website_dir, file_path)

            if content != fixed_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(fixed_content)
                print(f"Fixed internal references in {file_path}")

        except Exception:
            print(f"Error: {file_path}")
            sys.exit(1)


def construct_authors_html(authors_list: List[str], authors_dict: Dict[str, Dict[str, str]]) -> str:
    """Constructs HTML for displaying author cards in a blog.

    Args:
        authors_list: List of author identifiers
        authors_dict: Dictionary containing author information keyed by author identifier
    Returns:
        str: Formatted HTML string containing author cards
    """
    if not authors_list:
        return ""

    card_template = """
        <Card href="{url}">
            <div class="col card">
              <div class="img-placeholder">
                <img noZoom src="{image_url}" />
              </div>
              <div>
                <p class="name">{name}</p>
                <p>{title}</p>
              </div>
            </div>
        </Card>"""

    authors_html = [card_template.format(**authors_dict[author]) for author in authors_list]

    author_label = "Author:" if len(authors_list) == 1 else "Authors:"
    authors_html_str = indent("".join(authors_html), "        ")
    retval = dedent(
        f"""
            <div class="blog-authors">
              <p class="authors">{author_label}</p>
              <CardGroup cols={{2}}>{authors_html_str}
              </CardGroup>
            </div>
        """
    )
    return retval


def separate_front_matter_and_content(file_path: Path) -> Tuple[str, str]:
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


def add_authors_and_social_img_to_blog_posts(website_dir: Path) -> None:
    """Add authors info to blog posts.

    Args:
        website_dir (Path): Root directory of the website
    """
    blog_dir = website_dir / "_blogs"
    authors_yml = blog_dir / "authors.yml"
    generated_blog_dir = website_dir / "blog"

    # Remove existing generated directory if it exists
    if generated_blog_dir.exists():
        shutil.rmtree(generated_blog_dir)

    # Copy entire blog directory structure to generated_blog
    shutil.copytree(blog_dir, generated_blog_dir)

    try:
        all_authors_info = yaml.safe_load(authors_yml.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError) as e:
        print(f"Error reading authors file: {e}")
        sys.exit(1)

    for file_path in generated_blog_dir.glob("**/*.mdx"):
        try:
            front_matter_string, content = separate_front_matter_and_content(file_path)

            # Skip if authors section already exists
            # if '<div class="blog-authors">' in content:
            #     continue

            # Convert single author to list and handle authors
            front_matter = yaml.safe_load(front_matter_string[4:-3])
            authors = front_matter.get("authors", [])
            authors_list = [authors] if isinstance(authors, str) else authors

            # Social share image
            social_img_html = """\n<div>
<img noZoom className="social-share-img"
  src="https://media.githubusercontent.com/media/ag2ai/ag2/refs/heads/main/website/static/img/cover.png"
  alt="social preview"
  style={{ position: 'absolute', left: '-9999px' }}
/>
</div>"""

            # Generate authors HTML
            authors_html = (
                construct_authors_html(authors_list, all_authors_info)
                if '<div class="blog-authors">' not in content
                else ""
            )

            # Combine content
            new_content = f"{front_matter_string}\n{social_img_html}\n{authors_html}\n{content}"

            file_path.write_text(f"{new_content}\n", encoding="utf-8")
            print(f"Authors info and social share image checked in {file_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue


def ensure_mint_json_exists(website_dir: Path) -> None:
    mint_json_path = website_dir / "mint.json"
    if not mint_json_path.exists():
        print(f"mint.json not found at {mint_json_path}")
        print(
            "You can either run the 'generate_api_references.py' script before running this script or simply run the scripts/docs_build.sh script which will execute both 'generate_api_references.py' and 'process_notebooks.py' scripts in correct order."
        )
        sys.exit(1)


def cleanup_tmp_dirs_if_no_metadata(website_dir: Path) -> None:
    """Remove the temporary notebooks directory if NotebooksMetadata.mdx is not found.

    This is to ensure a clean build and generate the metadata file as well as to
    update the navigation with correct entries.
    """
    metadata_mdx = website_dir / "snippets" / "data" / "NotebooksMetadata.mdx"
    if not metadata_mdx.exists():
        print(f"NotebooksMetadata.mdx not found at {metadata_mdx}")

        notebooks_dir = website_dir / "notebooks"
        print(f"Removing the {notebooks_dir} and to ensure a clean build.")
        shutil.rmtree(notebooks_dir, ignore_errors=True)


def main() -> None:
    root_dir = Path(__file__).resolve().parents[2]
    website_dir = root_dir / "website"
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")

    parser.add_argument(
        "--notebook-directory",
        type=path,
        help="Directory containing notebooks to process",
        default=website_dir / "../notebook",
    )
    parser.add_argument(
        "--website-directory", type=path, help="Root directory of mintlify website", default=website_dir
    )

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--quarto-bin", help="Path to quarto binary", default="quarto")
    render_parser.add_argument("--dry-run", help="Don't render", action="store_true")
    render_parser.add_argument("notebooks", type=path, nargs="*", default=None)

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--timeout", help="Timeout for each notebook", type=int, default=60)
    test_parser.add_argument("--exit-on-first-fail", "-e", help="Exit after first test fail", action="store_true")
    test_parser.add_argument("notebooks", type=path, nargs="*", default=None)
    test_parser.add_argument("--workers", help="Number of workers to use", type=int, default=-1)

    args = parser.parse_args()
    if args.subcommand is None:
        print("No subcommand specified")
        sys.exit(1)

    ensure_mint_json_exists(args.website_directory)
    cleanup_tmp_dirs_if_no_metadata(args.website_directory)

    if args.notebooks:
        collected_notebooks = args.notebooks
    else:
        collected_notebooks = collect_notebooks(args.notebook_directory, args.website_directory)

    filtered_notebooks = []
    for notebook in collected_notebooks:
        reason = skip_reason_or_none_if_ok(notebook)
        if reason and isinstance(reason, str):
            print(fmt_skip(notebook, reason))
        else:
            filtered_notebooks.append(notebook)

    if args.subcommand == "test":
        if args.workers == -1:
            args.workers = None
        failure = False
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=start_thread_to_terminate_when_parent_process_dies,
            initargs=(os.getpid(),),
        ) as executor:
            futures = [executor.submit(test_notebook, f, args.timeout) for f in filtered_notebooks]
            for future in concurrent.futures.as_completed(futures):
                notebook, optional_error_or_skip = future.result()
                if isinstance(optional_error_or_skip, NotebookError):
                    if optional_error_or_skip.error_name == "timeout":
                        print(fmt_error(notebook, optional_error_or_skip.error_name))

                    else:
                        print("-" * 80)

                        print(fmt_error(notebook, optional_error_or_skip))
                        print(optional_error_or_skip.traceback)
                        print("-" * 80)
                    if args.exit_on_first_fail:
                        sys.exit(1)
                    failure = True
                elif isinstance(optional_error_or_skip, NotebookSkip):
                    print(fmt_skip(notebook, optional_error_or_skip.reason))
                else:
                    print(fmt_ok(notebook))

        if failure:
            sys.exit(1)

    elif args.subcommand == "render":
        check_quarto_bin(args.quarto_bin)

        if not notebooks_target_dir(args.website_directory).exists():
            notebooks_target_dir(args.website_directory).mkdir(parents=True)

        for notebook in filtered_notebooks:
            print(
                process_notebook(
                    notebook, args.website_directory, args.notebook_directory, args.quarto_bin, args.dry_run
                )
            )

        # Post-processing steps after all notebooks are handled
        if not args.dry_run:
            copy_examples_mdx_files(args.website_directory)
            update_navigation_with_notebooks(args.website_directory)
            fix_internal_references_in_mdx_files(args.website_directory)
            add_authors_and_social_img_to_blog_posts(args.website_directory)

    else:
        print("Unknown subcommand")
        sys.exit(1)
