# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
#!/usr/bin/env python

from __future__ import annotations

import argparse
import concurrent.futures
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
import typing
from dataclasses import dataclass
from multiprocessing import current_process
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from termcolor import colored

try:
    import yaml
except ImportError:
    print("pyyaml not found.\n\nPlease install pyyaml:\n\tpip install pyyaml\n")
    sys.exit(1)

try:
    import nbclient
    from nbclient.client import (
        CellExecutionError,
        CellTimeoutError,
        NotebookClient,
    )
except ImportError:
    if current_process().name == "MainProcess":
        print("nbclient not found.\n\nPlease install nbclient:\n\tpip install nbclient\n")
        print("test won't work without nbclient")

try:
    import nbformat
    from nbformat import NotebookNode
except ImportError:
    if current_process().name == "MainProcess":
        print("nbformat not found.\n\nPlease install nbformat:\n\tpip install nbformat\n")
        print("test won't work without nbclient")


class Result:
    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def check_quarto_bin(quarto_bin: str = "quarto") -> None:
    """Check if quarto is installed."""
    try:
        version = subprocess.check_output([quarto_bin, "--version"], text=True).strip()
        version = tuple(map(int, version.split(".")))
        if version < (1, 5, 23):
            print("Quarto version is too old. Please upgrade to 1.5.23 or later.")
            sys.exit(1)

    except FileNotFoundError:
        print("Quarto is not installed. Please install it from https://quarto.org")
        sys.exit(1)


def notebooks_target_dir(website_directory: Path) -> Path:
    """Return the target directory for notebooks."""
    return website_directory / "notebooks"


def load_metadata(notebook: Path) -> dict:
    content = json.load(notebook.open(encoding="utf-8"))
    return content["metadata"]


def skip_reason_or_none_if_ok(notebook: Path) -> str | None:
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
            f"Error in {str(notebook.resolve())} - Front matter should be defined in the notebook metadata now."
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
    if not all([isinstance(tag, str) for tag in front_matter["tags"]]):
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
            title = line[2:].strip()
            # Strip off the { if it exists
            if "{" in title:
                title = title[: title.find("{")].strip()
            return title

    return None


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
        if target_file.exists():
            if target_file.stat().st_mtime > src_notebook.stat().st_mtime:
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
        if target_file.exists():
            if target_file.stat().st_mtime > src_notebook.stat().st_mtime:
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


def test_notebook(notebook_path: Path, timeout: int = 300) -> tuple[Path, NotebookError | NotebookSkip | None]:
    nb = nbformat.read(str(notebook_path), NB_VERSION)

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
    front_matter: dict[str, str | list[str]], website_dir: Path, rendered_mdx: Path
) -> None:
    metadata_mdx = website_dir / "snippets" / "data" / "NotebooksMetadata.mdx"

    metadata = []
    if metadata_mdx.exists():
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
        "source": front_matter.get("source_notebook"),
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
    """
    Converts callout blocks in the following formats:
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

    def replace_callout(m: re.Match) -> str:
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
    """
    Converts MDX code block image syntax to regular markdown image syntax.

    Args:
        content (str): The markdown content containing mdx-code-block image syntax

    Returns:
        str: The converted markdown content with standard image syntax
    """

    def resolve_path(match):
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
def post_process_mdx(rendered_mdx: Path, source_notebooks: Path, front_matter: dict, website_dir: Path) -> None:
    with open(rendered_mdx, encoding="utf-8") as f:
        content = f.read()

    # If there is front matter in the mdx file, we need to remove it
    if content.startswith("---"):
        front_matter_end = content.find("---", 3)
        front_matter = yaml.safe_load(content[4:front_matter_end])
        content = content[front_matter_end + 3 :]

    # Clean heading IDs using regex - matches from # to the end of ID block
    content = re.sub(r"(#{1,6}[^{]+){#[^}]+}", r"\1", content)

    # Each intermediate path needs to be resolved for this to work reliably
    repo_root = Path(__file__).parent.resolve().parent.resolve()
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
    front_matter = yaml.dump(front_matter, default_flow_style=False)

    # Convert callout blocks
    content = convert_callout_blocks(content)

    # Convert mdx image syntax to mintly image syntax
    content = convert_mdx_image_blocks(content, rendered_mdx, website_dir)

    # Rewrite the content as
    # ---
    # front_matter
    # ---
    # content
    new_content = f"---\n{front_matter}---\n{content}"
    with open(rendered_mdx, "w", encoding="utf-8") as f:
        f.write(new_content)


def path(path_str: str) -> Path:
    """Return a Path object."""
    return Path(path_str)


def collect_notebooks(notebook_directory: Path, website_directory: Path) -> list[Path]:
    notebooks = list(notebook_directory.glob("*.ipynb"))
    notebooks.extend(list(website_directory.glob("docs/**/*.ipynb")))
    return notebooks


def fmt_skip(notebook: Path, reason: str) -> str:
    return f"{colored('[Skip]', 'yellow')} {colored(notebook.name, 'blue')}: {reason}"


def fmt_ok(notebook: Path) -> str:
    return f"{colored('[OK]', 'green')} {colored(notebook.name, 'blue')} ✅"


def fmt_error(notebook: Path, error: NotebookError | str) -> str:
    if isinstance(error, str):
        return f"{colored('[Error]', 'red')} {colored(notebook.name, 'blue')}: {error}"
    elif isinstance(error, NotebookError):
        return f"{colored('[Error]', 'red')} {colored(notebook.name, 'blue')}: {error.error_name} - {error.error_value}"
    else:
        raise ValueError("error must be a string or a NotebookError")


def start_thread_to_terminate_when_parent_process_dies(ppid: int):
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
    example_section_mdx_files = ["Examples", "Gallery", "Notebooks"]

    # Create notebooks directory if it doesn't exist
    website_dir = Path(website_dir)
    notebooks_dir = website_dir / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    for mdx_file in example_section_mdx_files:
        src_mdx_file_path = (website_dir / "docs" / f"{mdx_file}.mdx").resolve()
        dest_mdx_file_path = (notebooks_dir / f"{mdx_file}.mdx").resolve()
        # Copy mdx file to notebooks directory
        shutil.copy(src_mdx_file_path, dest_mdx_file_path)


def update_navigation_with_notebooks(website_dir: Path) -> None:
    """
    Updates mint.json navigation to include notebook entries from NotebooksMetadata.mdx.

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

    # Read NotebooksMetadata.mdx and extract metadata links
    with open(metadata_path, encoding="utf-8") as f:
        content = f.read()
        # Extract the array between the brackets
        start = content.find("export const notebooksMetadata = [")
        end = content.rfind("]")
        if start == -1 or end == -1:
            print("Could not find notebooksMetadata in the file")
            return
        metadata_str = content[start + 32 : end + 1]
        notebooks_metadata = json.loads(metadata_str)

    # Find the Examples group in navigation
    examples_group = None
    for group in mint_config["navigation"]:
        if group.get("group") == "Examples":
            examples_group = group
            break

    if examples_group is None:
        print("Examples group not found in navigation")
        return

    # Create notebooks entry
    notebooks_entry = {
        "group": "Examples by Notebook",
        "pages": ["notebooks/Notebooks"]
        + [
            Path(item["source"])
            .resolve()
            .with_suffix("")
            .as_posix()
            .replace("/website/", "/")
            .replace("/notebook/", "notebooks/")
            for item in notebooks_metadata
            if not item["source"].startswith("/website/docs/")
        ],
    }

    # Replace the pages list in Examples group with our standard pages plus notebooks
    examples_group["pages"] = ["notebooks/Examples", notebooks_entry, "notebooks/Gallery"]

    # Write back to mint.json
    with open(mint_json_path, "w", encoding="utf-8") as f:
        json.dump(mint_config, f, indent=2)
        f.write("\n")

    print(f"Updated navigation in {mint_json_path}")


def fix_internal_references(content: str, root_path: Path, current_file_path: Path) -> str:
    """
    Resolves internal markdown references relative to root_dir and returns fixed content.

    Args:
        content: Markdown content to fix
        root_path: Root directory for resolving paths
        current_file_path: Path of the current file being processed
    """

    def resolve_link(match):
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


def main() -> None:
    script_dir = Path(__file__).parent.absolute()
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")

    parser.add_argument(
        "--notebook-directory",
        type=path,
        help="Directory containing notebooks to process",
        default=script_dir / "../notebook",
    )
    parser.add_argument(
        "--website-directory", type=path, help="Root directory of docusarus website", default=script_dir
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

    if args.notebooks:
        collected_notebooks = args.notebooks
    else:
        collected_notebooks = collect_notebooks(args.notebook_directory, args.website_directory)

    filtered_notebooks = []
    for notebook in collected_notebooks:
        reason = skip_reason_or_none_if_ok(notebook)
        if reason:
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

    else:
        print("Unknown subcommand")
        sys.exit(1)


if __name__ == "__main__":
    main()
