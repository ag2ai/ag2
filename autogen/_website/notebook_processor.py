# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


import argparse
import concurrent.futures
import functools
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union

from ..import_utils import optional_import_block, require_optional_import

with optional_import_block():
    import nbformat
    from nbclient.client import NotebookClient
    from nbclient.exceptions import (
        CellExecutionError,
        CellTimeoutError,
    )
    from nbformat import NotebookNode
    from termcolor import colored


# Notebook execution based on nbmake: https://github.com/treebeardtech/nbmakes
@dataclass
class NotebookError:
    error_name: str
    error_value: Optional[str]
    traceback: str
    cell_source: str


@dataclass
class NotebookSkip:
    reason: str


NB_VERSION = 4


class Result:
    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def path(path_str: str) -> Path:
    """Return a Path object."""
    return Path(path_str)


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

        return wrapper  # type: ignore[return-value]


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


def extract_title(notebook: Path) -> Optional[str]:
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


@require_optional_import("termcolor", "docs")
def fmt_skip(notebook: Path, reason: str) -> str:
    return f"{colored('[Skip]', 'yellow')} {colored(notebook.name, 'blue')}: {reason}"


@require_optional_import("termcolor", "docs")
def fmt_ok(notebook: Path) -> str:
    return f"{colored('[OK]', 'green')} {colored(notebook.name, 'blue')} ✅"


@require_optional_import("termcolor", "docs")
def fmt_error(notebook: Path, error: Union[NotebookError, str]) -> str:
    if isinstance(error, str):
        return f"{colored('[Error]', 'red')} {colored(notebook.name, 'blue')}: {error}"
    elif isinstance(error, NotebookError):
        return f"{colored('[Error]', 'red')} {colored(notebook.name, 'blue')}: {error.error_name} - {error.error_value}"
    else:
        raise ValueError("error must be a string or a NotebookError")


@require_quarto_bin
@require_optional_import("nbclient", "docs")
def test_notebook(notebook_path: Path, timeout: int = 300) -> tuple[Path, Optional[Union[NotebookError, NotebookSkip]]]:
    nb = nbformat.read(str(notebook_path), NB_VERSION)  # type: ignore[arg-type,no-untyped-call]

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
) -> Optional[NotebookError]:
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
def get_error_info(nb: NotebookNode) -> Optional[NotebookError]:
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


def collect_notebooks(notebook_directory: Path, website_build_directory: Path) -> list[Path]:
    notebooks = list(notebook_directory.glob("*.ipynb"))
    notebooks.extend(list(website_build_directory.glob("docs/**/*.ipynb")))
    return notebooks


@require_quarto_bin
@require_optional_import(["nbclient", "termcolor"], "docs")
def process_notebook(
    src_notebook: Path,
    website_build_directory: Path,
    notebook_dir: Path,
    quarto_bin: str,
    dry_run: bool,
    target_dir_func: Callable[[Path], Path],
    post_processor: Optional[Callable[[Path, Path, dict[str, Any], Path], None]] = None,
) -> str:
    """Process a single notebook.

    Args:
        src_notebook: Source notebook path
        website_build_directory: Output directory
        notebook_dir: Base notebooks directory
        quarto_bin: Path to quarto binary
        dry_run: If True, don't actually process
        target_dir_func: Function to determine target directory for notebooks
        post_processor: Optional callback for post-processing
    """

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
        dest_dir = target_dir_func(website_build_directory)
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

    # Use post-processor if provided
    if post_processor and not dry_run:
        post_processor(target_file, src_notebook, front_matter, website_build_directory)

    return fmt_ok(src_notebook)


def create_base_argument_parser() -> argparse.ArgumentParser:
    """Create the base argument parser with common options."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")

    parser.add_argument(
        "--notebook-directory",
        type=path,
        help="Directory containing notebooks to process",
    )
    parser.add_argument("--website-build-directory", type=path, help="Root directory of website build")
    parser.add_argument("--force", help="Force re-rendering of all notebooks", action="store_true", default=False)

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--quarto-bin", help="Path to quarto binary", default="quarto")
    render_parser.add_argument("--dry-run", help="Don't render", action="store_true")
    render_parser.add_argument("notebooks", type=path, nargs="*", default=None)

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--timeout", help="Timeout for each notebook", type=int, default=60)
    test_parser.add_argument("--exit-on-first-fail", "-e", help="Exit after first test fail", action="store_true")
    test_parser.add_argument("notebooks", type=path, nargs="*", default=None)
    test_parser.add_argument("--workers", help="Number of workers to use", type=int, default=-1)

    return parser


def process_notebooks_core(
    args: argparse.Namespace,
    post_process_func: Optional[Callable[[Path, Path, dict[str, Any], Path], None]],
    target_dir_func: Callable[[Path], Path],
) -> list[Path]:
    """Core logic for processing notebooks shared across build systems.

    Args:
        args: Command line arguments
        post_process_func: Function for post-processing rendered notebooks
        target_dir_func: Function to determine target directory for notebooks
    """
    collected_notebooks = (
        args.notebooks if args.notebooks else collect_notebooks(args.notebook_directory, args.website_build_directory)
    )

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

        target_dir = target_dir_func(args.website_build_directory)
        if not target_dir.exists():
            target_dir.mkdir(parents=True)

        for notebook in filtered_notebooks:
            print(
                process_notebook(
                    notebook,
                    args.website_build_directory,
                    args.notebook_directory,
                    args.quarto_bin,
                    args.dry_run,
                    target_dir_func,
                    post_process_func,
                )
            )

    return filtered_notebooks
