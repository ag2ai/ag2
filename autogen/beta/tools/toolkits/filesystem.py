# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

from pydantic import Field

from autogen.beta.tools.final import Toolkit, tool
from autogen.beta.tools.final.function_tool import FunctionTool


class FilesystemToolset(Toolkit):
    """Toolkit that exposes local filesystem operations as function tools.

    All paths are resolved relative to *base_path*.  A path-traversal guard
    rejects any resolved path that escapes the base directory.

    Individual tools are available as attributes and can be passed to an agent
    separately::

        fs = FilesystemToolset(base_path="/tmp/workspace")

        # use the full toolset
        agent = Agent("a", config=config, tools=[fs])

        # or pick individual tools
        agent = Agent("a", config=config, tools=[fs.read_file, fs.find_files])
    """

    read_file: FunctionTool
    find_files: FunctionTool
    write_file: FunctionTool
    update_file: FunctionTool
    delete_file: FunctionTool

    def __init__(
        self,
        base_path: str | Path = ".",
        *,
        read_only: bool = False,
    ) -> None:
        base = Path(base_path).resolve()

        self.read_file = _make_read_file(base)
        self.find_files = _make_find_files(base)
        self.write_file = _make_write_file(base)
        self.update_file = _make_update_file(base)
        self.delete_file = _make_delete_file(base)

        tools = [self.read_file, self.find_files]
        if not read_only:
            tools.extend([self.write_file, self.update_file, self.delete_file])
        super().__init__(*tools)


def _make_read_file(base: Path) -> FunctionTool:
    @tool(
        description="Read the contents of a file. Returns text by default. Set raw=true to read binary content as base64."
    )
    def read_file(
        path: Annotated[str, Field(description="Relative path to the file to read.")],
        raw: bool = Field(
            default=False, description="If true, read the file as binary and return base64-encoded content."
        ),
    ) -> str:
        target = _resolve_path(base, path)
        if raw:
            return base64.b64encode(target.read_bytes()).decode("ascii")
        return target.read_text()

    return read_file


def _make_find_files(base: Path) -> FunctionTool:
    @tool(
        description="Search for files matching a glob pattern. Use recursive patterns like '**/*.py' to search subdirectories."
    )
    def find_files(
        pattern: Annotated[str, Field(description="Glob pattern to match files, e.g. '*.py' or '**/*.txt'.")],
        path: str = Field(
            default=".", description="Relative path to the directory to search from. Defaults to the base directory."
        ),
    ) -> list[str]:
        target = _resolve_path(base, path)
        return sorted(str(p.relative_to(base)) for p in _glob(target, pattern))

    return find_files


def _glob_files(target: Path, pattern: str) -> Iterable[Path]:
    """Yield files matching *pattern* under *target* with consistent behavior across Python versions."""
    return (p for p in target.glob(pattern) if p.is_file())


def _glob_files_compat(target: Path, pattern: str) -> Iterable[Path]:
    """Yield files matching *pattern* under *target*.

    Backfills Python 3.13 ``**`` semantics for older interpreters:
    - ``**`` matches zero directory segments (e.g. ``**/*.py`` matches root-level ``.py`` files).
    - A trailing ``**`` matches files, not only directories.
    """
    results = {p for p in target.glob(pattern) if p.is_file()}
    if "**" in pattern:
        prefix, suffix = pattern.split("**", 1)
        suffix = suffix.lstrip("/")
        prefix_dir = target / prefix if prefix else target
        if prefix_dir.is_dir():
            if suffix:
                results.update(p for p in prefix_dir.glob(suffix) if p.is_file())
            else:
                results.update(p for p in prefix_dir.rglob("*") if p.is_file())
    return results


_glob = _glob_files if sys.version_info >= (3, 13) else _glob_files_compat


def _make_write_file(base: Path) -> FunctionTool:
    @tool(
        description="Create or overwrite a file with the given content. Parent directories are created automatically."
    )
    def write_file(
        path: Annotated[str, Field(description="Relative path to the file to create or overwrite.")],
        content: Annotated[str, Field(description="The full content to write to the file.")],
    ) -> str:
        target = _resolve_path(base, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"Successfully wrote {len(content)} characters to {path}"

    return write_file


def _make_update_file(base: Path) -> FunctionTool:
    @tool(description="Update a file by replacing the first occurrence of old_content with new_content.")
    def update_file(
        path: Annotated[str, Field(description="Relative path to the file to update.")],
        old_content: Annotated[str, Field(description="The exact text to find and replace.")],
        new_content: Annotated[str, Field(description="The text to replace old_content with.")],
    ) -> str:
        target = _resolve_path(base, path)
        text = target.read_text()
        if old_content not in text:
            raise ValueError(f"old_content not found in {path}")
        target.write_text(text.replace(old_content, new_content, 1))
        return f"Successfully updated {path}"

    return update_file


def _make_delete_file(base: Path) -> FunctionTool:
    @tool(description="Delete a file.")
    def delete_file(
        path: Annotated[str, Field(description="Relative path to the file to delete.")],
    ) -> str:
        target = _resolve_path(base, path)
        target.unlink()
        return f"Successfully deleted {path}"

    return delete_file


def _resolve_path(base: Path, path: str) -> Path:
    """Resolve *path* relative to *base* and verify it stays within *base*."""
    resolved = (base / path).resolve()
    if not resolved.is_relative_to(base):
        raise PermissionError(f"Path '{path}' escapes base directory '{base}'")
    return resolved
