# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64
import os
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

from pydantic import Field

from autogen.beta.middleware import ToolMiddleware
from autogen.beta.tools.final import Toolkit, tool
from autogen.beta.tools.final.function_tool import FunctionTool


class FilesystemToolkit(Toolkit):
    """Toolkit that exposes local filesystem operations as function tools.

    All paths are resolved relative to *base_path*.  A path-traversal guard
    rejects any resolved path that escapes the base directory.

    Individual tools are available as attributes and can be passed to an agent
    separately::

        fs = FilesystemToolkit(base_path="/tmp/workspace")

        # use the full toolkit
        agent = Agent("agent", config=config, tools=[fs])

        # or pick individual tools
        agent = Agent(
            "agent",
            config=config,
            tools=[
                fs.read_file(),
                fs.find_files(),
            ],
        )
    """

    __slots__ = ("base_dir",)

    def __init__(
        self,
        base_path: str | os.PathLike[str] = ".",
        *,
        read_only: bool = False,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        self.base_dir = Path(base_path).resolve()

        tools = [
            self.read_file(),
            self.find_files(),
        ]

        if not read_only:
            tools.extend([
                self.write_file(),
                self.update_file(),
                self.delete_file(),
            ])

        super().__init__(
            *tools,
            name="filesystem_toolkit",
            middleware=middleware,
        )

    def read_file(
        self,
        base_path: str | os.PathLike[str] | None = None,
        *,
        name: str = "read_file",
        description: str = "Read the contents of a file. Returns text by default. Set raw=true to read binary content as base64.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        base_dir = _resolve_dir(base_path, self.base_dir)

        @tool(name=name, description=description, middleware=middleware)
        def _read_file(
            path: Annotated[
                str,
                Field(description="Relative path to the file to read."),
            ],
            raw: bool = Field(
                default=False,
                description="If true, read the file as binary and return base64-encoded content.",
            ),
        ) -> str:
            target = _resolve_path(base_dir, path)
            if raw:
                return base64.b64encode(target.read_bytes()).decode("ascii")
            return target.read_text()

        return _read_file

    def find_files(
        self,
        base_path: str | os.PathLike[str] | None = None,
        *,
        name: str = "find_files",
        description: str = "Search for files matching a glob pattern. Use recursive patterns like '**/*.py' to search subdirectories.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        base_dir = _resolve_dir(base_path, self.base_dir)

        @tool(name=name, description=description, middleware=middleware)
        def _find_files(
            pattern: Annotated[
                str,
                Field(description="Glob pattern to match files, e.g. '*.py' or '**/*.txt'."),
            ],
            path: str = Field(
                default=".",
                description="Relative path to the directory to search from. Defaults to the base directory.",
            ),
        ) -> list[str]:
            target = _resolve_path(base_dir, path)
            return sorted(str(p.relative_to(base_dir)) for p in _glob(target, pattern))

        return _find_files

    def write_file(
        self,
        base_path: str | os.PathLike[str] | None = None,
        *,
        name: str = "write_file",
        description: str = "Create or overwrite a file with the given content. Parent directories are created automatically.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        base_dir = _resolve_dir(base_path, self.base_dir)

        @tool(name=name, description=description, middleware=middleware)
        def _write_file(
            path: Annotated[
                str,
                Field(description="Relative path to the file to create or overwrite."),
            ],
            content: Annotated[
                str,
                Field(description="The full content to write to the file."),
            ],
        ) -> str:
            target = _resolve_path(base_dir, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            return f"Successfully wrote {len(content)} characters to {path}"

        return _write_file

    def update_file(
        self,
        base_path: str | os.PathLike[str] | None = None,
        *,
        name: str = "update_file",
        description: str = "Update a file by replacing the first occurrence of old_content with new_content.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        base_dir = _resolve_dir(base_path, self.base_dir)

        @tool(name=name, description=description, middleware=middleware)
        def _update_file(
            path: Annotated[
                str,
                Field(description="Relative path to the file to update."),
            ],
            old_content: Annotated[
                str,
                Field(description="The exact text to find and replace."),
            ],
            new_content: Annotated[
                str,
                Field(description="The text to replace old_content with."),
            ],
        ) -> str:
            target = _resolve_path(base_dir, path)
            text = target.read_text()
            if old_content not in text:
                raise ValueError(f"old_content not found in {path}")
            target.write_text(text.replace(old_content, new_content, 1))
            return f"Successfully updated {path}"

        return _update_file

    def delete_file(
        self,
        base_path: str | os.PathLike[str] | None = None,
        *,
        name: str = "delete_file",
        description: str = "Delete a file or directory.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        base_dir = _resolve_dir(base_path, self.base_dir)

        @tool(name=name, description=description, middleware=middleware)
        def _delete_file(
            path: Annotated[
                str,
                Field(description="Relative path to the file or directory to delete."),
            ],
        ) -> str:
            target = _resolve_path(base_dir, path)
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            return f"Successfully deleted {path}"

        return _delete_file


def _resolve_dir(
    arg_path: str | os.PathLike[str] | None,
    toolkit_path: Path,
) -> Path:
    path = Path(arg_path).resolve() if arg_path is not None else toolkit_path
    if not path.is_dir():
        raise ValueError(f"{path} is not a directory type.")
    return path


def _resolve_path(base: Path, path: str) -> Path:
    """Resolve *path* relative to *base* and verify it stays within *base*."""
    resolved = (base / path).resolve()
    if not resolved.is_relative_to(base):
        raise PermissionError(f"Path '{path}' escapes base directory '{base}'")
    return resolved


if sys.version_info >= (3, 13):

    def _glob(target: Path, pattern: str) -> Iterable[Path]:
        """Yield files matching *pattern* under *target* with consistent behavior across Python versions."""
        return (p for p in target.glob(pattern) if p.is_file())

else:

    def _glob(target: Path, pattern: str) -> Iterable[Path]:
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
