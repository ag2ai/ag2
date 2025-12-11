# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import re
from collections.abc import Callable
from pathlib import Path, PurePath
from typing import Any, Protocol

try:
    import aiofiles
    import aiofiles.os
except ImportError:
    aiofiles = None  # type: ignore[assignment]

from ....doc_utils import export_module
from ...tool import Tool


class PatchEditor(Protocol):
    """Protocol for implementing file operations for apply_patch."""

    def create_file(self, operation: dict[str, Any]) -> dict[str, Any]:
        """Create a new file sync.

        Args:
            operation: Dict with 'path' and 'diff' keys

        Returns:
            Dict with 'status' ("completed" or "failed") and optional 'output' message
        """
        ...

    async def a_create_file(self, operation: dict[str, Any]) -> dict[str, Any]:
        """Create a new file async.

        Args:
            operation: Dict with 'path' and 'diff' keys

        Returns:
            Dict with 'status' ("completed" or "failed") and optional 'output' message
        """
        ...

    def update_file(self, operation: dict[str, Any]) -> dict[str, Any]:
        """Update an existing file sync.

        Args:
            operation: Dict with 'path' and 'diff' keys

        Returns:
            Dict with 'status' ("completed" or "failed") and optional 'output' message
        """
        ...

    async def a_update_file(self, operation: dict[str, Any]) -> dict[str, Any]:
        """Update an existing file async.

        Args:
            operation: Dict with 'path' and 'diff' keys

        Returns:
            Dict with 'status' ("completed" or "failed") and optional 'output' message
        """
        ...

    def delete_file(self, operation: dict[str, Any]) -> dict[str, Any]:
        """Delete a file sync.

        Args:
            operation: Dict with 'path' key

        Returns:
            Dict with 'status' ("completed" or "failed") and optional 'output' message
        """
        ...

    async def a_delete_file(self, operation: dict[str, Any]) -> dict[str, Any]:
        """Delete a file async.

        Args:
            operation: Dict with 'path' key

        Returns:
            Dict with 'status' ("completed" or "failed") and optional 'output' message
        """
        ...


class _V4ADiffApplier:
    """Minimal V4A diff interpreter with no external deps."""

    __slots__ = ("_original", "_cursor", "_result")
    _HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

    def __init__(self, original_text: str):
        self._original = original_text.splitlines()
        self._cursor = 0
        self._result: list[str] = []

    # ---- public -----------------------------------------------------------
    def apply(self, diff: str, create: bool) -> str:
        if create or not self._original:
            return self._reconstruct_from_create(diff)

        lines = diff.splitlines()
        idx = 0
        while idx < len(lines):
            header = lines[idx]
            match = self._HUNK_RE.match(header)
            if not match:
                idx += 1
                continue

            old_start = int(match.group(1))
            idx += 1
            self._emit_unchanged_until(old_start - 1)

            while idx < len(lines) and not lines[idx].startswith("@@"):
                self._consume_hunk_line(lines[idx])
                idx += 1

        self._result.extend(self._original[self._cursor :])
        return "\n".join(self._result)

    # ---- private ---------------------------------------------------------
    def _reconstruct_from_create(self, diff: str) -> str:
        new_lines: list[str] = []
        for line in diff.splitlines():
            if not line:
                new_lines.append("")
            elif line.startswith(("@@", "---", "+++")):
                continue
            elif line.startswith("+"):
                new_lines.append(line[1:])
            elif line.startswith("-"):
                continue
            else:
                new_lines.append(line)
        return "\n".join(new_lines)

    def _emit_unchanged_until(self, target_line: int) -> None:
        while self._cursor < target_line and self._cursor < len(self._original):
            self._result.append(self._original[self._cursor])
            self._cursor += 1

    def _consume_hunk_line(self, line: str) -> None:
        if not line or line.startswith("\\ No newline"):
            return
        prefix = line[0] if line else " "
        payload = line[1:] if len(line) > 1 and prefix in "+- " else line

        if prefix == "+":
            self._result.append(payload)
        elif prefix == "-":
            if self._cursor >= len(self._original):
                raise ValueError(f"Deletion beyond file end at line {len(self._result) + 1}")
            if payload != self._original[self._cursor]:
                raise ValueError(
                    f"Deletion mismatch at line {self._cursor + 1}:\n"
                    f"  Expected: {self._original[self._cursor]!r}\n"
                    f"  Got:      {payload!r}"
                )
            self._cursor += 1
        else:  # context (' ' or no prefix)
            if self._cursor < len(self._original):
                if payload != self._original[self._cursor]:
                    raise ValueError(
                        f"Context mismatch at line {self._cursor + 1}:\n"
                        f"  Expected: {self._original[self._cursor]!r}\n"
                        f"  Got:      {payload!r}"
                    )
                self._result.append(payload)
                self._cursor += 1
            else:
                self._result.append(payload)


def apply_diff(current_content: str, diff: str, create: bool = False) -> str:
    """Apply a V4A diff to file content."""
    applier = _V4ADiffApplier(current_content)
    return applier.apply(diff, create)


class WorkspaceEditor:
    """File system editor for apply_patch operations.

    currently supports local filesystem through allowed_paths patterns.
    """

    def __init__(
        self,
        workspace_dir: str | Path,
        allowed_paths: list[str] | None = None,
    ):
        """Initialize workspace editor.

        Args:
            workspace_dir: Root directory for file operations (local filesystem path).

            allowed_paths: List of allowed path patterns (for security).
                Supports glob-style patterns with ** for recursive matching.
                currently works for local filesystem.

                Examples:
                    - ["**"] - Allow all paths (default)
                    - ["src/**"] - Allow all files in src/ and subdirectories
                    - ["*.py"] - Allow Python files in root directory
                    - ["src/**", "tests/**"] - Allow paths in multiple directories
        """
        workspace_dir = workspace_dir if workspace_dir is not None else os.getcwd()
        self.workspace_dir = Path(workspace_dir).resolve()
        # Use "**" to match all files and directories recursively (including root)
        self.allowed_paths = allowed_paths if allowed_paths is not None else ["**"]

    def _validate_path(self, path: str) -> Path:
        """Validate and resolve a file path.

        Args:
            path: Relative path to validate (local filesystem)

        Returns:
            Absolute resolved path (for local filesystem)

        Raises:
            ValueError: If path is invalid or outside workspace, or not in allowed_paths
        """
        # Check if path matches any allowed pattern first (works for both local and cloud paths)
        # This allows cloud storage bucket patterns in allowed_paths
        # Use PurePath.match() which properly supports ** for recursive matching
        matches_any = any(PurePath(path).match(pattern) for pattern in self.allowed_paths)

        if not matches_any:
            raise ValueError(f"Path {path} is not allowed by allowed_paths patterns: {self.allowed_paths}")

        # Validate path is within workspace
        try:
            full_path = (Path(self.workspace_dir) / path).resolve()

            # Use Path.relative_to() for secure path containment check
            # This prevents path traversal attacks that startswith() would miss
            try:
                full_path.relative_to(self.workspace_dir)
            except ValueError:
                raise ValueError(f"Path {path} is outside workspace directory")

            return full_path
        except (OSError, ValueError) as e:
            raise ValueError(f"Path {path} is invalid: {str(e)}")

    def create_file(self, operation: dict[str, Any]) -> dict[str, Any]:
        """Create a new file."""
        try:
            path = operation.get("path")
            if not path:
                return {"status": "failed", "output": "Missing 'path' in operation"}

            diff = operation.get("diff", "")

            full_path = self._validate_path(path)

            # Ensure parent directory exists
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Apply diff to get file content
            content = apply_diff("", diff, create=True)

            # Write file
            full_path.write_text(content, encoding="utf-8")

            return {"status": "completed", "output": f"Created {path}"}
        except Exception as e:
            return {"status": "failed", "output": f"Error creating {path}: {str(e)}"}

    async def a_create_file(self, operation: dict[str, Any]) -> dict[str, Any]:
        """Create a new file."""
        if aiofiles is None:
            raise RuntimeError("aiofiles is required for async file operations. Install it with: pip install aiofiles")

        try:
            path = operation.get("path")
            if not path:
                return {"status": "failed", "output": "Missing 'path' in operation"}

            diff = operation.get("diff", "")

            full_path = self._validate_path(path)

            # Ensure parent directory exists (use asyncio.to_thread for blocking mkdir)
            await asyncio.to_thread(full_path.parent.mkdir, parents=True, exist_ok=True)

            # Apply diff to get file content
            content = apply_diff("", diff, create=True)

            # Write file using async I/O
            async with aiofiles.open(str(full_path), "w", encoding="utf-8") as f:
                await f.write(content)

            return {"status": "completed", "output": f"Created {path}"}
        except Exception as e:
            return {"status": "failed", "output": f"Error creating {path}: {str(e)}"}

    def update_file(self, operation: dict[str, Any]) -> dict[str, Any]:
        """Update an existing file."""
        try:
            path = operation.get("path")
            if not path:
                return {"status": "failed", "output": "Missing 'path' in operation"}

            diff = operation.get("diff", "")

            full_path = self._validate_path(path)

            if not full_path.exists():
                return {"status": "failed", "output": f"Error: File not found at path '{path}'"}

            # Read current content
            current_content = full_path.read_text(encoding="utf-8")

            # Apply diff
            new_content = apply_diff(current_content, diff)

            # Write updated content
            full_path.write_text(new_content, encoding="utf-8")

            return {"status": "completed", "output": f"Updated {path}"}
        except Exception as e:
            return {"status": "failed", "output": f"Error updating {path}: {str(e)}"}

    async def a_update_file(self, operation: dict[str, Any]) -> dict[str, Any]:
        """Update an existing file."""
        if aiofiles is None:
            raise RuntimeError("aiofiles is required for async file operations. Install it with: pip install aiofiles")

        try:
            path = operation.get("path")
            if not path:
                return {"status": "failed", "output": "Missing 'path' in operation"}

            diff = operation.get("diff", "")

            full_path = self._validate_path(path)

            # Check if file exists using asyncio.to_thread
            exists = await asyncio.to_thread(full_path.exists)
            if not exists:
                return {"status": "failed", "output": f"Error: File not found at path '{path}'"}

            # Read current content using async I/O
            async with aiofiles.open(str(full_path), "r", encoding="utf-8") as f:
                current_content = await f.read()

            # Apply diff
            new_content = apply_diff(current_content, diff)

            # Write updated content using async I/O
            async with aiofiles.open(str(full_path), "w", encoding="utf-8") as f:
                await f.write(new_content)

            return {"status": "completed", "output": f"Updated {path}"}
        except Exception as e:
            return {"status": "failed", "output": f"Error updating {path}: {str(e)}"}

    def delete_file(self, operation: dict[str, Any]) -> dict[str, Any]:
        """Delete a file."""
        try:
            path = operation.get("path")
            if not path:
                return {"status": "failed", "output": "Missing 'path' in operation"}

            full_path = self._validate_path(path)

            if not full_path.exists():
                return {"status": "failed", "output": f"Error: File not found at path '{path}'"}

            full_path.unlink()

            return {"status": "completed", "output": f"Deleted {path}"}
        except Exception as e:
            return {"status": "failed", "output": f"Error deleting {path}: {str(e)}"}

    async def a_delete_file(self, operation: dict[str, Any]) -> dict[str, Any]:
        """Delete a file."""
        if aiofiles is None:
            raise RuntimeError("aiofiles is required for async file operations. Install it with: pip install aiofiles")

        try:
            path = operation.get("path")
            if not path:
                return {"status": "failed", "output": "Missing 'path' in operation"}

            full_path = self._validate_path(path)

            # Check if file exists using asyncio.to_thread
            exists = await asyncio.to_thread(full_path.exists)
            if not exists:
                return {"status": "failed", "output": f"Error: File not found at path '{path}'"}

            # Delete file using async I/O
            await aiofiles.os.remove(str(full_path))

            return {"status": "completed", "output": f"Deleted {path}"}
        except Exception as e:
            return {"status": "failed", "output": f"Error deleting {path}: {str(e)}"}


# Valid operation types for apply_patch tool
VALID_OPERATIONS = {"create_file", "update_file", "delete_file"}


@export_module("autogen.tools")
class ApplyPatchTool(Tool):
    """Tool for applying code patches with GPT-5.1.

    This tool enables agents to create, update, and delete files using
    structured diffs from the OpenAI Responses API.

    Example:
        from autogen.tools import ApplyPatchTool, WorkspaceEditor
        from autogen import ConversableAgent

        # Create editor for workspace
        editor = WorkspaceEditor(workspace_dir="./my_project")

        # Create tool
        patch_tool = ApplyPatchTool(
            editor=editor,
            needs_approval=True,
            on_approval=lambda ctx, item: {"approve": True}
        )

        # Register with agent
        agent = ConversableAgent(
            name="coding_assistant",
            llm_config={
                "api_type": "responses",
                "model": "gpt-5.1",
                "built_in_tools": ["apply_patch"]
            }
        )
        patch_tool.register_tool(agent)
    """

    def __init__(
        self,
        *,
        editor: PatchEditor | None = None,
        workspace_dir: str | Path | None = None,
        needs_approval: bool = False,
        on_approval: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
        allowed_paths: list[str] | None = ["**"],
        async_patches: bool = False,
    ):
        """Initialize ApplyPatchTool.

        Args:
            editor: Custom PatchEditor implementation (optional)
            workspace_dir: Directory for file operations (used if editor not provided).
                For local filesystem operations only.
            needs_approval: Whether operations require approval
            on_approval: Callback for approval decisions
            allowed_paths: List of allowed path patterns (for security).
                Supports glob-style patterns with ** for recursive matching.
                currently works for local filesystem.
            async_patches: apply patches asynchronously / synchronously
                Examples:
                    - ["**"] - Allow all paths (default)
                    - ["src/**"] - Allow all files in src/ and subdirectories
                    - ["*.py"] - Allow Python files in root directory

                Note:
                The workspace_dir should remain a local path for default operations.
        """
        if editor is None and workspace_dir is None:
            raise ValueError("Either 'editor' or 'workspace_dir' must be provided")

        if editor is None:
            editor = WorkspaceEditor(workspace_dir=workspace_dir, allowed_paths=allowed_paths)  # type: ignore

        self.editor = editor
        self.needs_approval = needs_approval
        self.on_approval = on_approval or (lambda ctx, item: {"approve": True})
        self.async_patches = async_patches

        # Create the tool function

        async def _apply_patch_handler(call_id: str, operation: dict[str, Any]) -> dict[str, Any]:
            """Handle apply_patch operations from GPT-5.1.

            Args:
                call_id: Unique identifier for this patch operation
                operation: Dict containing operation type, path, and diff

            Returns:
                Result dict with status and output
            """
            operation_type = operation.get("type")

            # Validate operation type early
            if operation_type not in VALID_OPERATIONS:
                return {
                    "type": "apply_patch_call_output",
                    "call_id": call_id,
                    "status": "failed",
                    "output": f"Invalid operation type: {operation_type}. Must be one of {VALID_OPERATIONS}",
                }

            # Check approval if needed
            if self.needs_approval:
                approval = self.on_approval({}, operation)
                if not approval.get("approve", False):
                    return {
                        "type": "apply_patch_call_output",
                        "call_id": call_id,
                        "status": "failed",
                        "output": "Operation rejected by user",
                    }

            # Route to appropriate handler
            if self.async_patches:
                if operation_type == "create_file":
                    result = await self.editor.a_create_file(operation)  # type: ignore[union-attr]
                elif operation_type == "update_file":
                    result = await self.editor.a_update_file(operation)  # type: ignore[union-attr]
                elif operation_type == "delete_file":
                    result = await self.editor.a_delete_file(operation)  # type: ignore[union-attr]
                else:
                    # This should never happen due to validation above, but kept for safety
                    result = {"status": "failed", "output": f"Unknown operation type: {operation_type}"}
            else:
                if operation_type == "create_file":
                    result = self.editor.create_file(operation)  # type: ignore[union-attr]
                elif operation_type == "update_file":
                    result = self.editor.update_file(operation)  # type: ignore[union-attr]
                elif operation_type == "delete_file":
                    result = self.editor.delete_file(operation)  # type: ignore[union-attr]
                else:
                    # This should never happen due to validation above, but kept for safety
                    result = {"status": "failed", "output": f"Unknown operation type: {operation_type}"}
            # Format response for Responses API
            return {"type": "apply_patch_call_output", "call_id": call_id, **result}

        # Initialize as Tool
        super().__init__(
            name="apply_patch_tool",
            description="Apply code patches to create, update, or delete files",
            func_or_tool=_apply_patch_handler,
        )
