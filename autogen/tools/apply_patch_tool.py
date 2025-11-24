# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from ..doc_utils import export_module
from .tool import Tool


class PatchEditor(Protocol):
    """Protocol for implementing file operations for apply_patch."""

    async def create_file(self, operation: dict[str, Any]) -> dict[str, Any]:
        """Create a new file.

        Args:
            operation: Dict with 'path' and 'diff' keys

        Returns:
            Dict with 'status' ("completed" or "failed") and optional 'output' message
        """
        ...

    async def update_file(self, operation: dict[str, Any]) -> dict[str, Any]:
        """Update an existing file.

        Args:
            operation: Dict with 'path' and 'diff' keys

        Returns:
            Dict with 'status' ("completed" or "failed") and optional 'output' message
        """
        ...

    async def delete_file(self, operation: dict[str, Any]) -> dict[str, Any]:
        """Delete a file.

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
    """Default file system editor for apply_patch operations."""

    def __init__(
        self,
        workspace_dir: str | Path,
        allowed_paths: list[str] | None = None,
    ):
        """Initialize workspace editor.

        Args:
            workspace_dir: Root directory for file operations
            allowed_paths: List of allowed path patterns (for security)
        """
        print("workspace_dir", workspace_dir)
        self.workspace_dir = Path(workspace_dir).resolve()
        # Use "**" to match all files and directories recursively (including root)
        self.allowed_paths = allowed_paths or ["**"]

    def _validate_path(self, path: str) -> Path:
        """Validate and resolve a file path.

        Args:
            path: Relative path to validate

        Returns:
            Absolute resolved path

        Raises:
            ValueError: If path is invalid or outside workspace, or not in allowed_paths
        """
        full_path = (self.workspace_dir / path).resolve()
        print("workspace_dir", self.workspace_dir)
        print("full_path", full_path)
        # Security: Ensure path is within workspace
        if not str(full_path).startswith(str(self.workspace_dir)):
            raise ValueError(f"Path {path} is outside workspace directory")

        # Security: Ensure path matches allowed_paths patterns
        # Get relative path from workspace directory for pattern matching
        try:
            relative_path = full_path.relative_to(self.workspace_dir)
        except ValueError:
            # This shouldn't happen if the above check passed, but handle it anyway
            raise ValueError(f"Path {path} is outside workspace directory")

        # Check if path matches any allowed pattern using pathlib.Path.match()
        # which supports ** for recursive matching
        # Note: **/* matches files in subdirectories, ** matches everything
        matches_any = False
        for pattern in self.allowed_paths:
            # pathlib.Path.match() supports ** for recursive matching
            if relative_path.match(pattern):
                matches_any = True
                break

        if not matches_any:
            raise ValueError(f"Path {path} is not allowed by allowed_paths patterns: {self.allowed_paths}")

        return full_path

    async def create_file(self, operation: dict[str, Any]) -> dict[str, Any]:
        """Create a new file."""
        try:
            path = operation.get("path")
            diff = operation.get("diff", "")

            full_path = self._validate_path(path)  # type: ignore[arg-type]

            # Ensure parent directory exists
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Apply diff to get file content
            content = apply_diff("", diff, create=True)

            # Write file
            full_path.write_text(content, encoding="utf-8")

            return {"status": "completed", "output": f"Created {path}"}
        except Exception as e:
            return {"status": "failed", "output": f"Error creating {path}: {str(e)}"}

    async def update_file(self, operation: dict[str, Any]) -> dict[str, Any]:
        """Update an existing file."""
        try:
            path = operation.get("path")
            diff = operation.get("diff", "")

            full_path = self._validate_path(path)  # type: ignore[arg-type]

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

    async def delete_file(self, operation: dict[str, Any]) -> dict[str, Any]:
        """Delete a file."""
        try:
            path = operation.get("path")
            full_path = self._validate_path(path)  # type: ignore[arg-type]

            if not full_path.exists():
                return {"status": "failed", "output": f"Error: File not found at path '{path}'"}

            full_path.unlink()

            return {"status": "completed", "output": f"Deleted {path}"}
        except Exception as e:
            return {"status": "failed", "output": f"Error deleting {path}: {str(e)}"}


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
    ):
        """Initialize ApplyPatchTool.

        Args:
            editor: Custom PatchEditor implementation (optional)
            workspace_dir: Directory for file operations (used if editor not provided)
            needs_approval: Whether operations require approval
            on_approval: Callback for approval decisions
        """
        if editor is None and workspace_dir is None:
            raise ValueError("Either 'editor' or 'workspace_dir' must be provided")

        if editor is None:
            editor = WorkspaceEditor(workspace_dir=workspace_dir)  # type: ignore

        self.editor = editor
        self.needs_approval = needs_approval
        self.on_approval = on_approval or (lambda ctx, item: {"approve": True})

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
            if operation_type == "create_file":
                result = await self.editor.create_file(operation)
            elif operation_type == "update_file":
                result = await self.editor.update_file(operation)
            elif operation_type == "delete_file":
                result = await self.editor.delete_file(operation)
            else:
                result = {"status": "failed", "output": f"Unknown operation type: {operation_type}"}

            # Format response for Responses API
            return {"type": "apply_patch_call_output", "call_id": call_id, **result}

        # Initialize as Tool
        super().__init__(
            name="apply_patch_tool",
            description="Apply code patches to create, update, or delete files",
            func_or_tool=_apply_patch_handler,
        )
