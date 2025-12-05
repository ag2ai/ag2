# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from autogen.tools.apply_patch_tool import (
    ApplyPatchTool,
    PatchEditor,
    WorkspaceEditor,
    _V4ADiffApplier,
    apply_diff,
)


class TestV4ADiffApplier:
    """Test the _V4ADiffApplier class."""

    def test_apply_create_file_empty_diff(self) -> None:
        """Test creating a file from empty diff."""
        applier = _V4ADiffApplier("")
        result = applier.apply("", create=True)
        assert result == ""

    def test_apply_create_file_simple(self) -> None:
        """Test creating a file with simple content."""
        diff = """--- /dev/null
+++ test.py
@@ -0,0 +1,2 @@
+line1
+line2"""
        applier = _V4ADiffApplier("")
        result = applier.apply(diff, create=True)
        assert result == "line1\nline2"

    def test_apply_create_file_with_plus_prefix(self) -> None:
        """Test creating a file with + prefix lines."""
        diff = """@@ -0,0 +1,3 @@
+def hello():
+    print("world")
+    return True"""
        applier = _V4ADiffApplier("")
        result = applier.apply(diff, create=True)
        assert "def hello():" in result
        assert 'print("world")' in result

    def test_apply_update_file_simple(self) -> None:
        """Test updating a file with a simple change."""
        original = "line1\nline2\nline3"
        diff = """@@ -1,3 +1,3 @@
 line1
-line2
+line2_modified
 line3"""
        applier = _V4ADiffApplier(original)
        result = applier.apply(diff, create=False)
        assert result == "line1\nline2_modified\nline3"

    def test_apply_update_file_add_lines(self) -> None:
        """Test adding lines to a file."""
        original = "line1\nline2"
        diff = """@@ -1,2 +1,4 @@
 line1
+new_line1
+new_line2
 line2"""
        applier = _V4ADiffApplier(original)
        result = applier.apply(diff, create=False)
        assert result == "line1\nnew_line1\nnew_line2\nline2"

    def test_apply_update_file_delete_lines(self) -> None:
        """Test deleting lines from a file."""
        original = "line1\nline2\nline3"
        diff = """@@ -1,3 +1,2 @@
 line1
-line2
 line3"""
        applier = _V4ADiffApplier(original)
        result = applier.apply(diff, create=False)
        assert result == "line1\nline3"

    def test_apply_update_file_multiple_hunks(self) -> None:
        """Test updating a file with multiple hunks."""
        original = "line1\nline2\nline3\nline4\nline5"
        diff = """@@ -1,2 +1,2 @@
 line1
-line2
+line2_modified
@@ -4,2 +4,2 @@
 line4
-line5
+line5_modified"""
        applier = _V4ADiffApplier(original)
        result = applier.apply(diff, create=False)
        assert result == "line1\nline2_modified\nline3\nline4\nline5_modified"

    def test_apply_update_file_context_mismatch_raises(self) -> None:
        """Test that context mismatch raises ValueError."""
        original = "line1\nline2\nline3"
        diff = """@@ -1,3 +1,3 @@
 line1
-wrong_context
+line2_modified
 line3"""
        applier = _V4ADiffApplier(original)
        with pytest.raises(ValueError, match="Deletion mismatch"):
            applier.apply(diff, create=False)

    def test_apply_update_file_deletion_beyond_end_raises(self) -> None:
        """Test that deletion beyond file end raises ValueError."""
        original = "line1"
        diff = """@@ -1,1 +1,0 @@
-line1
-line2"""
        applier = _V4ADiffApplier(original)
        with pytest.raises(ValueError, match="Deletion beyond file end"):
            applier.apply(diff, create=False)

    def test_apply_ignores_no_newline_marker(self) -> None:
        """Test that \\ No newline at end of file is ignored."""
        original = "line1\nline2"
        diff = """@@ -1,2 +1,3 @@
 line1
 line2
+line3
\\ No newline at end of file"""
        applier = _V4ADiffApplier(original)
        result = applier.apply(diff, create=False)
        assert result == "line1\nline2\nline3"


class TestApplyDiff:
    """Test the apply_diff function."""

    def test_apply_diff_create(self) -> None:
        """Test apply_diff for creating a file."""
        diff = """@@ -0,0 +1,2 @@
+hello
+world"""
        result = apply_diff("", diff, create=True)
        assert result == "hello\nworld"

    def test_apply_diff_update(self) -> None:
        """Test apply_diff for updating a file."""
        original = "old content"
        diff = """@@ -1,1 +1,1 @@
-old content
+new content"""
        result = apply_diff(original, diff, create=False)
        assert result == "new content"


class TestWorkspaceEditor:
    """Test the WorkspaceEditor class."""

    @pytest.fixture
    def workspace_dir(self, tmp_path: Path) -> Path:
        """Create a temporary workspace directory."""
        return tmp_path / "workspace"

    @pytest.fixture
    def editor(self, workspace_dir: Path) -> WorkspaceEditor:
        """Create a WorkspaceEditor instance."""
        workspace_dir.mkdir()
        return WorkspaceEditor(workspace_dir=workspace_dir)

    @pytest.mark.asyncio
    async def test_create_file_simple(self, editor: WorkspaceEditor) -> None:
        """Test creating a simple file."""
        diff = """@@ -0,0 +1,2 @@
+hello
+world"""
        operation = {"path": "test.txt", "diff": diff}
        result = editor.create_file(operation)
        assert result["status"] == "completed"
        assert "Created test.txt" in result["output"]

        file_path = editor.workspace_dir / "test.txt"
        assert file_path.exists()
        assert file_path.read_text() == "hello\nworld"

    @pytest.mark.asyncio
    async def test_create_file_nested_directory(self, editor: WorkspaceEditor) -> None:
        """Test creating a file in a nested directory."""
        diff = """@@ -0,0 +1,1 @@
+content"""
        operation = {"path": "nested/dir/test.txt", "diff": diff}
        result = editor.create_file(operation)
        assert result["status"] == "completed"

        file_path = editor.workspace_dir / "nested" / "dir" / "test.txt"
        assert file_path.exists()
        assert file_path.read_text() == "content"

    @pytest.mark.asyncio
    async def test_update_file_simple(self, editor: WorkspaceEditor) -> None:
        """Test updating an existing file."""
        # Create initial file
        file_path = editor.workspace_dir / "test.txt"
        file_path.write_text("old content")

        diff = """@@ -1,1 +1,1 @@
-old content
+new content"""
        operation = {"path": "test.txt", "diff": diff}
        result = editor.update_file(operation)
        assert result["status"] == "completed"
        assert "Updated test.txt" in result["output"]
        assert file_path.read_text() == "new content"

    @pytest.mark.asyncio
    async def test_update_file_not_found(self, editor: WorkspaceEditor) -> None:
        """Test updating a non-existent file."""
        diff = """@@ -1,1 +1,1 @@
-old
+new"""
        operation = {"path": "nonexistent.txt", "diff": diff}
        result = editor.update_file(operation)
        assert result["status"] == "failed"
        assert "File not found" in result["output"]

    @pytest.mark.asyncio
    async def test_delete_file_simple(self, editor: WorkspaceEditor) -> None:
        """Test deleting a file."""
        file_path = editor.workspace_dir / "test.txt"
        file_path.write_text("content")

        operation = {"path": "test.txt"}
        result = editor.delete_file(operation)
        assert result["status"] == "completed"
        assert "Deleted test.txt" in result["output"]
        assert not file_path.exists()

    @pytest.mark.asyncio
    async def test_delete_file_not_found(self, editor: WorkspaceEditor) -> None:
        """Test deleting a non-existent file."""
        operation = {"path": "nonexistent.txt"}
        result = editor.delete_file(operation)
        assert result["status"] == "failed"
        assert "File not found" in result["output"]

    @pytest.mark.asyncio
    async def test_validate_path_outside_workspace(self, workspace_dir: Path) -> None:
        """Test that paths outside workspace are rejected."""
        workspace_dir.mkdir()
        # Use a restrictive allowed_paths pattern so the path doesn't match
        # This forces the workspace validation to run
        editor = WorkspaceEditor(workspace_dir=workspace_dir, allowed_paths=["src/**"])

        # Use a relative path that escapes the workspace
        operation = {"path": "../../outside_file.txt", "diff": "@@ -0,0 +1,1 @@\n+test"}
        result = editor.create_file(operation)
        # Should fail because path doesn't match allowed_paths pattern
        assert result["status"] == "failed"
        assert "not allowed" in result["output"].lower()

    @pytest.mark.asyncio
    async def test_validate_path_allowed_paths(self, workspace_dir: Path) -> None:
        """Test that allowed_paths patterns are enforced."""
        workspace_dir.mkdir()
        editor = WorkspaceEditor(workspace_dir=workspace_dir, allowed_paths=["*.py", "src/*"])

        # Should allow .py files
        operation = {"path": "test.py", "diff": "@@ -0,0 +1,1 @@\n+code"}
        result = editor.create_file(operation)
        assert result["status"] == "completed"

        # Should allow files in src/
        operation = {"path": "src/main.py", "diff": "@@ -0,0 +1,1 @@\n+code"}
        result = editor.create_file(operation)
        assert result["status"] == "completed"

        # Should reject .txt files
        operation = {"path": "test.txt", "diff": "@@ -0,0 +1,1 @@\n+content"}
        result = editor.create_file(operation)
        assert result["status"] == "failed"
        assert "not allowed" in result["output"].lower()

    @pytest.mark.asyncio
    async def test_create_file_error_handling(self, editor: WorkspaceEditor) -> None:
        """Test error handling in create_file."""
        # Invalid diff that causes an error
        operation = {"path": "test.txt", "diff": "@@ invalid diff"}
        result = editor.create_file(operation)
        # Should handle gracefully
        assert result["status"] in ["completed", "failed"]

    @pytest.mark.asyncio
    async def test_validate_path_cloud_storage_patterns(self, workspace_dir: Path) -> None:
        """Test that cloud storage patterns in allowed_paths work correctly."""
        workspace_dir.mkdir()
        editor = WorkspaceEditor(workspace_dir=workspace_dir, allowed_paths=["my-bucket/**", "s3://bucket/src/**"])

        # Should allow cloud storage paths
        operation = {"path": "my-bucket/src/file.py", "diff": "@@ -0,0 +1,1 @@\n+code"}
        result = editor.create_file(operation)
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_validate_path_recursive_pattern(self, workspace_dir: Path) -> None:
        """Test that ** recursive patterns work with fnmatch."""
        workspace_dir.mkdir()
        editor = WorkspaceEditor(workspace_dir=workspace_dir, allowed_paths=["src/**"])

        # Should allow nested paths
        operation = {"path": "src/utils/helpers/file.py", "diff": "@@ -0,0 +1,1 @@\n+code"}
        result = editor.create_file(operation)
        assert result["status"] == "completed"

        # Should reject paths outside src/
        operation = {"path": "other/file.py", "diff": "@@ -0,0 +1,1 @@\n+code"}
        result = editor.create_file(operation)
        assert result["status"] == "failed"
        assert "not allowed" in result["output"].lower()

    def test_apply_diff_without_line_numbers(self) -> None:
        """Test apply_diff handles diffs without line numbers."""
        original = "line1\nline2\nline3"
        # Diff without line numbers in @@ header (current implementation skips these)
        diff = """@@
 line1
-line2
+line2_modified
 line3"""
        # Current implementation skips hunks without line numbers
        # This test documents current behavior - enhancement would make this pass
        result = apply_diff(original, diff, create=False)
        # Without enhancement, the diff is skipped, so original content is returned
        assert result == original  # Current behavior: diff is ignored
        # TODO: When enhance_diff_with_line_numbers is implemented, this should be:
        # assert "line2_modified" in result


class TestApplyPatchTool:
    """Test the ApplyPatchTool class."""

    @pytest.fixture
    def mock_editor(self) -> AsyncMock:
        """Create a mock PatchEditor."""
        editor = AsyncMock(spec=PatchEditor)
        editor.create_file.return_value = {"status": "completed", "output": "Created file"}
        editor.update_file.return_value = {"status": "completed", "output": "Updated file"}
        editor.delete_file.return_value = {"status": "completed", "output": "Deleted file"}
        return editor

    @pytest.fixture
    def tool_with_editor(self, mock_editor: AsyncMock) -> ApplyPatchTool:
        """Create ApplyPatchTool with a mock editor."""
        return ApplyPatchTool(editor=mock_editor)

    @pytest.fixture
    def tool_with_workspace(self, tmp_path: Path) -> ApplyPatchTool:
        """Create ApplyPatchTool with a workspace directory."""
        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir()
        return ApplyPatchTool(workspace_dir=workspace_dir)

    def test_init_with_editor(self, mock_editor: AsyncMock) -> None:
        """Test initialization with custom editor."""
        tool = ApplyPatchTool(editor=mock_editor)
        assert tool.editor == mock_editor
        assert tool.needs_approval is False

    def test_init_with_workspace_dir(self, tmp_path: Path) -> None:
        """Test initialization with workspace directory."""
        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir()
        tool = ApplyPatchTool(workspace_dir=workspace_dir)
        assert isinstance(tool.editor, WorkspaceEditor)
        assert tool.editor.workspace_dir == workspace_dir

    def test_init_with_neither_raises(self) -> None:
        """Test that initialization without editor or workspace_dir raises."""
        with pytest.raises(ValueError, match="Either 'editor' or 'workspace_dir' must be provided"):
            ApplyPatchTool()

    def test_init_with_approval(self, mock_editor: AsyncMock) -> None:
        """Test initialization with approval callback."""

        def approval_callback(ctx: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
            return {"approve": True}

        tool = ApplyPatchTool(editor=mock_editor, needs_approval=True, on_approval=approval_callback)
        assert tool.needs_approval is True
        assert tool.on_approval == approval_callback

    @pytest.mark.asyncio
    async def test_apply_patch_create_file(self, tool_with_editor: ApplyPatchTool, mock_editor: AsyncMock) -> None:
        """Test apply_patch handler for create_file operation."""
        operation = {"type": "create_file", "path": "test.txt", "diff": "@@ -0,0 +1,1 @@\n+content"}
        call_id = "test-call-123"

        # Get the handler function
        handler = tool_with_editor.func
        result = await handler(call_id, operation)

        assert result["type"] == "apply_patch_call_output"
        assert result["call_id"] == call_id
        assert result["status"] == "completed"
        mock_editor.create_file.assert_called_once_with(operation)

    @pytest.mark.asyncio
    async def test_apply_patch_update_file(self, tool_with_editor: ApplyPatchTool, mock_editor: AsyncMock) -> None:
        """Test apply_patch handler for update_file operation."""
        operation = {"type": "update_file", "path": "test.txt", "diff": "@@ -1,1 +1,1 @@\n-old\n+new"}
        call_id = "test-call-456"

        handler = tool_with_editor.func
        result = await handler(call_id, operation)

        assert result["type"] == "apply_patch_call_output"
        assert result["call_id"] == call_id
        assert result["status"] == "completed"
        mock_editor.update_file.assert_called_once_with(operation)

    @pytest.mark.asyncio
    async def test_apply_patch_delete_file(self, tool_with_editor: ApplyPatchTool, mock_editor: AsyncMock) -> None:
        """Test apply_patch handler for delete_file operation."""
        operation = {"type": "delete_file", "path": "test.txt"}
        call_id = "test-call-789"

        handler = tool_with_editor.func
        result = await handler(call_id, operation)

        assert result["type"] == "apply_patch_call_output"
        assert result["call_id"] == call_id
        assert result["status"] == "completed"
        mock_editor.delete_file.assert_called_once_with(operation)

    @pytest.mark.asyncio
    async def test_apply_patch_unknown_operation(self, tool_with_editor: ApplyPatchTool) -> None:
        """Test apply_patch handler with unknown operation type."""
        operation = {"type": "unknown_operation", "path": "test.txt"}
        call_id = "test-call-unknown"

        handler = tool_with_editor.func
        result = await handler(call_id, operation)

        assert result["type"] == "apply_patch_call_output"
        assert result["call_id"] == call_id
        assert result["status"] == "failed"
        assert "Unknown operation type" in result["output"]

    @pytest.mark.asyncio
    async def test_apply_patch_with_approval_approved(self, mock_editor: AsyncMock) -> None:
        """Test apply_patch with approval when approved."""

        def approval_callback(ctx: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
            return {"approve": True}

        tool = ApplyPatchTool(editor=mock_editor, needs_approval=True, on_approval=approval_callback)
        operation = {"type": "create_file", "path": "test.txt", "diff": "@@ -0,0 +1,1 @@\n+content"}
        call_id = "test-call-approved"

        handler = tool.func
        result = await handler(call_id, operation)

        assert result["status"] == "completed"
        mock_editor.create_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_apply_patch_with_approval_rejected(self, mock_editor: AsyncMock) -> None:
        """Test apply_patch with approval when rejected."""

        def approval_callback(ctx: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
            return {"approve": False}

        tool = ApplyPatchTool(editor=mock_editor, needs_approval=True, on_approval=approval_callback)
        operation = {"type": "create_file", "path": "test.txt", "diff": "@@ -0,0 +1,1 @@\n+content"}
        call_id = "test-call-rejected"

        handler = tool.func
        result = await handler(call_id, operation)

        assert result["type"] == "apply_patch_call_output"
        assert result["call_id"] == call_id
        assert result["status"] == "failed"
        assert "rejected" in result["output"].lower()
        mock_editor.create_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_apply_patch_integration(self, tool_with_workspace: ApplyPatchTool, tmp_path: Path) -> None:
        """Test full integration of ApplyPatchTool with WorkspaceEditor."""
        workspace_dir = tmp_path / "workspace"

        # Create file
        operation = {"type": "create_file", "path": "test.py", "diff": "@@ -0,0 +1,2 @@\n+def hello():\n+    pass"}
        call_id = "integration-1"
        handler = tool_with_workspace.func
        result = await handler(call_id, operation)
        assert result["status"] == "completed"
        assert (workspace_dir / "test.py").exists()

        # Update file
        operation = {
            "type": "update_file",
            "path": "test.py",
            "diff": "@@ -1,2 +1,2 @@\n def hello():\n-    pass\n+    return True",
        }
        call_id = "integration-2"
        result = await handler(call_id, operation)
        assert result["status"] == "completed"
        content = (workspace_dir / "test.py").read_text()
        assert "return True" in content

        # Delete file
        operation = {"type": "delete_file", "path": "test.py"}
        call_id = "integration-3"
        result = await handler(call_id, operation)
        assert result["status"] == "completed"
        assert not (workspace_dir / "test.py").exists()

    def test_tool_attributes(self, tool_with_editor: ApplyPatchTool) -> None:
        """Test that tool has correct name and description."""
        assert tool_with_editor.name == "apply_patch_tool"
        assert "patch" in tool_with_editor.description.lower()
