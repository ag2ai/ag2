# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MCP resource URI path sanitization.

These tests verify the path sanitization logic in autogen/mcp/mcp_client.py
(fix: sanitize MCP resource URI to prevent path traversal). No MCP server is
required -- tests exercise the extracted _sanitize_resource_filename function
directly.
"""

from pathlib import Path

from autogen.mcp.mcp_client import _sanitize_resource_filename


class TestMCPResourcePathSanitization:
    """Tests for URI-to-filename sanitization in MCP resource downloads."""

    # ------------------------------------------------------------------
    # Traversal attack cases -- must be blocked
    # ------------------------------------------------------------------

    def test_absolute_file_uri_sanitized_to_flat_filename(self, tmp_path: Path) -> None:
        """file:///etc/cron.d/backdoor must resolve inside download_folder."""
        result = _sanitize_resource_filename("file:///etc/cron.d/backdoor", tmp_path, "20260316")
        assert result.parent == tmp_path
        assert "etc" not in str(result)
        assert result.name.startswith("backdoor_")

    def test_relative_traversal_uri_sanitized(self, tmp_path: Path) -> None:
        """res://../../tmp/pwn must not escape download_folder."""
        result = _sanitize_resource_filename("res://../../tmp/pwn", tmp_path, "20260316")
        assert result.parent == tmp_path
        # basename extraction strips all directory components including ../
        assert result.name.startswith("pwn_")

    def test_windows_backslash_traversal_sanitized(self, tmp_path: Path) -> None:
        """Backslash traversal (Windows-style) must be sanitized."""
        result = _sanitize_resource_filename(r"res://..\..\.ssh\id_rsa", tmp_path, "20260316")
        assert result.parent == tmp_path

    def test_nested_path_stripped_to_basename(self, tmp_path: Path) -> None:
        """Deep nested path in URI keeps only the last component."""
        result = _sanitize_resource_filename("res://a/b/c/d/e/target.txt", tmp_path, "20260316")
        assert result.parent == tmp_path
        assert result.name.startswith("target.txt_")

    def test_empty_path_component_uses_fallback_name(self, tmp_path: Path) -> None:
        """When URI has no path component, fallback name 'resource' is used."""
        result = _sanitize_resource_filename("res://", tmp_path, "20260316")
        assert result.parent == tmp_path
        assert result.name.startswith("resource_")

    # ------------------------------------------------------------------
    # Normal URI cases -- must succeed
    # ------------------------------------------------------------------

    def test_normal_flat_filename_uri_works(self, tmp_path: Path) -> None:
        """res://data/report.json must resolve cleanly inside download_folder."""
        result = _sanitize_resource_filename("res://data/report.json", tmp_path, "20260316")
        assert result.parent == tmp_path
        assert "report.json" in result.name

    def test_simple_filename_uri_works(self, tmp_path: Path) -> None:
        """file://output.txt (no directory component) must work."""
        result = _sanitize_resource_filename("file://output.txt", tmp_path, "20260316")
        assert result.parent == tmp_path
        assert "output.txt" in result.name

    def test_timestamp_appended_to_filename(self, tmp_path: Path) -> None:
        """Timestamp suffix must be present in the generated filename."""
        result = _sanitize_resource_filename("res://report.json", tmp_path, "20260316120000")
        assert result.name == "report.json_20260316120000"

    def test_result_is_absolute_path(self, tmp_path: Path) -> None:
        """Returned path must be absolute (resolved)."""
        result = _sanitize_resource_filename("res://data.txt", tmp_path, "ts")
        assert result.is_absolute()

    def test_result_is_relative_to_download_folder(self, tmp_path: Path) -> None:
        """Returned path must be a descendant of download_folder."""
        result = _sanitize_resource_filename("res://data.txt", tmp_path, "ts")
        assert result.is_relative_to(tmp_path.resolve())
