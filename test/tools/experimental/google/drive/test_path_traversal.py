# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Google Drive download path traversal prevention.

These tests verify the _validate_download_path function in
autogen/tools/experimental/google/drive/drive_functions.py
(fix: validate subfolder_path and file_name against path traversal).

No Google API credentials or network access are required.
"""

from pathlib import Path

import pytest

from autogen.tools.experimental.google.drive.drive_functions import _validate_download_path


class TestDriveSubfolderPathTraversal:
    """Tests for subfolder_path validation against path traversal."""

    def test_parent_traversal_blocked(self, tmp_path: Path) -> None:
        """../../.ssh must not resolve inside download_folder."""
        download_folder = tmp_path / "downloads"
        download_folder.mkdir()
        with pytest.raises(ValueError, match="subfolder_path escapes"):
            _validate_download_path(download_folder, "../../.ssh", "file.txt")

    def test_relative_outside_blocked(self, tmp_path: Path) -> None:
        """../outside must not resolve inside download_folder."""
        download_folder = tmp_path / "downloads"
        download_folder.mkdir()
        with pytest.raises(ValueError, match="subfolder_path escapes"):
            _validate_download_path(download_folder, "../outside", "file.txt")

    def test_double_dot_deep_traversal_blocked(self, tmp_path: Path) -> None:
        """Many levels of ../.. must not escape download_folder."""
        download_folder = tmp_path / "downloads"
        download_folder.mkdir()
        with pytest.raises(ValueError, match="subfolder_path escapes"):
            _validate_download_path(download_folder, "sub/../../../home/attacker", "file.txt")

    def test_normal_subfolder_allowed(self, tmp_path: Path) -> None:
        """A legitimate subfolder such as 'videos' must be allowed."""
        download_folder = tmp_path / "downloads"
        download_folder.mkdir()
        result = _validate_download_path(download_folder, "videos", "clip.mp4")
        assert result.resolve().is_relative_to(download_folder.resolve())

    def test_nested_legitimate_subfolder_allowed(self, tmp_path: Path) -> None:
        """Nested subfolders like 'a/b/c' that stay inside must be allowed."""
        download_folder = tmp_path / "downloads"
        download_folder.mkdir()
        result = _validate_download_path(download_folder, "a/b/c", "file.txt")
        assert result.resolve().is_relative_to(download_folder.resolve())


class TestDriveFileNamePathTraversal:
    """Tests for file_name validation against path traversal."""

    def test_file_traversal_via_dotdot_blocked(self, tmp_path: Path) -> None:
        """../../.ssh/authorized_keys as file_name must not escape download_folder."""
        download_folder = tmp_path / "downloads"
        download_folder.mkdir()
        with pytest.raises(ValueError, match="file_name escapes"):
            _validate_download_path(download_folder, "subfolder", "../../.ssh/authorized_keys")

    def test_parent_only_traversal_blocked(self, tmp_path: Path) -> None:
        """../../../etc/passwd as file_name must be blocked."""
        download_folder = tmp_path / "downloads"
        download_folder.mkdir()
        with pytest.raises(ValueError, match="file_name escapes"):
            _validate_download_path(download_folder, "sub", "../../../etc/passwd")

    def test_normal_filename_allowed(self, tmp_path: Path) -> None:
        """A plain filename like 'clip.mp4' must be allowed."""
        download_folder = tmp_path / "downloads"
        download_folder.mkdir()
        result = _validate_download_path(download_folder, "videos", "clip.mp4")
        assert result.name == "clip.mp4"

    def test_filename_in_root_download_folder_allowed(self, tmp_path: Path) -> None:
        """File directly in download_folder (no subfolder) must be allowed."""
        download_folder = tmp_path / "downloads"
        download_folder.mkdir()
        result = _validate_download_path(download_folder, None, "report.pdf")
        assert result.name == "report.pdf"

    def test_file_and_subfolder_combined_allowed(self, tmp_path: Path) -> None:
        """Legitimate subfolder + filename combination must be allowed."""
        download_folder = tmp_path / "downloads"
        download_folder.mkdir()
        result = _validate_download_path(download_folder, "docs/q1", "summary.docx")
        assert result.name == "summary.docx"
        assert result.resolve().is_relative_to(download_folder.resolve())
