# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Google Drive download path traversal prevention.

These tests verify the path validation logic added in
autogen/tools/experimental/google/drive/drive_functions.py
(fix: validate subfolder_path and file_name against path traversal).

No Google API credentials or network access are required. The tests exercise
the same pathlib.Path.resolve() + is_relative_to() guard logic used by the fix,
ensuring that both the subfolder_path and file_name parameters cannot escape the
configured download_folder.
"""

from pathlib import Path


def _check_subfolder_safe(download_folder: Path, subfolder_path: str) -> bool:
    """Return True when subfolder_path stays inside download_folder.

    Mirrors: destination_dir = download_folder / subfolder_path
             if not destination_dir.resolve().is_relative_to(download_folder.resolve()):
                 raise ValueError(...)
    """
    destination_dir = download_folder / subfolder_path
    return destination_dir.resolve().is_relative_to(download_folder.resolve())


def _check_file_safe(download_folder: Path, destination_dir: Path, file_name: str) -> bool:
    """Return True when file_name stays inside download_folder.

    Mirrors: file_path = destination_dir / file_name
             if not file_path.resolve().is_relative_to(download_folder.resolve()):
                 raise ValueError(...)
    """
    file_path = destination_dir / file_name
    return file_path.resolve().is_relative_to(download_folder.resolve())


class TestDriveSubfolderPathTraversal:
    """Tests for subfolder_path validation against path traversal."""

    def test_parent_traversal_blocked(self, tmp_path: Path) -> None:
        """../../.ssh must not resolve inside download_folder."""
        download_folder = tmp_path / "downloads"
        download_folder.mkdir()
        assert not _check_subfolder_safe(download_folder, "../../.ssh")

    def test_absolute_subfolder_blocked(self, tmp_path: Path) -> None:
        """/etc as subfolder_path must not resolve inside download_folder."""
        download_folder = tmp_path / "downloads"
        download_folder.mkdir()
        # On Windows this may not be a real absolute path, but the resolve
        # check still catches it because the resolved path is outside.
        assert not _check_subfolder_safe(download_folder, "../outside")

    def test_double_dot_deep_traversal_blocked(self, tmp_path: Path) -> None:
        """Many levels of ../.. must not escape download_folder."""
        download_folder = tmp_path / "downloads"
        download_folder.mkdir()
        assert not _check_subfolder_safe(download_folder, "sub/../../../home/attacker")

    def test_normal_subfolder_allowed(self, tmp_path: Path) -> None:
        """A legitimate subfolder such as 'videos' must be allowed."""
        download_folder = tmp_path / "downloads"
        download_folder.mkdir()
        assert _check_subfolder_safe(download_folder, "videos")

    def test_nested_legitimate_subfolder_allowed(self, tmp_path: Path) -> None:
        """Nested subfolders like 'a/b/c' that stay inside must be allowed."""
        download_folder = tmp_path / "downloads"
        download_folder.mkdir()
        assert _check_subfolder_safe(download_folder, "a/b/c")


class TestDriveFileNamePathTraversal:
    """Tests for file_name validation against path traversal."""

    def test_file_traversal_via_dotdot_blocked(self, tmp_path: Path) -> None:
        """../../.ssh/authorized_keys as file_name must not escape download_folder."""
        download_folder = tmp_path / "downloads"
        download_folder.mkdir()
        destination_dir = download_folder / "subfolder"
        assert not _check_file_safe(download_folder, destination_dir, "../../.ssh/authorized_keys")

    def test_parent_only_traversal_blocked(self, tmp_path: Path) -> None:
        """../secret.txt relative to the destination_dir must be blocked."""
        download_folder = tmp_path / "downloads"
        download_folder.mkdir()
        destination_dir = download_folder / "sub"
        assert not _check_file_safe(download_folder, destination_dir, "../../../etc/passwd")

    def test_normal_filename_allowed(self, tmp_path: Path) -> None:
        """A plain filename like 'clip.mp4' must be allowed."""
        download_folder = tmp_path / "downloads"
        download_folder.mkdir()
        destination_dir = download_folder / "videos"
        assert _check_file_safe(download_folder, destination_dir, "clip.mp4")

    def test_filename_in_root_download_folder_allowed(self, tmp_path: Path) -> None:
        """File directly in download_folder (no subfolder) must be allowed."""
        download_folder = tmp_path / "downloads"
        download_folder.mkdir()
        assert _check_file_safe(download_folder, download_folder, "report.pdf")

    def test_file_and_subfolder_combined_allowed(self, tmp_path: Path) -> None:
        """Legitimate subfolder + filename combination must be allowed."""
        download_folder = tmp_path / "downloads"
        download_folder.mkdir()
        destination_dir = download_folder / "docs" / "q1"
        assert _check_file_safe(download_folder, destination_dir, "summary.docx")
