# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

from autogen._website.utils import copy_files


def test_copy_files(tmp_path: Path) -> None:
    # Create source directory structure
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    # Create some test files in source
    test_files = [src_dir / "file1.txt", src_dir / "subdir" / "file2.txt"]

    # Create the subdir and files
    (src_dir / "subdir").mkdir(parents=True)
    for file in test_files:
        file.write_text("test content")

    # Create destination directory
    dst_dir = tmp_path / "dst"
    dst_dir.mkdir()

    # Call the function
    copy_files(src_dir, dst_dir, test_files)

    # Verify files were copied correctly
    for src_file in test_files:
        dst_file = dst_dir / src_file.relative_to(src_dir)
        assert dst_file.exists()
        assert dst_file.read_text() == "test content"
