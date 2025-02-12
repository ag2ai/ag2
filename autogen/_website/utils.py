# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
from pathlib import Path


def get_git_tracked_and_untracked_files_in_directory(directory: Path) -> set[Path]:
    """Get all files in the directory that are tracked by git or newly added."""
    proc = subprocess.run(
        ["git", "-C", str(directory), "ls-files", "--others", "--exclude-standard", "--cached"],
        capture_output=True,
        text=True,
        check=True,
    )
    return {directory / p for p in proc.stdout.splitlines()}


def copy_only_git_tracked_and_untracked_files(src_dir: Path, dst_dir: Path) -> None:
    """Copy only the files that are tracked by git or newly added from src_dir to dst_dir."""
    tracked_and_new_files = get_git_tracked_and_untracked_files_in_directory(src_dir)

    for src in tracked_and_new_files:
        if src.is_file():
            dst = dst_dir / src.relative_to(src_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
