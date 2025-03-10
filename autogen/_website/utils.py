# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
from pathlib import Path
from typing import Optional, TypedDict, Union


class NavigationGroup(TypedDict):
    group: str
    pages: list[Union[str, "NavigationGroup"]]


def get_git_tracked_and_untracked_files_in_directory(directory: Path) -> list[Path]:
    """Get all files in the directory that are tracked by git or newly added."""
    proc = subprocess.run(
        ["git", "-C", str(directory), "ls-files", "--others", "--exclude-standard", "--cached"],
        capture_output=True,
        text=True,
        check=True,
    )
    return list({directory / p for p in proc.stdout.splitlines()})


def copy_files(src_dir: Path, dst_dir: Path, files_to_copy: list[Path]) -> None:
    """Copy files from src_dir to dst_dir."""
    for file in files_to_copy:
        if file.is_file():
            dst = dst_dir / file.relative_to(src_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dst)


def copy_only_git_tracked_and_untracked_files(src_dir: Path, dst_dir: Path, ignore_dir: Optional[str] = None) -> None:
    """Copy only the files that are tracked by git or newly added from src_dir to dst_dir."""
    tracked_and_new_files = get_git_tracked_and_untracked_files_in_directory(src_dir)

    if ignore_dir:
        ignore_dir_rel_path = src_dir / ignore_dir

        tracked_and_new_files = list({
            file for file in tracked_and_new_files if not any(parent == ignore_dir_rel_path for parent in file.parents)
        })

    copy_files(src_dir, dst_dir, tracked_and_new_files)
