# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import re
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


def remove_marker_blocks(content: str, marker_prefix: str) -> str:
    """Remove marker blocks from the content.

    Args:
        content: The source content to process
        marker_prefix: The marker prefix to identify blocks to remove (without the START/END suffix)

    Returns:
        Processed content with appropriate blocks handled
    """
    # First, remove blocks with the specified marker completely
    if f"{{/* {marker_prefix}-START */}}" in content:
        pattern = rf"\{{/\* {re.escape(marker_prefix)}-START \*/\}}.*?\{{/\* {re.escape(marker_prefix)}-END \*/\}}"
        content = re.sub(pattern, "", content, flags=re.DOTALL)

    # Now, remove markers but keep content for the other marker type
    other_prefix = (
        "DELETE-ME-WHILE-BUILDING-MKDOCS"
        if marker_prefix == "DELETE-ME-WHILE-BUILDING-MINTLIFY"
        else "DELETE-ME-WHILE-BUILDING-MINTLIFY"
    )

    # Remove start markers
    start_pattern = rf"\{{/\* {re.escape(other_prefix)}-START \*/\}}\s*"
    content = re.sub(start_pattern, "", content)

    # Remove end markers
    end_pattern = rf"\s*\{{/\* {re.escape(other_prefix)}-END \*/\}}"
    content = re.sub(end_pattern, "", content)

    # Fix any double newlines that might have been created
    content = re.sub(r"\n\s*\n\s*\n", "\n\n", content)

    return content
