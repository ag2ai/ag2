"""Shared utilities for artifact installers."""

from __future__ import annotations

import shutil
from pathlib import Path


def copy_tree(source: Path, dest: Path) -> list[Path]:
    """Copy a directory tree, returning list of created files."""
    dest.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for src_file in sorted(source.rglob("*")):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(source)
        dst_file = dest / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        created.append(dst_file)
    return created
