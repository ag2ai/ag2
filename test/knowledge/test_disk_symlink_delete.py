# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

pytest.importorskip("watchdog")

from ag2.knowledge import DiskKnowledgeStore


@pytest.mark.skipif(sys.platform == "win32", reason="DiskKnowledgeStore is POSIX-only")
@pytest.mark.asyncio
@pytest.mark.parametrize("is_directory", [False, True], ids=["file", "directory"])
async def test_delete_symlink_preserves_target(tmp_path: Path, is_directory: bool) -> None:
    root = tmp_path / "store"
    root.mkdir()
    target = root / "target"
    if is_directory:
        target.mkdir()
        (target / "data.txt").write_text("keep", encoding="utf-8")
    else:
        target.write_text("keep", encoding="utf-8")

    alias = root / "alias"
    alias.symlink_to(target, target_is_directory=is_directory)

    await DiskKnowledgeStore(root).delete("/alias")

    assert not alias.is_symlink()
    if is_directory:
        assert (target / "data.txt").read_text(encoding="utf-8") == "keep"
    else:
        assert target.read_text(encoding="utf-8") == "keep"


@pytest.mark.skipif(sys.platform == "win32", reason="DiskKnowledgeStore is POSIX-only")
@pytest.mark.asyncio
@pytest.mark.parametrize("target_exists", [False, True], ids=["dangling", "external"])
async def test_delete_symlink_to_outside_root(tmp_path: Path, target_exists: bool) -> None:
    root = tmp_path / "store"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    if target_exists:
        outside.write_text("keep", encoding="utf-8")
    alias = root / "alias"
    alias.symlink_to(outside)

    await DiskKnowledgeStore(root).delete("/alias")

    assert not alias.is_symlink()
    if target_exists:
        assert outside.read_text(encoding="utf-8") == "keep"
    else:
        assert not outside.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="DiskKnowledgeStore is POSIX-only")
@pytest.mark.asyncio
async def test_delete_rejects_symlinked_parent_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    protected = outside / "protected.txt"
    protected.write_text("keep", encoding="utf-8")
    (root / "escaped").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="Path traversal blocked"):
        await DiskKnowledgeStore(root).delete("/escaped/protected.txt")

    assert protected.read_text(encoding="utf-8") == "keep"
