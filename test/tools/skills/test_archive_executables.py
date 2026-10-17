# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import io
import os
import stat
import tarfile
from pathlib import Path

import pytest

from ag2 import Context
from ag2.stream import MemoryStream
from ag2.tools.skills import LocalRuntime
from ag2.tools.skills.skill_search.extractor import extract_skill


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == "nt", reason="POSIX executable permissions")
@pytest.mark.parametrize("helper_mode", [0o755, 0o4755])
async def test_installed_script_can_execute_its_bundled_helper(tmp_path: Path, helper_mode: int) -> None:
    archive = tmp_path / "skill.tar.gz"
    entries = [
        ("SKILL.md", b"---\nname: helper-skill\ndescription: Run a bundled helper\n---\n", 0o644),
        ("scripts/run.sh", b"#!/bin/sh\n./helper\n", 0o755),
        ("scripts/helper", b"#!/bin/sh\nprintf 'helper output'\n", helper_mode),
        ("references/note.txt", b"Not executable", 0o644),
    ]
    with tarfile.open(archive, "w:gz") as tar:
        for name, data, mode in entries:
            member = tarfile.TarInfo(f"repo-main/{name}")
            member.size = len(data)
            member.mode = mode
            tar.addfile(member, io.BytesIO(data))
    install_dir = tmp_path / "skills"
    extract_skill(archive, "", install_dir)
    runtime = LocalRuntime(install_dir)

    output = await runtime.execute("helper-skill", "run.sh", Context(stream=MemoryStream()))

    assert output == "helper output"
    skill_dir = install_dir / "helper-skill"
    assert stat.S_IMODE((skill_dir / "scripts/helper").stat().st_mode) == 0o755
    assert (skill_dir / "references/note.txt").stat().st_mode & 0o111 == 0
