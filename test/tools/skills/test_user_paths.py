# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from ag2.tools.skills.runtime import LocalRuntime
from ag2.tools.skills.runtime.local.loader import SkillLoader


@pytest.mark.parametrize("as_path", [False, True])
@pytest.mark.parametrize("source", ["loader", "runtime", "extra_paths"])
def test_discover_skills_under_home(
    skill_tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, as_path: bool, source: str
) -> None:
    monkeypatch.setenv("HOME", str(skill_tree))
    monkeypatch.setenv("USERPROFILE", str(skill_tree))
    path = Path("~") if as_path else "~"

    if source == "loader":
        skills = SkillLoader(path).discover()
    elif source == "runtime":
        skills = LocalRuntime(path).skills
    else:
        skills = LocalRuntime(tmp_path / "empty", extra_paths=[path]).skills

    assert {skill.name for skill in skills} == {"markdown-guide", "react-best-practices"}


def test_install_skill_under_home(skill_tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(tmp_path)
    runtime = LocalRuntime("~/.agents/skills")

    runtime.ensure_storage()
    runtime.install(skill_tree / "markdown-guide", "markdown-guide")

    assert (home / ".agents/skills/markdown-guide/SKILL.md").is_file()
    assert not (tmp_path / "~").exists()
