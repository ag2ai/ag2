# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest


@pytest.fixture()
def server_url() -> str:
    return "http://test"


@pytest.fixture()
def local_skills_dir(tmp_path: Path) -> Path:
    """Populate ``tmp_path`` with two ``agentskills.io``-style skills."""

    def write(name: str, description: str) -> None:
        skill_dir = tmp_path / name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {description}\n---\n\nbody",
            encoding="utf-8",
        )

    write("code-review", "Review code for bugs and style")
    write("data-analysis", "Analyse CSV/JSON datasets")
    return tmp_path
