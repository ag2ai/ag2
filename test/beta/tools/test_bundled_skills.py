# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the bundled repository-development skills in skills/."""

import subprocess
import sys
from pathlib import Path

import pytest

from autogen.beta.tools.skills.local_skills.loader import SkillLoader

# Bundled skills live at <repo_root>/skills/
REPO_ROOT = Path(__file__).parent.parent.parent.parent
SKILLS_DIR = REPO_ROOT / "skills"

EXPECTED_SKILLS = {"python-code", "repo-structure", "docs-writer", "test-scaffold"}


@pytest.fixture
def loader() -> SkillLoader:
    return SkillLoader(SKILLS_DIR)


def test_skills_directory_exists() -> None:
    assert SKILLS_DIR.is_dir(), f"Expected .agents/skills/ at {SKILLS_DIR}"


def test_all_expected_skills_present(loader: SkillLoader) -> None:
    names = {s.name for s in loader.discover()}
    assert EXPECTED_SKILLS.issubset(names), f"Missing skills: {EXPECTED_SKILLS - names}"


@pytest.mark.parametrize("skill_name", sorted(EXPECTED_SKILLS))
def test_skill_is_valid(skill_name: str, loader: SkillLoader) -> None:
    skills = {s.name: s for s in loader.discover()}
    meta = skills[skill_name]

    assert meta.name == skill_name
    assert meta.description, f"{skill_name}: description must not be empty"
    assert len(meta.description) <= 1024, f"{skill_name}: description too long"
    assert meta.path.is_dir()
    assert (meta.path / "SKILL.md").is_file()


@pytest.mark.parametrize("skill_name", sorted(EXPECTED_SKILLS))
def test_skill_loadable(skill_name: str, loader: SkillLoader) -> None:
    content = loader.load(skill_name)

    assert content.startswith("---"), f"{skill_name}: SKILL.md must start with frontmatter"
    assert len(content) > 100, f"{skill_name}: SKILL.md content seems too short"


def test_python_code_skill_content(loader: SkillLoader) -> None:
    content = loader.load("python-code")

    assert "@tool" in content
    assert "type annotation" in content.lower() or "Type annotation" in content


def test_repo_structure_skill_content(loader: SkillLoader) -> None:
    content = loader.load("repo-structure")

    assert "autogen/beta/" in content
    assert "test/beta/" in content


def test_docs_writer_skill_content(loader: SkillLoader) -> None:
    content = loader.load("docs-writer")

    assert "frontmatter" in content.lower() or "YAML" in content
    assert "linenums" in content


def test_test_scaffold_skill_content(loader: SkillLoader) -> None:
    content = loader.load("test-scaffold")

    assert "pytest" in content
    assert "asyncio" in content


def test_test_scaffold_has_scripts(loader: SkillLoader) -> None:
    skills = {s.name: s for s in loader.discover()}
    meta = skills["test-scaffold"]

    assert meta.has_scripts, "test-scaffold must have a scripts/ directory"
    scaffold_script = meta.path / "scripts" / "scaffold.py"
    assert scaffold_script.is_file()


def test_scaffold_script_runs(loader: SkillLoader) -> None:
    script = loader.get_path("test-scaffold") / "scripts" / "scaffold.py"

    result = subprocess.run(
        [sys.executable, str(script), "autogen.beta.exceptions"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    assert result.returncode == 0, f"scaffold.py failed:\n{result.stderr}"
    assert "def test_" in result.stdout


def test_scaffold_script_unknown_module_exits_nonzero(loader: SkillLoader) -> None:
    script = loader.get_path("test-scaffold") / "scripts" / "scaffold.py"

    result = subprocess.run(
        [sys.executable, str(script), "this.module.does.not.exist"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    assert result.returncode != 0


def test_skills_discoverable_via_default_runtime_from_repo_root() -> None:
    """SkillsToolkit() from repo root should find bundled skills automatically."""
    from autogen.beta.tools.skills import LocalRuntime

    runtime = LocalRuntime(dir=str(SKILLS_DIR))
    names = {m.name for m in runtime.discover()}

    assert EXPECTED_SKILLS.issubset(names)
