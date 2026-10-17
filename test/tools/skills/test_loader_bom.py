# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import textwrap
from pathlib import Path

from ag2.tools.skills.runtime.local.loader import SkillLoader, parse_frontmatter, strip_frontmatter


def test_parse_frontmatter_with_bom() -> None:
    text = "\ufeff---\nname: windows-skill\ndescription: Authored with PowerShell\nversion: 1.0\n---\n# Body"
    result = parse_frontmatter(text)
    assert result.get("name") == "windows-skill"
    assert result.get("description") == "Authored with PowerShell"
    assert result.get("version") == 1.0


def test_strip_frontmatter_with_bom() -> None:
    text = "\ufeff---\nname: windows-skill\ndescription: Authored with PowerShell\n---\n# Body\nInstructions here."
    body = strip_frontmatter(text)
    assert body == "# Body\nInstructions here."
    assert "---" not in body


def test_loader_lenient_discover_with_bom(tmp_path: Path) -> None:
    skill_dir = tmp_path / "powershell-skill"
    skill_dir.mkdir(parents=True)
    content = textwrap.dedent("""\
        ---
        name: powershell-skill
        description: Skill written on Windows PowerShell with UTF-8 BOM
        version: 1.0.0
        ---
        # PowerShell Skill
        Runs tasks on Windows.
    """)
    (skill_dir / "SKILL.md").write_bytes(content.encode("utf-8-sig"))

    loader = SkillLoader(tmp_path, strict=False)
    skills = loader.discover()

    assert len(skills) == 1
    assert skills[0].name == "powershell-skill"
    assert skills[0].metadata.description == "Skill written on Windows PowerShell with UTF-8 BOM"
    assert skills[0].metadata.version == "1.0.0"


def test_loader_strict_discover_with_bom(tmp_path: Path) -> None:
    skill_dir = tmp_path / "powershell-skill"
    skill_dir.mkdir(parents=True)
    content = textwrap.dedent("""\
        ---
        name: powershell-skill
        description: Skill written on Windows PowerShell with UTF-8 BOM
        ---
        # Strict Skill
    """)
    (skill_dir / "SKILL.md").write_bytes(content.encode("utf-8-sig"))

    loader = SkillLoader(tmp_path, strict=True)
    skills = loader.discover()

    assert len(skills) == 1
    assert skills[0].name == "powershell-skill"
    assert skills[0].metadata.description == "Skill written on Windows PowerShell with UTF-8 BOM"


def test_loader_bomless_unchanged(tmp_path: Path) -> None:
    skill_dir = tmp_path / "clean-skill"
    skill_dir.mkdir(parents=True)
    content = textwrap.dedent("""\
        ---
        name: clean-skill
        description: Standard UTF-8 without BOM
        ---
        # Clean Body
    """)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")

    assert parse_frontmatter(content)["name"] == "clean-skill"
    assert strip_frontmatter(content) == "# Clean Body"

    loader = SkillLoader(tmp_path, strict=True)
    skills = loader.discover()
    assert len(skills) == 1
    assert skills[0].name == "clean-skill"
