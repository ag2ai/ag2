# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import textwrap
from pathlib import Path

import pytest

from ag2.exceptions import InvalidSkillError, InvalidSkillNameError, SkillNotFoundError
from ag2.tools.skills.runtime.local.loader import SkillLoader, parse_frontmatter


def test_parse_frontmatter_basic() -> None:
    text = "---\nname: my-skill\ndescription: A great skill\nversion: 2.0\n---\nBody"
    result = parse_frontmatter(text)
    assert result["name"] == "my-skill"
    assert result["description"] == "A great skill"
    assert result["version"] == 2.0  # yaml.safe_load parses numbers


def test_parse_frontmatter_no_header() -> None:
    assert parse_frontmatter("No frontmatter here") == {}


def test_parse_frontmatter_unclosed() -> None:
    assert parse_frontmatter("---\nname: broken\n") == {}


def test_parse_frontmatter_quoted_values() -> None:
    text = '---\nname: "my-skill"\ndescription: "A skill with: colons"\n---\nBody'
    result = parse_frontmatter(text)
    assert result["name"] == "my-skill"
    assert result["description"] == "A skill with: colons"


def test_parse_frontmatter_multiline_description() -> None:
    text = "---\nname: my-skill\ndescription: >\n  A long\n  description\n---\nBody"
    result = parse_frontmatter(text)
    assert "A long" in str(result["description"])


def test_loader_discover_names(skill_tree: Path) -> None:
    loader = SkillLoader(skill_tree)

    names = {s.name for s in loader.discover()}

    assert names == {"react-best-practices", "markdown-guide"}


def test_loader_discover_metadata(skill_tree: Path) -> None:
    loader = SkillLoader(skill_tree)

    skills = {s.name: s for s in loader.discover()}

    assert skills["react-best-practices"].metadata.description == "Best practices for React development"
    assert skills["react-best-practices"].metadata.version == "1.2.0"
    assert [s.name for s in skills["react-best-practices"].scripts] == ["scaffold.py"]

    assert skills["markdown-guide"].metadata.description == "Guide for writing Markdown"
    assert skills["markdown-guide"].metadata.version is None
    assert skills["markdown-guide"].scripts == ()


def test_loader_priority(tmp_path: Path) -> None:
    """First path wins when the same skill name appears in multiple paths."""
    for name in ("project", "user"):
        skill_dir = tmp_path / name / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: my-skill\ndescription: from {name}\n---\n",
            encoding="utf-8",
        )

    loader = SkillLoader(tmp_path / "project", tmp_path / "user")
    [meta] = loader.discover()

    assert meta.metadata.description == "from project"


def test_loader_nonexistent_path(tmp_path: Path) -> None:
    loader = SkillLoader(tmp_path / "no-such-dir")

    assert loader.discover() == []


def test_loader_load(skill_tree: Path) -> None:
    loader = SkillLoader(skill_tree)

    content = loader.load("react-best-practices")

    assert "React Best Practices" in content
    assert "functional components" in content


def test_loader_load_missing(skill_tree: Path) -> None:
    loader = SkillLoader(skill_tree)

    with pytest.raises(SkillNotFoundError, match="nonexistent"):
        loader.load("nonexistent")


def test_loader_get_path(skill_tree: Path) -> None:
    loader = SkillLoader(skill_tree)

    path = loader.get_path("react-best-practices")

    assert path == skill_tree / "react-best-practices"


def test_loader_rejects_invalid_skill_name(skill_tree: Path) -> None:
    loader = SkillLoader(skill_tree)
    with pytest.raises(InvalidSkillNameError):
        loader.load("../react-best-practices")


def test_parse_frontmatter_recovers_unquoted_colon() -> None:
    # A value containing a colon is invalid YAML; the best-effort fallback
    # quotes it and recovers (cross-client compatibility).
    text = "---\nname: my-skill\ndescription: Use this skill when: the user asks\n---\nBody"
    result = parse_frontmatter(text)
    assert result["name"] == "my-skill"
    assert result["description"] == "Use this skill when: the user asks"


def _write_skill(base: Path, dir_name: str, body: str) -> None:
    skill_dir = base / dir_name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")


def test_loader_lenient_warns_but_loads_name_mismatch(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # Per the agentskills.io client guide, a name that doesn't match the
    # directory is a cosmetic issue: warn but load anyway (cross-client compat).
    _write_skill(tmp_path, "good", "---\nname: good\ndescription: A valid skill\n---\n")
    _write_skill(tmp_path, "bad-dir", "---\nname: other-name\ndescription: Mismatched\n---\n")

    loader = SkillLoader(tmp_path, strict=False)
    with caplog.at_level("WARNING"):
        names = {m.name for m in loader.discover()}

    assert names == {"good", "other-name"}
    assert "does not match directory 'bad-dir'" in caplog.text


def test_loader_lenient_skips_empty_description(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _write_skill(tmp_path, "no-desc", "---\nname: no-desc\n---\n")

    loader = SkillLoader(tmp_path, strict=False)
    with caplog.at_level("WARNING"):
        skills = loader.discover()

    assert skills == []
    assert "Skipping skill 'no-desc'" in caplog.text
    assert "description" in caplog.text


def test_loader_collision_warns_and_first_wins(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    for scope in ("project", "user"):
        _write_skill(tmp_path / scope, "my-skill", f"---\nname: my-skill\ndescription: from {scope}\n---\n")

    loader = SkillLoader(tmp_path / "project", tmp_path / "user", strict=False)
    with caplog.at_level("WARNING"):
        [meta] = loader.discover()

    assert meta.metadata.description == "from project"
    assert "shadowed" in caplog.text


def test_loader_strict_requires_name_and_description(tmp_path: Path) -> None:
    skill_dir = tmp_path / "no-frontmatter-required-fields"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nlicense: Apache-2.0\n---\n", encoding="utf-8")

    loader = SkillLoader(tmp_path, strict=True)
    with pytest.raises(InvalidSkillError, match="missing required frontmatter field"):
        loader.discover()


def test_loader_strict_rejects_mismatched_name(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skill-dir-name"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        textwrap.dedent("""\
            ---
            name: different-name
            description: Valid description.
            ---
        """),
        encoding="utf-8",
    )

    loader = SkillLoader(tmp_path, strict=True)
    with pytest.raises(InvalidSkillError, match="must match directory name"):
        loader.discover()


def test_loader_cache_avoids_rescan(skill_tree: Path) -> None:
    loader = SkillLoader(skill_tree)

    first = loader.discover()
    # Add a new skill after first scan
    new_dir = skill_tree / "new-skill"
    new_dir.mkdir()
    (new_dir / "SKILL.md").write_text("---\nname: new-skill\ndescription: New\n---\n")

    # Should return cached result (no new-skill)
    second = loader.discover()
    assert {m.name for m in second} == {m.name for m in first}


def test_loader_invalidate_forces_rescan(skill_tree: Path) -> None:
    loader = SkillLoader(skill_tree)

    loader.discover()
    # Add a new skill
    new_dir = skill_tree / "new-skill"
    new_dir.mkdir()
    (new_dir / "SKILL.md").write_text("---\nname: new-skill\ndescription: New\n---\n")

    loader.invalidate()
    refreshed = loader.discover()
    assert "new-skill" in {m.name for m in refreshed}
