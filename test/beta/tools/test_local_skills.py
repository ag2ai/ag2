# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import textwrap
from dataclasses import asdict
from pathlib import Path

import pytest
from dirty_equals import IsPartialDict

from autogen.beta.context import ConversationContext
from autogen.beta.exceptions import InvalidSkillError, InvalidSkillNameError, SkillNotFoundError
from autogen.beta.tools.shell.environment.local import LocalShellEnvironment
from autogen.beta.tools.toolkits.skills.local_skills import SkillsToolkit
from autogen.beta.tools.toolkits.skills.local_skills.loader import SkillLoader, parse_frontmatter
from autogen.beta.tools.toolkits.skills.runtime import LocalRuntime


@pytest.fixture
def skill_tree(tmp_path: Path) -> Path:
    """Minimal skill tree for testing.

    Structure::

        tmp_path/
          react-best-practices/
            SKILL.md  (has version, has scripts/)
            scripts/
              scaffold.py
          markdown-guide/
            SKILL.md  (no version, no scripts/)
    """
    react_dir = tmp_path / "react-best-practices"
    react_dir.mkdir(parents=True)
    (react_dir / "SKILL.md").write_text(
        textwrap.dedent("""\
            ---
            name: react-best-practices
            description: Best practices for React development
            version: 1.2.0
            ---
            # React Best Practices
            Use functional components and hooks.
        """),
        encoding="utf-8",
    )
    scripts_dir = react_dir / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "scaffold.py").write_text('print("scaffold")\n', encoding="utf-8")

    md_dir = tmp_path / "markdown-guide"
    md_dir.mkdir(parents=True)
    (md_dir / "SKILL.md").write_text(
        textwrap.dedent("""\
            ---
            name: markdown-guide
            description: Guide for writing Markdown
            ---
            # Markdown Guide
            Use headings, lists, and code blocks.
        """),
        encoding="utf-8",
    )

    return tmp_path


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


# ---------------------------------------------------------------------------
# SkillLoader — discover
# ---------------------------------------------------------------------------


def test_loader_discover_names(skill_tree: Path) -> None:
    loader = SkillLoader(skill_tree)

    names = {s.name for s in loader.discover()}

    assert names == {"react-best-practices", "markdown-guide"}


def test_loader_discover_metadata(skill_tree: Path) -> None:
    loader = SkillLoader(skill_tree)

    skills = {s.name: s for s in loader.discover()}

    assert skills["react-best-practices"].description == "Best practices for React development"
    assert skills["react-best-practices"].version == "1.2.0"
    assert skills["react-best-practices"].has_scripts is True

    assert skills["markdown-guide"].description == "Guide for writing Markdown"
    assert skills["markdown-guide"].version is None
    assert skills["markdown-guide"].has_scripts is False


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

    assert meta.description == "from project"


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


@pytest.mark.asyncio
async def test_tool_exposes_three_functions(skill_tree: Path, context: ConversationContext) -> None:
    tool = SkillsToolkit(runtime=LocalRuntime(dir=skill_tree))

    schemas = await tool.schemas(context)

    assert len(schemas) == 3
    names = {s.function.name for s in schemas}  # type: ignore[union-attr]
    assert names == {"list_skills", "load_skill", "run_skill_script"}


@pytest.mark.asyncio
async def test_run_skill_script_schema(skill_tree: Path, context: ConversationContext) -> None:
    run_tool = SkillsToolkit(LocalRuntime(dir=skill_tree)).run_skill_script

    [schema] = await run_tool.schemas(context)

    assert asdict(schema) == {
        "type": "function",
        "function": IsPartialDict({
            "name": "run_skill_script",
            "parameters": IsPartialDict({
                "properties": IsPartialDict({
                    "name": IsPartialDict({"type": "string"}),
                    "script": IsPartialDict({"type": "string"}),
                }),
                "required": ["name", "script"],
            }),
        }),
    }


def test_run_skill_script_executes(skill_tree: Path) -> None:

    scripts_dir = skill_tree / "react-best-practices" / "scripts"
    env = LocalShellEnvironment(path=scripts_dir, cleanup=False)

    # cwd is scripts_dir, so pass just the filename — same as tool.py does
    result = env.run("python scaffold.py")

    assert "scaffold" in result
