# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import io
import json
import tarfile
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from autogen.beta.context import ConversationContext
from autogen.beta.exceptions import InvalidSkillError, SkillInstallError
from autogen.beta.tools.runtime import LocalRuntime, SkillMetadata
from autogen.beta.tools.toolkits.skill_search import SkillSearchToolset
from autogen.beta.tools.toolkits.skill_search.client import SkillsClient
from autogen.beta.tools.toolkits.skill_search.extractor import extract_skill
from autogen.beta.tools.toolkits.skill_search.lock import SkillsLock

MONOREPO_SKILL_MD = textwrap.dedent("""\
    ---
    name: vercel-react-best-practices
    description: React and Next.js performance optimization guidelines
    version: 1.0.0
    ---
    # React Best Practices
    Use functional components and hooks.
""")

STANDALONE_SKILL_MD = textwrap.dedent("""\
    ---
    name: last30days
    description: Last 30 days analytics script
    ---
    # Last 30 Days
    Run last30days.py to get analytics.
""")


def _make_tarball(entries: dict[str, bytes | str]) -> bytes:
    """Build an in-memory .tar.gz archive from a name->content mapping."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in entries.items():
            data = content.encode() if isinstance(content, str) else content
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _monorepo_tarball(skill_id: str = "react-best-practices") -> bytes:
    return _make_tarball({
        f"owner-repo-abc123/skills/{skill_id}/SKILL.md": MONOREPO_SKILL_MD,
        f"owner-repo-abc123/skills/{skill_id}/rules/rule1.md": "# Rule 1\nDo good.",
    })


def _standalone_tarball() -> bytes:
    return _make_tarball({
        "owner-repo-abc123/SKILL.md": STANDALONE_SKILL_MD,
        "owner-repo-abc123/scripts/last30days.py": 'print("hello")\n',
    })


def _make_meta(tmp_path: Path, name: str = "vercel-react-best-practices") -> SkillMetadata:
    """Build a SkillMetadata for mocked install tests."""
    skill_dir = tmp_path / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    return SkillMetadata(
        name=name,
        description="React and Next.js performance optimization guidelines",
        path=skill_dir,
        has_scripts=False,
        version="1.0.0",
    )


# ---------------------------------------------------------------------------
# extract_skill
# ---------------------------------------------------------------------------


def test_extract_skill_monorepo(tmp_path: Path) -> None:
    dest = tmp_path / "skills"
    dest.mkdir()
    tar_path = tmp_path / "skill.tar.gz"
    tar_path.write_bytes(_monorepo_tarball())

    meta = extract_skill(tar_path, "react-best-practices", dest)

    assert meta.name == "vercel-react-best-practices"
    assert meta.description == "React and Next.js performance optimization guidelines"
    assert meta.version == "1.0.0"
    skill_dir = dest / "vercel-react-best-practices"
    assert (skill_dir / "SKILL.md").exists()
    assert (skill_dir / "rules" / "rule1.md").exists()


def test_extract_skill_standalone(tmp_path: Path) -> None:
    dest = tmp_path / "skills"
    dest.mkdir()
    tar_path = tmp_path / "skill.tar.gz"
    tar_path.write_bytes(_standalone_tarball())

    meta = extract_skill(tar_path, "", dest)

    assert meta.name == "last30days"
    assert meta.has_scripts is True
    assert (dest / "last30days" / "SKILL.md").exists()
    assert (dest / "last30days" / "scripts" / "last30days.py").exists()


def test_extract_skill_no_skill_md_raises(tmp_path: Path) -> None:
    dest = tmp_path / "skills"
    dest.mkdir()
    tar_path = tmp_path / "skill.tar.gz"
    tar_path.write_bytes(_make_tarball({"owner-repo-abc123/README.md": "# Nothing\n"}))

    with pytest.raises(SkillInstallError, match="No SKILL.md found"):
        extract_skill(tar_path, "", dest)


def test_extract_skill_excludes_git_dir(tmp_path: Path) -> None:
    dest = tmp_path / "skills"
    dest.mkdir()
    tar_path = tmp_path / "skill.tar.gz"
    tar_path.write_bytes(
        _make_tarball({
            "owner-repo-abc123/SKILL.md": STANDALONE_SKILL_MD,
            "owner-repo-abc123/.git/config": "[core]\nbare = false\n",
        })
    )

    extract_skill(tar_path, "", dest)

    assert not (dest / "last30days" / ".git").exists()


def test_extract_skill_overwrites_existing(tmp_path: Path) -> None:
    dest = tmp_path / "skills"
    dest.mkdir()
    (dest / "last30days").mkdir()
    (dest / "last30days" / "stale.txt").write_text("old")
    tar_path = tmp_path / "skill.tar.gz"
    tar_path.write_bytes(_standalone_tarball())

    extract_skill(tar_path, "", dest)

    assert not (dest / "last30days" / "stale.txt").exists()
    assert (dest / "last30days" / "SKILL.md").exists()


def test_extract_skill_validates_metadata(tmp_path: Path) -> None:
    """Extracted skill with invalid name format should raise."""
    dest = tmp_path / "skills"
    dest.mkdir()
    bad_skill_md = "---\nname: INVALID_NAME\ndescription: Bad\n---\n"
    tar_path = tmp_path / "skill.tar.gz"
    tar_path.write_bytes(_make_tarball({"owner-repo-abc123/SKILL.md": bad_skill_md}))

    with pytest.raises(InvalidSkillError):
        extract_skill(tar_path, "", dest)


# ---------------------------------------------------------------------------
# search_skills  (patch SkillsClient.search at class level)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_skills_formats_output(tmp_path: Path) -> None:
    skills_data = [
        {
            "skillId": "react-best-practices",
            "name": "vercel-react-best-practices",
            "installs": 229780,
            "source": "vercel-labs/agent-skills",
        },
        {"skillId": "nextjs-patterns", "name": "nextjs-patterns", "installs": 5000, "source": "some-user/nextjs-skill"},
    ]
    with patch.object(SkillsClient, "search", AsyncMock(return_value=skills_data)):
        toolset = SkillSearchToolset(runtime=LocalRuntime(dir=tmp_path / "skills"))
        result = await toolset.search_skills.model.call(query="react")

    assert 'Found 2 skill(s) for "react"' in result
    assert "vercel-react-best-practices" in result
    assert "229,780 installs" in result
    assert 'install_skill("vercel-labs/agent-skills/react-best-practices")' in result


@pytest.mark.asyncio
async def test_search_skills_no_results(tmp_path: Path) -> None:
    with patch.object(SkillsClient, "search", AsyncMock(return_value=[])):
        toolset = SkillSearchToolset(runtime=LocalRuntime(dir=tmp_path / "skills"))
        result = await toolset.search_skills.model.call(query="xyzzy-nonexistent")

    assert "No skills found" in result


@pytest.mark.asyncio
async def test_search_skills_network_error(tmp_path: Path) -> None:
    with patch.object(SkillsClient, "search", AsyncMock(side_effect=Exception("connection refused"))):
        toolset = SkillSearchToolset(runtime=LocalRuntime(dir=tmp_path / "skills"))
        result = await toolset.search_skills.model.call(query="react")

    assert "Error searching skills.sh" in result
    assert "connection refused" in result


# ---------------------------------------------------------------------------
# install_skill  (patch SkillsClient.download_skill at class level)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_install_skill_monorepo(tmp_path: Path) -> None:
    meta = _make_meta(tmp_path)
    with patch.object(SkillsClient, "download_skill", AsyncMock(return_value=(meta, "abc123hash"))):
        toolset = SkillSearchToolset(runtime=LocalRuntime(dir=tmp_path / "skills"))
        result = await toolset.install_skill.model.call(skill_id="vercel-labs/agent-skills/react-best-practices")

    assert "Installed: vercel-react-best-practices" in result
    assert "Description:" in result
    assert 'load_skill("vercel-react-best-practices")' in result


@pytest.mark.asyncio
async def test_install_skill_standalone(tmp_path: Path) -> None:
    meta = _make_meta(tmp_path, name="last30days")
    with patch.object(SkillsClient, "download_skill", AsyncMock(return_value=(meta, "def456hash"))):
        toolset = SkillSearchToolset(runtime=LocalRuntime(dir=tmp_path / "skills"))
        result = await toolset.install_skill.model.call(skill_id="mvanhorn/last30days-skill")

    assert "Installed: last30days" in result


@pytest.mark.asyncio
async def test_install_skill_records_hash(tmp_path: Path) -> None:
    """install_skill should write hash to skills-lock.json."""
    install_dir = tmp_path / "skills"
    meta = _make_meta(tmp_path)
    with patch.object(SkillsClient, "download_skill", AsyncMock(return_value=(meta, "abc123hash"))):
        toolset = SkillSearchToolset(runtime=LocalRuntime(dir=install_dir))
        await toolset.install_skill.model.call(skill_id="vercel-labs/agent-skills/react-best-practices")

    lock_path = install_dir / "skills-lock.json"
    assert lock_path.exists()
    lock_data = json.loads(lock_path.read_text())
    assert lock_data["skills"]["vercel-react-best-practices"]["computedHash"] == "abc123hash"
    assert lock_data["skills"]["vercel-react-best-practices"]["source"] == "vercel-labs/agent-skills"


@pytest.mark.asyncio
async def test_install_skill_rate_limit(tmp_path: Path) -> None:
    err = RuntimeError("GitHub rate limit exceeded. Set GITHUB_TOKEN")
    with patch.object(SkillsClient, "download_skill", AsyncMock(side_effect=err)):
        toolset = SkillSearchToolset(runtime=LocalRuntime(dir=tmp_path / "skills"))
        result = await toolset.install_skill.model.call(skill_id="some/repo/skill")

    assert "rate limit" in result.lower()


@pytest.mark.asyncio
async def test_install_skill_not_found(tmp_path: Path) -> None:
    err = RuntimeError("Skill not found: no-such/repo")
    with patch.object(SkillsClient, "download_skill", AsyncMock(side_effect=err)):
        toolset = SkillSearchToolset(runtime=LocalRuntime(dir=tmp_path / "skills"))
        result = await toolset.install_skill.model.call(skill_id="no-such/repo/skill")

    assert "not found" in result.lower()


@pytest.mark.asyncio
async def test_install_skill_invalid_id(tmp_path: Path) -> None:
    toolset = SkillSearchToolset(runtime=LocalRuntime(dir=tmp_path / "skills"))
    result = await toolset.install_skill.model.call(skill_id="invalid")

    assert "Invalid skill_id format" in result


# ---------------------------------------------------------------------------
# remove_skill
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remove_skill_success(tmp_path: Path) -> None:
    install_dir = tmp_path / "skills"
    skill_dir = install_dir / "my-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\n")

    toolset = SkillSearchToolset(runtime=LocalRuntime(dir=install_dir))
    result = await toolset.remove_skill.model.call(name="my-skill")

    assert result == "Removed: my-skill"
    assert not skill_dir.exists()


@pytest.mark.asyncio
async def test_remove_skill_cleans_lock_file(tmp_path: Path) -> None:
    """remove_skill should remove entry from skills-lock.json."""
    install_dir = tmp_path / "skills"
    skill_dir = install_dir / "my-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\n")

    # Pre-populate lock file
    lock_path = install_dir / "skills-lock.json"
    lock_data = {"version": 1, "skills": {"my-skill": {"source": "x/y", "sourceType": "github", "computedHash": "aaa"}}}
    lock_path.write_text(json.dumps(lock_data))

    toolset = SkillSearchToolset(runtime=LocalRuntime(dir=install_dir))
    await toolset.remove_skill.model.call(name="my-skill")

    updated = json.loads(lock_path.read_text())
    assert "my-skill" not in updated["skills"]


@pytest.mark.asyncio
async def test_remove_skill_not_found(tmp_path: Path) -> None:
    install_dir = tmp_path / "skills"
    install_dir.mkdir()

    toolset = SkillSearchToolset(runtime=LocalRuntime(dir=install_dir))
    result = await toolset.remove_skill.model.call(name="nonexistent")

    assert "Cannot remove" in result


@pytest.mark.asyncio
async def test_remove_skill_path_traversal_blocked(tmp_path: Path) -> None:
    install_dir = tmp_path / "skills"
    install_dir.mkdir()
    outside = tmp_path / "secret"
    outside.mkdir()

    toolset = SkillSearchToolset(runtime=LocalRuntime(dir=install_dir))
    result = await toolset.remove_skill.model.call(name="../secret")

    assert "Cannot remove" in result
    assert outside.exists()


# ---------------------------------------------------------------------------
# SkillsLock
# ---------------------------------------------------------------------------


def test_lock_record_and_read(tmp_path: Path) -> None:
    lock = SkillsLock(tmp_path / "skills-lock.json")

    lock.record("my-skill", "owner/repo", "abc123")

    data = lock.read()
    assert data["version"] == 1
    assert data["skills"]["my-skill"]["computedHash"] == "abc123"
    assert data["skills"]["my-skill"]["source"] == "owner/repo"


def test_lock_remove(tmp_path: Path) -> None:
    lock = SkillsLock(tmp_path / "skills-lock.json")
    lock.record("skill-a", "a/b", "hash1")
    lock.record("skill-b", "c/d", "hash2")

    lock.remove("skill-a")

    data = lock.read()
    assert "skill-a" not in data["skills"]
    assert "skill-b" in data["skills"]


def test_lock_get_hash(tmp_path: Path) -> None:
    lock = SkillsLock(tmp_path / "skills-lock.json")
    lock.record("my-skill", "owner/repo", "abc123")

    assert lock.get_hash("my-skill") == "abc123"
    assert lock.get_hash("nonexistent") is None


def test_lock_read_nonexistent(tmp_path: Path) -> None:
    lock = SkillsLock(tmp_path / "no-such-file.json")

    data = lock.read()
    assert data == {"version": 1, "skills": {}}


# ---------------------------------------------------------------------------
# SkillSearchToolset — schema
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_toolset_exposes_six_tools(tmp_path: Path, context: ConversationContext) -> None:
    toolset = SkillSearchToolset(runtime=LocalRuntime(dir=tmp_path / "skills"))

    schemas = list(await toolset.schemas(context))

    assert len(schemas) == 6
    names = {s.function.name for s in schemas}  # type: ignore[union-attr]
    assert names == {"search_skills", "install_skill", "remove_skill", "list_skills", "load_skill", "run_skill_script"}


@pytest.mark.asyncio
async def test_toolset_individual_tools_accessible(tmp_path: Path, context: ConversationContext) -> None:
    toolset = SkillSearchToolset(runtime=LocalRuntime(dir=tmp_path / "skills"))

    for attr in ("search_skills", "install_skill", "remove_skill", "list_skills", "load_skill", "run_skill_script"):
        [schema] = await getattr(toolset, attr).schemas(context)
        assert schema.function.name == attr  # type: ignore[union-attr]
