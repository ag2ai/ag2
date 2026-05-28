# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import shutil
import tarfile
import tempfile
from pathlib import Path

from autogen.beta.exceptions import SkillInstallError
from autogen.beta.tools.skills.local_skills.loader import SkillLoader, parse_frontmatter
from autogen.beta.tools.skills.skill_types import SkillMetadata

_EXCLUDE_NAMES = frozenset({".git", ".env", "__pycache__", ".DS_Store", "node_modules"})
_MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB


def _find_target_prefix(
    tar: tarfile.TarFile,
    members: list[tarfile.TarInfo],
    root_dir: str,
    skill_id: str,
) -> str:
    """Resolve the archive prefix that contains SKILL.md for *skill_id*.

    Resolution order:
    1. ``{root}/skills/{skill_id}/SKILL.md``  — classic monorepo with ``skills/`` subdir
    2. ``{root}/{skill_id}/SKILL.md``          — monorepo without ``skills/`` subdir
    3. Scan every ``SKILL.md`` in the archive and match by ``name`` frontmatter —
       handles registries where the ``skillId`` differs from the directory name
    4. ``{root}/``                              — standalone / single-skill repo
    """
    member_names = {m.name for m in members}

    # 1. Classic monorepo: {root}/skills/{skill_id}/SKILL.md
    if skill_id and f"{root_dir}/skills/{skill_id}/SKILL.md" in member_names:
        return f"{root_dir}/skills/{skill_id}/"

    # 2. Direct subdir monorepo: {root}/{skill_id}/SKILL.md
    if skill_id and f"{root_dir}/{skill_id}/SKILL.md" in member_names:
        return f"{root_dir}/{skill_id}/"

    # 3. Fuzzy: read every SKILL.md in the archive and match by name frontmatter
    if skill_id:
        for member in members:
            if not (member.isfile() and member.name.endswith("/SKILL.md")):
                continue
            fobj = tar.extractfile(member)
            if fobj is None:
                continue
            try:
                fm = parse_frontmatter(fobj.read().decode("utf-8", errors="replace"))
                if fm.get("name") == skill_id:
                    return member.name[: -len("SKILL.md")]
            except Exception:
                continue

    # 4. Standalone repo fallback
    return f"{root_dir}/"


def extract_skill(tar_path: Path, skill_id: str, dest: Path) -> SkillMetadata:
    """Extract a skill from *tar_path* into ``dest/<skill_name>/``.

    Supports multiple monorepo layouts (with or without a ``skills/`` subdir),
    fuzzy matching by SKILL.md ``name`` frontmatter, and standalone repos.
    Returns validated :class:`~autogen.beta.tools.local_skills.loader.SkillMetadata`
    for the extracted skill.

    Raises:
        SkillInstallError: If the archive has no valid ``SKILL.md`` or contains
            a file exceeding the 25 MB size limit.
    """
    with tempfile.TemporaryDirectory() as extract_tmp:
        skill_content_dir = Path(extract_tmp) / "skill"
        skill_content_dir.mkdir()
        skill_fm: dict[str, object] | None = None

        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()
            root_dir = next((m.name.split("/")[0] for m in members if m.name.split("/")[0]), "")
            target_prefix = _find_target_prefix(tar, members, root_dir, skill_id)

            for member in members:
                if not member.name.startswith(target_prefix):
                    continue
                rel_path = member.name[len(target_prefix) :]
                if not rel_path:
                    continue

                parts = Path(rel_path).parts
                if any(p in _EXCLUDE_NAMES for p in parts) or any(p == ".." for p in parts):
                    continue
                if member.issym() or member.islnk():
                    continue
                if member.isfile() and member.size > _MAX_FILE_BYTES:
                    raise SkillInstallError(f"File too large (>25MB): {rel_path}")

                target_path = skill_content_dir / rel_path
                if member.isdir():
                    target_path.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    file_obj = tar.extractfile(member)
                    if file_obj is not None:
                        target_path.write_bytes(file_obj.read())
                    if rel_path == "SKILL.md" and skill_fm is None:
                        skill_fm = parse_frontmatter(target_path.read_text(encoding="utf-8"))

        if skill_fm is None or "name" not in skill_fm:
            raise SkillInstallError(f"No SKILL.md found in archive (skill_id={skill_id!r})")

        skill_name = str(skill_fm["name"])
        fm_str = {k: str(v) for k, v in skill_fm.items()}

        final_dest = dest / skill_name
        if final_dest.exists():
            shutil.rmtree(final_dest)
        shutil.copytree(skill_content_dir, final_dest)

        meta = SkillMetadata(
            name=skill_name,
            description=fm_str.get("description") or "",
            path=final_dest,
            has_scripts=(final_dest / "scripts").is_dir(),
            version=fm_str.get("version") or None,
            license=fm_str.get("license") or None,
            compatibility=fm_str.get("compatibility") or None,
        )
        SkillLoader.validate_skill_metadata(final_dest, skill_fm, meta)
        return meta


def format_install_result(meta: SkillMetadata, install_dir: Path) -> str:
    """Build a human-readable install summary from skill metadata."""
    lines = [f"Installed: {meta.name} \u2192 {install_dir / meta.name}/"]
    lines.append(f"Description: {meta.description}")
    if meta.version:
        lines.append(f"Version: {meta.version}")
    if meta.has_scripts:
        script_names = sorted(p.name for p in (meta.path / "scripts").iterdir() if p.is_file())
        if script_names:
            lines.append(f"Scripts: {', '.join(script_names)}")
    lines.append(f'Use load_skill("{meta.name}") to read full instructions.')
    return "\n".join(lines)
