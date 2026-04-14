# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

import yaml

from autogen.beta.exceptions import InvalidSkillError, InvalidSkillNameError, SkillNotFoundError
from autogen.beta.tools.toolkits.skills.skill_types import SkillMetadata


def parse_frontmatter(text: str) -> dict[str, object]:
    """Parse YAML frontmatter (``--- ... ---``) from a SKILL.md file.

    Returns a dict of parsed key-value pairs using :func:`yaml.safe_load`.
    Returns an empty dict when there is no valid frontmatter block.
    """
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    parsed = yaml.safe_load(text[3:end].strip())
    return {str(k): v for k, v in parsed.items()} if isinstance(parsed, dict) else {}


class SkillLoader:
    """Discovers and loads skills from the filesystem.

    Follows the `agentskills.io <https://agentskills.io>`_ progressive-disclosure
    convention: each skill lives in its own directory that contains a ``SKILL.md``
    file with a YAML frontmatter header.

    Frontmatter parsing and strict validation rules are aligned with:
    https://agentskills.io/specification

    Search priority (first match wins for duplicate names):

    1. ``{cwd}/.agents/skills/``  — project-level, cross-client
    2. ``~/.agents/skills/``      — user-level, cross-client
    3. Any *paths* supplied to the constructor (appended in order)
    """

    DEFAULT_PATHS: list[Path] = [
        Path(".agents/skills"),
        Path.home() / ".agents/skills",
    ]

    _SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

    def __init__(self, *paths: str | Path, strict: bool = True) -> None:
        if paths:
            self._paths = [Path(p) for p in paths]
        else:
            self._paths = list(self.DEFAULT_PATHS)
        self._strict = strict
        self._cache: dict[str, SkillMetadata] | None = None

    def invalidate(self) -> None:
        """Clear the cached skill metadata.

        The next call to :meth:`discover` will rescan the filesystem.
        """
        self._cache = None

    def discover(self) -> list[SkillMetadata]:
        """Scan all configured paths and return metadata for every skill found.

        Results are cached after the first scan.  Call :meth:`invalidate` to
        force a rescan (e.g. after installing or removing a skill).

        When the same skill name appears in more than one path, the first
        occurrence (higher-priority path) wins.
        """
        if self._cache is not None:
            return sorted(self._cache.values(), key=lambda m: m.name)

        seen: dict[str, SkillMetadata] = {}
        for base in self._paths:
            if not base.exists():
                continue
            for skill_dir in sorted(base.iterdir()):
                if not skill_dir.is_dir():
                    continue
                skill_md = skill_dir / "SKILL.md"
                if not skill_md.exists():
                    continue
                text = skill_md.read_text(encoding="utf-8")
                fm_raw = {k: v for k, v in parse_frontmatter(text).items() if v is not None}
                fm = {k: str(v) for k, v in fm_raw.items()}
                meta = SkillMetadata(
                    name=fm.get("name") or skill_dir.name,
                    description=fm.get("description") or "",
                    path=skill_dir,
                    has_scripts=(skill_dir / "scripts").is_dir(),
                    version=fm.get("version") or None,
                    license=fm.get("license") or None,
                    compatibility=fm.get("compatibility") or None,
                )
                if self._strict:
                    self.validate_skill_metadata(skill_dir, fm_raw, meta)
                if meta.name not in seen:
                    seen[meta.name] = meta
        self._cache = seen
        return sorted(seen.values(), key=lambda m: m.name)

    def load(self, name: str) -> str:
        """Return the full text of a skill's ``SKILL.md`` by skill name.

        Raises:
            KeyError: if no skill with that name is found.
        """
        skill_dir = self._find_dir(name)
        return (skill_dir / "SKILL.md").read_text(encoding="utf-8")

    def get_path(self, name: str) -> Path:
        """Return the directory path of a skill by name.

        Raises:
            KeyError: if no skill with that name is found.
        """
        return self._find_dir(name)

    def _find_dir(self, name: str) -> Path:
        if not name.strip():
            raise InvalidSkillNameError("skill name must not be empty")
        if "/" in name or "\\" in name:
            raise InvalidSkillNameError("skill name must not contain path separators")
        for meta in self.discover():
            if meta.name == name:
                return meta.path
        raise SkillNotFoundError(f"Skill {name!r} not found in any configured path")

    @classmethod
    def validate_skill_metadata(cls, skill_dir: Path, fm: dict[str, object], meta: SkillMetadata) -> None:
        if "name" not in fm:
            raise InvalidSkillError(f"Skill {skill_dir.name!r} is missing required frontmatter field: name")
        if "description" not in fm:
            raise InvalidSkillError(f"Skill {skill_dir.name!r} is missing required frontmatter field: description")

        name = meta.name
        description = meta.description
        compatibility = meta.compatibility

        if not (1 <= len(name) <= 64):
            raise InvalidSkillError(f"Invalid skill name {name!r}: expected length 1-64")
        if not cls._SKILL_NAME_RE.fullmatch(name):
            raise InvalidSkillError(f"Invalid skill name {name!r}: expected lowercase alnum and single hyphens")
        if name != skill_dir.name:
            raise InvalidSkillError(f"Skill name {name!r} must match directory name {skill_dir.name!r}")

        if not (1 <= len(description) <= 1024):
            raise InvalidSkillError(f"Invalid description for {name!r}: expected length 1-1024")

        if compatibility is not None and not (1 <= len(compatibility) <= 500):
            raise InvalidSkillError(f"Invalid compatibility for {name!r}: expected length 1-500")

        metadata = fm.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise InvalidSkillError(f"Invalid metadata for {name!r}: expected mapping")

        allowed_tools = fm.get("allowed-tools")
        if allowed_tools is not None and not isinstance(allowed_tools, str):
            raise InvalidSkillError(f"Invalid allowed-tools for {name!r}: expected string")
