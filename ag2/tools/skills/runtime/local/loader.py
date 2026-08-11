# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
from pathlib import Path

import yaml

from ag2.exceptions import InvalidSkillError, InvalidSkillNameError, SkillNotFoundError
from ag2.tools.skills.skill_types import Resource, Script, Skill, SkillMetadata

logger = logging.getLogger(__name__)

_UNQUOTED_COLON_RE = re.compile(r"^(\s*[A-Za-z0-9_-]+:\s+)(.*)$")

# Cap on the number of bundled resources listed per skill. Discovery stops
# scanning once this many are found and flags the skill as truncated.
_RESOURCE_CAP = 50


def parse_frontmatter(text: str) -> dict[str, object]:
    """Parse YAML frontmatter (``--- ... ---``) from a SKILL.md file.

    Returns a dict of parsed key-value pairs using :func:`yaml.safe_load`.
    Returns an empty dict when there is no valid frontmatter block.

    Skill files authored for other clients sometimes contain technically
    invalid YAML that those clients' parsers happen to accept — most commonly an
    unquoted scalar whose value contains a colon (``description: Use when: ...``).
    A best-effort recovery quotes such values and retries once before giving up.
    """
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    block = text[3:end].strip()
    try:
        parsed = yaml.safe_load(block)
    except yaml.YAMLError:
        parsed = _recover_malformed_yaml(block)
    return {str(k): v for k, v in parsed.items()} if isinstance(parsed, dict) else {}


def _recover_malformed_yaml(block: str) -> object:
    """Quote unquoted ``key: value`` scalars containing a colon and retry once.

    Returns ``None`` when the block is still unparsable after the fix.
    """
    fixed_lines: list[str] = []
    for line in block.splitlines():
        m = _UNQUOTED_COLON_RE.match(line)
        if m and ":" in m.group(2):
            value = m.group(2).strip()
            if value and value[0] not in "\"'>|[{":
                escaped = value.replace('"', '\\"')
                line = f'{m.group(1)}"{escaped}"'
        fixed_lines.append(line)
    try:
        return yaml.safe_load("\n".join(fixed_lines))
    except yaml.YAMLError:
        return None


def strip_frontmatter(text: str) -> str:
    """Return the markdown body after the YAML frontmatter block, trimmed.

    Returns the whole text (trimmed) when there is no ``--- ... ---`` block.
    """
    if not text.startswith("---"):
        return text.strip()
    end = text.find("\n---", 3)
    if end == -1:
        return text.strip()
    # Skip past the closing '---' line to the start of the body.
    after = text[end + 1 :]
    newline = after.find("\n")
    body = after[newline + 1 :] if newline != -1 else ""
    return body.strip()


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

    def __init__(
        self,
        *paths: str | os.PathLike[str],
        strict: bool = True,
    ) -> None:
        self._paths = [Path(p) for p in (paths or self.DEFAULT_PATHS)]
        self._strict = strict
        # name -> (descriptor, skill directory). The directory is kept here, not
        # on the Skill, so the descriptor stays a pure (path-free) value object.
        self._cache: dict[str, tuple[Skill, Path]] | None = None

    def invalidate(self) -> None:
        """Clear the cached skill metadata.

        The next call to :meth:`discover` will rescan the filesystem.
        """
        self._cache = None

    def discover(self) -> list[Skill]:
        """Scan all configured paths and return a :class:`Skill` for each found.

        Results are cached after the first scan.  Call :meth:`invalidate` to
        force a rescan (e.g. after installing or removing a skill).

        When the same skill name appears in more than one path, the first
        occurrence (higher-priority path) wins; the shadowed skill is skipped
        with a warning.

        In ``strict`` mode a malformed skill raises (the install path relies on
        this). In lenient mode the malformed skill is skipped with a warning and
        the remaining skills still load — a single bad skill never aborts the
        whole scan.
        """
        if self._cache is None:
            self._cache = self._scan()
        return [skill for skill, _ in sorted(self._cache.values(), key=lambda e: e[0].name)]

    def _scan(self) -> dict[str, tuple[Skill, Path]]:
        seen: dict[str, tuple[Skill, Path]] = {}
        for base in self._paths:
            if not base.exists():
                continue
            for skill_dir in sorted(base.iterdir()):
                if not skill_dir.is_dir():
                    continue
                skill_md = skill_dir / "SKILL.md"
                if not skill_md.exists():
                    continue
                try:
                    skill = self._load_skill(skill_dir, skill_md)
                except (InvalidSkillError, OSError, yaml.YAMLError) as exc:
                    if self._strict:
                        raise
                    logger.warning("Skipping skill %r: %s", skill_dir.name, exc)
                    continue
                if skill.name in seen:
                    logger.warning(
                        "Skill name %r in %s is shadowed by a higher-priority skill; ignoring",
                        skill.name,
                        skill_dir,
                    )
                    continue
                seen[skill.name] = (skill, skill_dir)
        return seen

    def _load_skill(self, skill_dir: Path, skill_md: Path) -> Skill:
        """Build and validate the :class:`Skill` for one directory.

        Strict mode raises :class:`InvalidSkillError` on any spec violation (the
        install path relies on this). Lenient mode follows the agentskills.io
        client guide: cosmetic name issues (mismatched directory, length,
        charset) warn but still load, while a missing/empty description is the
        only metadata reason to skip. :meth:`discover` routes the raised error
        (re-raise in strict mode, skip-with-warning in lenient mode).
        """
        text = skill_md.read_text(encoding="utf-8")
        fm_raw = {k: v for k, v in parse_frontmatter(text).items() if v is not None}
        fm = {k: str(v) for k, v in fm_raw.items()}
        meta = SkillMetadata(
            name=fm.get("name") or skill_dir.name,
            description=fm.get("description") or "",
            version=fm.get("version") or None,
            license=fm.get("license") or None,
            compatibility=fm.get("compatibility") or None,
        )
        if self._strict:
            self.validate_skill_metadata(skill_dir, fm_raw, meta)
        else:
            self._validate_lenient(skill_dir, meta)
        resources, truncated = _list_resources(skill_dir)
        return Skill(
            metadata=meta,
            scripts=_list_scripts(skill_dir),
            resources=resources,
            resources_truncated=truncated,
            location=str(skill_md),
        )

    def _validate_lenient(self, skill_dir: Path, meta: SkillMetadata) -> None:
        """Warn on cosmetic issues, raise only when the skill is unusable.

        See https://agentskills.io/client-implementation/adding-skills-support:
        name mismatch / over-length / unexpected charset relax to warnings to
        keep skills authored for other clients usable; a missing or empty
        description is essential for disclosure, so it raises (→ skip).
        """
        if not meta.description:
            raise InvalidSkillError(f"Skill {skill_dir.name!r} has a missing or empty description")
        name = meta.name
        if name != skill_dir.name:
            logger.warning("Skill name %r does not match directory %r; loading anyway", name, skill_dir.name)
        if len(name) > 64:
            logger.warning("Skill name %r exceeds 64 characters; loading anyway", name)
        if not self._SKILL_NAME_RE.fullmatch(name):
            logger.warning(
                "Skill name %r is not lowercase alnum with single hyphens; loading anyway",
                name,
            )

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

    def get_skill(self, name: str) -> Skill:
        """Return the :class:`Skill` descriptor by name.

        Raises:
            SkillNotFoundError: if no skill with that name is found.
        """
        if self._cache is None:
            self._cache = self._scan()
        entry = self._cache.get(name)
        if entry is None:
            raise SkillNotFoundError(f"Skill {name!r} not found in any configured path")
        return entry[0]

    def _find_dir(self, name: str) -> Path:
        if not name.strip():
            raise InvalidSkillNameError("skill name must not be empty")
        if "/" in name or "\\" in name:
            raise InvalidSkillNameError("skill name must not contain path separators")
        if self._cache is None:
            self._cache = self._scan()
        entry = self._cache.get(name)
        if entry is None:
            raise SkillNotFoundError(f"Skill {name!r} not found in any configured path")
        return entry[1]

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


def _list_scripts(skill_dir: Path) -> tuple[Script, ...]:
    """List executable files under ``scripts/`` as :class:`Script` descriptors.

    Names are paths relative to ``scripts/`` (posix). Returns an empty tuple
    when the skill has no ``scripts/`` directory.
    """
    scripts_dir = skill_dir / "scripts"
    if not scripts_dir.is_dir():
        return ()
    return tuple(
        Script(name=p.relative_to(scripts_dir).as_posix()) for p in sorted(scripts_dir.rglob("*")) if p.is_file()
    )


def _list_resources(skill_dir: Path, *, cap: int = _RESOURCE_CAP) -> tuple[tuple[Resource, ...], bool]:
    """List bundled resources as :class:`Resource` descriptors.

    A resource is any file under *skill_dir* that is not ``SKILL.md`` and not
    under ``scripts/``. Names are paths relative to the skill directory (posix).
    Returns ``(resources, truncated)``; scanning stops once *cap* resources are
    found and flags ``truncated``.
    """
    scripts_dir = skill_dir / "scripts"
    out: list[Resource] = []
    truncated = False
    for p in sorted(skill_dir.rglob("*")):
        if not p.is_file() or p.name == "SKILL.md" or p.is_relative_to(scripts_dir):
            continue
        if len(out) >= cap:
            truncated = True
            break
        out.append(Resource(name=p.relative_to(skill_dir).as_posix()))
    return tuple(out), truncated
