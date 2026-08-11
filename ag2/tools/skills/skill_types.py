# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SkillMetadata:
    """Metadata parsed from a skill's SKILL.md frontmatter.

    Pure frontmatter: it carries only what the ``SKILL.md`` header declares.
    Folder facts (where the skill lives, which scripts/resources it bundles)
    belong to :class:`Skill`, not here.
    """

    name: str
    description: str
    version: str | None = None
    license: str | None = None
    compatibility: str | None = None


@dataclass(slots=True)
class Script:
    """A runnable unit of a skill — a descriptor only; running it is the
    runtime's job (``runtime.execute``).

    Two forms, distinguished by ``parameters_schema``:

    - **File-backed** (``LocalRuntime``): ``name`` is the path relative to
      ``scripts/`` (e.g. ``build.sh``) and ``parameters_schema`` is ``None`` —
      the script takes positional string arguments.
    - **In-process** (``MemoryRuntime``): ``name`` is the script identifier and
      ``parameters_schema`` is the JSON Schema for its named arguments, generated
      from the underlying callable's signature and disclosed inside the loaded
      skill content.
    """

    name: str
    parameters_schema: dict[str, Any] | None = field(default=None)


@dataclass(slots=True)
class Resource:
    """A bundled file in a skill directory that is not ``SKILL.md`` and not
    under ``scripts/``.

    A descriptor only — reading it is the runtime's job
    (``runtime.read_resource``). ``name`` is the path relative to the skill
    directory (e.g. ``references/guide.md``).
    """

    name: str


@dataclass(slots=True)
class Skill:
    """A discovered skill: its frontmatter plus the scripts and resources it
    bundles.

    A pure descriptor — it performs no IO. Reading and executing a skill is the
    owning :class:`~ag2.tools.skills.runtime.SkillRuntime`'s job; a
    skill carries no runtime reference. ``location`` is an optional display
    pointer (the ``SKILL.md`` path for filesystem runtimes; ``None`` for
    in-memory ones).
    """

    metadata: SkillMetadata
    scripts: tuple[Script, ...] = ()
    resources: tuple[Resource, ...] = ()
    resources_truncated: bool = False
    location: str | None = None

    @property
    def name(self) -> str:
        return self.metadata.name
