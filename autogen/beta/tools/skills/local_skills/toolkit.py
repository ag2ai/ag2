# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
import shlex
import stat
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from autogen.beta.middleware import ToolMiddleware
from autogen.beta.tools.final import Toolkit, tool
from autogen.beta.tools.final.function_tool import FunctionTool
from autogen.beta.tools.skills.local_skills.loader import strip_frontmatter
from autogen.beta.tools.skills.runtime import LocalRuntime, SkillRuntime

_RESOURCE_CAP = 50
_RESOURCE_READ_CAP = 100_000


class SkillsToolkit(Toolkit):
    """Client-side toolkit for ``agentskills.io``-style local skills.

    Packs the progressive-disclosure pattern as a single toolkit so any
    provider can ship local skills without extra wiring:

    1. ``list_skills()`` — lightweight catalog (name + description).
    2. ``load_skill(name)`` — full ``SKILL.md`` instructions on demand.
    3. ``read_skill_resource(name, resource)`` — read a bundled resource
       file (the ones listed in ``<skill_resources>``) on demand.
    4. ``run_skill_script(name, script, args)`` — execute a script from
       the skill's ``scripts/`` directory.

    Default runtime scans ``./.agents/skills`` and ``~/.agents/skills``::

        SkillsToolkit()

    Custom install directory::

        SkillsToolkit(runtime=LocalRuntime("./skills"))

    Extra read-only search paths::

        SkillsToolkit(runtime=LocalRuntime("./skills", extra_paths=["./shared-skills"]))

    Pick individual tools instead of the full toolkit::

        skills = SkillsToolkit()
        agent = Agent(
            "a",
            config=config,
            tools=[skills.list_skills(), skills.load_skill()],
        )
    """

    __slots__ = ("_runtime",)

    def __init__(
        self,
        runtime: SkillRuntime | str | os.PathLike[str] | None = None,
        *,
        name: str = "local_skills_toolkit",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        if runtime is not None:
            self._runtime: SkillRuntime = LocalRuntime.ensure_runtime(runtime)
        else:
            self._runtime = LocalRuntime()

        super().__init__(
            self.list_skills(),
            self.load_skill(),
            self.read_skill_resource(),
            self.run_skill_script(),
            name=name,
            middleware=middleware,
        )

    @property
    def runtime(self) -> SkillRuntime:
        """The underlying ``SkillRuntime`` used to discover and load skills."""
        return self._runtime

    def discover_skills(self) -> list[dict[str, str]]:
        return [
            {
                "name": m.name,
                "description": m.description,
                "location": str(m.path / "SKILL.md"),
            }
            for m in self._runtime.discover()
        ]

    def _name_annotation(self, description: str) -> object:
        """Build the ``name`` parameter annotation.

        Constrains ``name`` to a :class:`~typing.Literal` enum of the skills
        discovered at construction time so the model cannot pass an unknown
        skill name. Falls back to ``str`` when no skills are present (a
        ``Literal`` cannot be empty).
        """
        names = [s["name"] for s in self.discover_skills()]
        base: object = Literal[tuple(names)] if names else str  # type: ignore[valid-type]
        return Annotated[base, Field(description=description)]

    def list_skills(
        self,
        *,
        name: str = "list_skills",
        description: str = "List available local skills with name and short description.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        @tool(name=name, description=description, middleware=middleware)
        def _list_skills() -> list[dict[str, str]]:
            return self.discover_skills()

        return _list_skills

    def load_skill(
        self,
        *,
        name: str = "load_skill",
        description: str = "Load the full SKILL.md content for a specific skill.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        name_type = self._name_annotation("Skill name returned by list_skills.")

        @tool(name=name, description=description, middleware=middleware)
        def _load_skill(name: name_type) -> str:  # type: ignore[valid-type]
            # Body only: strip the YAML frontmatter (name/description already
            # surfaced via the catalog) before wrapping, per the agentskills.io
            # client guide.
            body = strip_frontmatter(self._runtime.load(name))
            skill_dir = self._runtime.get_path(name)
            return _wrap_skill_content(name, body, skill_dir, _list_resources(skill_dir))

        return _load_skill

    def read_skill_resource(
        self,
        *,
        name: str = "read_skill_resource",
        description: str = (
            "Read a bundled resource file from a skill's directory, given a path "
            "relative to the skill (as listed in <skill_resources>)."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        name_type = self._name_annotation("Skill name returned by list_skills.")

        @tool(name=name, description=description, middleware=middleware)
        def _read_skill_resource(
            name: name_type,  # type: ignore[valid-type]
            resource: Annotated[
                str,
                Field(description="Resource path relative to the skill directory, for example references/guide.md."),
            ],
        ) -> str:
            skill_dir = self._runtime.get_path(name)
            resolved = (skill_dir / resource).resolve()
            # Reject path traversal: the target must stay inside the skill dir.
            if not resolved.is_file() or not resolved.is_relative_to(skill_dir.resolve()):
                raise FileNotFoundError(f"resource {resource!r} not found in {skill_dir}")
            text = resolved.read_text(encoding="utf-8", errors="replace")
            if len(text) > _RESOURCE_READ_CAP:
                return text[:_RESOURCE_READ_CAP] + "\n<!-- resource truncated -->"
            return text

        return _read_skill_resource

    def run_skill_script(
        self,
        *,
        name: str = "run_skill_script",
        description: str = "Run a script from a skill's scripts directory. Only .py and .sh scripts are supported.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        name_type = self._name_annotation("Skill name returned by list_skills.")

        @tool(name=name, description=description, middleware=middleware)
        async def _run_skill_script(
            name: name_type,  # type: ignore[valid-type]
            script: Annotated[
                str,
                Field(description="Script filename inside scripts/, for example scaffold.py or build.sh."),
            ],
            args: Annotated[
                list[str] | None,
                Field(description="Optional script arguments passed as positional parameters."),
            ] = None,
        ) -> str:
            skill_dir = self._runtime.get_path(name)
            scripts_dir = skill_dir / "scripts"
            resolved_script = (scripts_dir / script).resolve()
            if not resolved_script.is_file() or not resolved_script.is_relative_to(scripts_dir.resolve()):
                raise FileNotFoundError(f"script {script!r} not found in {scripts_dir}")

            first_line = resolved_script.read_text(encoding="utf-8", errors="replace").split("\n", 1)[0]
            has_shebang = first_line.startswith("#!")

            if has_shebang:
                resolved_script.chmod(resolved_script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                command = [f"./{resolved_script.name}"]
            elif resolved_script.suffix.lower() == ".py":
                command = ["python3", f"./{resolved_script.name}"]
            elif resolved_script.suffix.lower() == ".sh":
                command = ["sh", f"./{resolved_script.name}"]
            else:
                resolved_script.chmod(resolved_script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                command = [f"./{resolved_script.name}"]

            if args:
                command.extend(args)

            # async + await env.run(...) so the command runs in the agent's own
            # event loop. A sync path would drive remote backends via a throwaway
            # asyncio.run() per call (a fresh loop each time), breaking clients
            # bound to the first loop (e.g. Daytona's httpx keep-alive pool).
            env = self._runtime.shell(scripts_dir)
            return await env.run(shlex.join(command))

        return _run_skill_script


def _list_resources(skill_dir: Path, *, cap: int = _RESOURCE_CAP) -> tuple[list[str], bool]:
    """List bundled resource files (relative paths), excluding ``SKILL.md``.

    Returns ``(paths, truncated)``. Does not read file contents — the model
    loads them on demand once it sees the skill instructions.
    """
    if not skill_dir.is_dir():
        return [], False
    rels: list[str] = []
    truncated = False
    for p in sorted(skill_dir.rglob("*")):
        if not p.is_file() or p.name == "SKILL.md":
            continue
        if len(rels) >= cap:
            truncated = True
            break
        rels.append(p.relative_to(skill_dir).as_posix())
    return rels, truncated


def _wrap_skill_content(name: str, body: str, skill_dir: Path, resources: tuple[list[str], bool]) -> str:
    """Wrap a SKILL.md body with identifying tags, base path, and a resource list."""
    files, truncated = resources
    lines = [
        f'<skill_content name="{name}">',
        body.strip(),
        "",
        f"Skill directory: {skill_dir}",
        "Relative paths in this skill are relative to the skill directory.",
    ]
    if files:
        lines.append("<skill_resources>")
        lines.extend(f"  <file>{f}</file>" for f in files)
        if truncated:
            lines.append("  <!-- resource list truncated -->")
        lines.append("</skill_resources>")
    lines.append("</skill_content>")
    return "\n".join(lines)
