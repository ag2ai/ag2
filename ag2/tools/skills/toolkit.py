# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Iterable, Sequence
from typing import Annotated, Any, Literal

from pydantic import Field

from ag2.annotations import Context
from ag2.context import ConversationContext
from ag2.exceptions import SkillNotFoundError
from ag2.middleware import ToolMiddleware
from ag2.tools.final import Toolkit, tool
from ag2.tools.final.function_tool import FunctionTool
from ag2.tools.skills.runtime import LocalRuntime, MemoryRuntime, MemorySkill, SkillRuntime
from ag2.tools.skills.skill_types import Skill


class SkillsToolkit(Toolkit):
    """Client-side toolkit for ``agentskills.io``-style local skills.

    Packs the progressive-disclosure pattern as a single toolkit so any
    provider can ship local skills without extra wiring:

    1. ``list_skills()`` — lightweight catalog (name + description + location).
    2. ``load_skill(name)`` — full ``SKILL.md`` instructions on demand.
    3. ``read_skill_resource(name, resource)`` — read a bundled resource file
       (the ones listed in ``<skill_resources>``) on demand.
    4. ``run_skill_script(name, script, args)`` — execute a script from the
       skill's ``scripts/`` directory.

    Accepts **one or more** runtimes; a path is wrapped in a
    :class:`LocalRuntime`. Each tool routes by **last-to-first chain
    delegation**: the runtimes are tried from last to first, falling through to
    the next only on :class:`~ag2.exceptions.SkillNotFoundError`, so the
    last runtime shadows earlier ones on a name clash (project overrides global).

    Default runtime scans ``./.agents/skills``::

        SkillsToolkit()

    Custom install directory::

        SkillsToolkit(LocalRuntime("./skills"))

    Global + project, each with its own config::

        SkillsToolkit(LocalRuntime("~/.agents/skills"), LocalRuntime(".agents/skills"))

    Pick individual tools instead of the full toolkit::

        skills = SkillsToolkit()
        agent = Agent(
            "a",
            config=config,
            tools=[skills.list_skills(), skills.load_skill()],
        )
    """

    __slots__ = ("_runtimes",)

    def __init__(
        self,
        *runtimes: SkillRuntime | str | os.PathLike[str] | MemorySkill,
        name: str = "skills_toolkit",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        self._runtimes: tuple[SkillRuntime, ...] = (
            tuple(_ensure_runtime(r) for r in runtimes) if runtimes else (LocalRuntime(),)
        )

        super().__init__(
            self.list_skills(),
            self.load_skill(),
            self.read_skill_resource(),
            self.run_skill_script(),
            name=name,
            middleware=middleware,
        )

    @property
    def runtimes(self) -> tuple[SkillRuntime, ...]:
        """The runtimes this toolkit composes, in declaration order."""
        return self._runtimes

    def merged_skills(self) -> list[Skill]:
        """Skills across all runtimes, deduped by name (last runtime wins)."""
        merged: dict[str, Skill] = {}
        for runtime in self._runtimes:
            for skill in runtime.skills:
                merged[skill.name] = skill
        return sorted(merged.values(), key=lambda s: s.name)

    def discover_skills(self) -> list[dict[str, str]]:
        return [
            {
                "name": s.name,
                "description": s.metadata.description,
                "location": s.location or "",
            }
            for s in self.merged_skills()
        ]

    def _name_annotation(self, description: str) -> object:
        """Build the ``name`` parameter annotation.

        Constrains ``name`` to a :class:`~typing.Literal` enum of the skills
        discovered at schema-build time so the model cannot pass an unknown
        skill name. Falls back to ``str`` when no skills are present (a
        ``Literal`` cannot be empty).
        """
        names = [s.name for s in self.merged_skills()]
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
            return _route_read(self._runtimes, name)

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
        async def _read_skill_resource(
            name: name_type,  # type: ignore[valid-type]
            resource: Annotated[
                str,
                Field(description="Resource path relative to the skill directory, for example references/guide.md."),
            ],
            ctx: Context,  # injected; absent from the tool schema
        ) -> str:
            return await _route_read_resource(self._runtimes, name, resource, ctx)

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
            ctx: Context,
            script: Annotated[
                str,
                Field(description="Script name, for example scaffold.py, build.sh, or an in-process script."),
            ],
            args: Annotated[
                dict[str, Any] | list[str] | None,
                Field(
                    description=(
                        "Script arguments. Use an object of named arguments for in-process scripts "
                        '(for example {"value": 10, "factor": 2}), matching the script\'s '
                        "<parameters_schema> shown in the loaded skill. Use an array of strings for "
                        'file-based scripts\' positional CLI arguments (for example ["input.docx"]).'
                    )
                ),
            ] = None,
        ) -> str:
            return await _route_execute(self._runtimes, name, script, ctx, args)

        return _run_skill_script


def _route_read(runtimes: Sequence[SkillRuntime], name: str) -> str:
    for runtime in reversed(runtimes):
        try:
            return runtime.read(name)
        except SkillNotFoundError:
            continue
    raise SkillNotFoundError(f"Skill {name!r} not found in any runtime")


async def _route_read_resource(
    runtimes: Sequence[SkillRuntime],
    name: str,
    resource: str,
    context: ConversationContext,
) -> str:
    for runtime in reversed(runtimes):
        try:
            return await runtime.read_resource(name, resource, context)
        except SkillNotFoundError:
            continue
    raise SkillNotFoundError(f"Skill {name!r} not found in any runtime")


async def _route_execute(
    runtimes: Sequence[SkillRuntime],
    name: str,
    script: str,
    context: ConversationContext,
    args: dict[str, Any] | Sequence[str] | None,
) -> str:
    for runtime in reversed(runtimes):
        try:
            return await runtime.execute(name, script, context, args)
        except SkillNotFoundError:
            continue
    raise SkillNotFoundError(f"Skill {name!r} not found in any runtime")


def _ensure_runtime(runtime: SkillRuntime | str | os.PathLike[str] | MemorySkill) -> SkillRuntime:
    """Coerce a toolkit argument into a runtime.

    A loose :class:`MemorySkill` is wrapped in its own single-skill
    :class:`MemoryRuntime` at its declared position, so the existing
    last-to-first chain governs precedence with no special-casing.
    """
    if isinstance(runtime, MemorySkill):
        return MemoryRuntime(runtime)
    return LocalRuntime.ensure_runtime(runtime)
