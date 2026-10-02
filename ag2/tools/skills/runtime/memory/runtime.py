# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Sequence
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from ag2.context import ConversationContext
from ag2.exceptions import SkillError, SkillNotFoundError
from ag2.tools.final.function_tool import FunctionTool
from ag2.tools.skills.skill_types import Skill
from ag2.utils import CONTEXT_OPTION_NAME

from ..protocol import SkillRuntime
from .skill import MemorySkill

# Max characters returned from a single resource read before truncation.
_RESOURCE_READ_CAP = 100_000

_READ_ONLY_MSG = "MemoryRuntime is read-only; in-memory skills are defined in code, not installed"


class MemoryRuntime(SkillRuntime):
    """RAM-backed runtime owning code-defined :class:`MemorySkill` instances.

    Reads instructions and Resources straight from memory and runs Scripts as
    in-process callables. A skill with a callable body has it rendered on every
    read, so the instructions can reflect the live conversation. Read-only: it
    owns no installable storage, so ``install`` / ``remove`` / ``lock_dir`` raise.

    Holds one or more skills::

        MemoryRuntime(MemorySkill(name="a", ...), MemorySkill(name="b", ...))

    Skills are kept in declaration order; on an intra-runtime name clash the last
    wins, mirroring the toolkit's inter-runtime rule so that grouping skills into
    runtimes never changes behaviour.
    """

    __slots__ = ("_skills",)

    def __init__(self, *skills: MemorySkill) -> None:
        # dict preserves insertion order and resolves a name clash last-wins.
        self._skills: dict[str, MemorySkill] = {s.name: s for s in skills}

    @property
    def cleanup(self) -> bool:
        return False

    @property
    def lock_dir(self) -> Path:
        raise NotImplementedError(_READ_ONLY_MSG)

    @property
    def skills(self) -> list[Skill]:
        return [s.descriptor for s in self._skills.values()]

    async def read(self, name: str, context: "ConversationContext") -> str:
        skill = self._get(name)
        return _wrap_memory_content(skill, await _render_instructions(skill, context))

    async def read_resource(self, name: str, resource: str, context: "ConversationContext") -> str:
        skill = self._get(name)
        entry = skill.get_resource(resource)
        if entry is None:
            raise FileNotFoundError(f"resource {resource!r} not found in memory skill {name!r}")
        text = _to_text(await _run_tool(entry.tool, {}, context, skill=name, part=f"resource {resource!r}"))
        if len(text) > _RESOURCE_READ_CAP:
            return text[:_RESOURCE_READ_CAP] + "\n<!-- resource truncated -->"
        return text

    async def execute(
        self,
        name: str,
        script: str,
        context: "ConversationContext",
        args: dict[str, Any] | Sequence[str] | None = None,
    ) -> str:
        skill = self._get(name)
        entry = skill.get_script(script)
        if entry is None:
            raise FileNotFoundError(f"script {script!r} not found in memory skill {name!r}")
        if args is not None and not isinstance(args, dict):
            raise TypeError(
                f"in-process script {script!r} requires named arguments (an object); "
                "positional arguments (an array) are only supported for file-based scripts"
            )
        return _to_text(await _run_tool(entry.tool, args or {}, context, skill=name, part=f"script {script!r}"))

    def invalidate(self) -> None:
        # Nothing is cached — discovery reads the in-memory dict directly.
        return None

    def ensure_storage(self) -> None:
        # No storage backend to prepare.
        return None

    def install(self, source: Path, name: str) -> None:
        raise NotImplementedError(_READ_ONLY_MSG)

    def remove(self, name: str) -> None:
        raise NotImplementedError(_READ_ONLY_MSG)

    def _get(self, name: str) -> MemorySkill:
        skill = self._skills.get(name)
        if skill is None:
            raise SkillNotFoundError(f"Skill {name!r} not found in memory runtime")
        return skill


async def _run_tool(
    ft: FunctionTool,
    args: dict[str, Any],
    context: "ConversationContext",
    *,
    skill: str,
    part: str,
) -> Any:
    """Invoke a body/resource/script ``FunctionTool`` through its FastDepends model.

    Threads the live *context* as ``__ctx__`` so the callable's ``Context`` /
    ``Variable`` / ``Inject`` parameters resolve exactly as a regular tool's do
    (including the agent's dependency provider, carried on the context), and sync
    callables run in a worker thread. A ``SkillNotFoundError`` from the callable is
    re-raised as a plain ``SkillError`` naming *skill* and *part*: that type is the
    toolkit chain's "this runtime does not own the skill" signal, so leaving it
    would report a routing miss instead of the failure the callable hit.
    """
    call_kwargs: dict[str, Any] = {**args, CONTEXT_OPTION_NAME: context}
    try:
        async with AsyncExitStack() as stack:
            return await ft.model.asolve(
                **call_kwargs,
                stack=stack,
                cache_dependencies={},
                dependency_provider=context.dependency_provider,
            )
    except SkillNotFoundError as exc:
        raise SkillError(f"{part} of skill {skill!r} raised SkillNotFoundError: {exc}") from exc


def _to_text(value: Any) -> str:
    return value if isinstance(value, str) else str(value)


async def _render_instructions(skill: MemorySkill, context: "ConversationContext") -> str:
    """Return the skill body — static text, or the result of its callable.

    A callable body runs on every read through the same ``FunctionTool`` path as a
    resource, so it sees the live conversation and can use dependency injection.
    """
    body = skill.get_instructions()
    if isinstance(body, str):
        return body
    return _to_text(await _run_tool(body, {}, context, skill=skill.name, part="the body"))


def _wrap_memory_content(skill: MemorySkill, instructions: str) -> str:
    """Wrap a MemorySkill's instructions with its resource list and script schemas."""
    descriptor = skill.descriptor
    lines = [
        f'<skill_content name="{skill.name}">',
        instructions.strip(),
    ]
    if descriptor.resources:
        lines.append("<skill_resources>")
        lines.extend(f"  <file>{r.name}</file>" for r in descriptor.resources)
        lines.append("</skill_resources>")
    if descriptor.scripts:
        lines.append("<scripts>")
        lines.extend(_script_element(skill, s.name) for s in descriptor.scripts)
        lines.append("</scripts>")
    lines.append("</skill_content>")
    return "\n".join(lines)


def _script_element(skill: MemorySkill, name: str) -> str:
    entry = skill.get_script(name)
    assert entry is not None  # name comes from the descriptor, always present
    attrs = f'name="{entry.name}"'
    if entry.description:
        attrs += f' description="{entry.description}"'
    if entry.parameters_schema is not None:
        params = json.dumps(entry.parameters_schema)
        return f"  <script {attrs}>\n    <parameters_schema>{params}</parameters_schema>\n  </script>"
    return f"  <script {attrs}/>"
