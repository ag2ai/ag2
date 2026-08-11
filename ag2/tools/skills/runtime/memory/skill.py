# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar, overload

from typing_extensions import ParamSpec

from ag2.tools.final import tool
from ag2.tools.final.function_tool import FunctionTool
from ag2.tools.skills.skill_types import Resource, Script, Skill, SkillMetadata

P = ParamSpec("P")
T = TypeVar("T")


class MemorySkill:
    """A Skill defined inline in code rather than discovered on disk.

    It carries its instructions, Resources, and Scripts as in-memory values.
    Resources and Scripts are registered with decorators::

        skill = MemorySkill(name="unit-converter", description="Convert units")


        @skill.resource
        def conversion_table() -> str:
            return "miles->km: 1.60934"


        @skill.script(description="Multiply value by factor")
        def convert(value: float, factor: float) -> str:
            return str(value * factor)

    A Resource callable runs every read (so it may return live data). A Script
    callable runs in-process when invoked via ``run_skill_script``; its JSON
    Schema is generated from the signature and disclosed inside the loaded skill
    content. Owned by a :class:`MemoryRuntime`; pass one (or several) straight to
    ``SkillPlugin`` / ``SkillsToolkit`` and it is wrapped automatically.
    """

    __slots__ = ("name", "description", "instructions", "version", "_resources", "_scripts")

    def __init__(
        self,
        *,
        name: str,
        description: str,
        instructions: str = "",
        version: str | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.instructions = instructions
        self.version = version
        self._resources: dict[str, _MemoryResource] = {}
        self._scripts: dict[str, _MemoryScript] = {}

    @overload
    def resource(self, func: Callable[P, T]) -> Callable[P, T]: ...

    @overload
    def resource(
        self, func: None = None, *, name: str | None = None, description: str | None = None
    ) -> Callable[[Callable[P, T]], Callable[P, T]]: ...

    def resource(
        self,
        func: Callable[P, T] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
        """Register a callable as a Resource of this skill.

        Usable bare (``@skill.resource``) or with arguments
        (``@skill.resource(name=..., description=...)``). The callable's name and
        docstring supply the defaults. Returns the original function unchanged.
        """

        def register(f: Callable[P, T]) -> Callable[P, T]:
            rname = name or f.__name__
            desc = description or f.__doc__ or ""
            self._resources[rname] = _MemoryResource(rname, desc, tool(f, name=rname, description=desc))
            return f

        return register(func) if func is not None else register

    @overload
    def script(self, func: Callable[P, T]) -> Callable[P, T]: ...

    @overload
    def script(
        self,
        func: None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[[Callable[P, T]], Callable[P, T]]: ...

    def script(
        self,
        func: Callable[P, T] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
        """Register a callable as an in-process Script of this skill.

        Usable bare (``@skill.script``) or with arguments. The parameter JSON
        Schema is generated from the callable's signature. Returns the original
        function unchanged.
        """

        def register(f: Callable[P, T]) -> Callable[P, T]:
            sname = name or f.__name__
            desc = description or f.__doc__ or ""
            self._scripts[sname] = _MemoryScript(sname, desc, tool(f, name=sname, description=desc))
            return f

        return register(func) if func is not None else register

    @property
    def descriptor(self) -> Skill:
        """The inert :class:`Skill` descriptor for catalog and discovery."""
        return Skill(
            metadata=SkillMetadata(name=self.name, description=self.description, version=self.version),
            scripts=tuple(
                Script(name=s.name, parameters_schema=s.parameters_schema)
                for s in sorted(self._scripts.values(), key=lambda s: s.name)
            ),
            resources=tuple(Resource(name=r.name) for r in sorted(self._resources.values(), key=lambda r: r.name)),
            location=None,
        )

    def get_resource(self, resource: str) -> "_MemoryResource | None":
        return self._resources.get(resource)

    def get_script(self, script: str) -> "_MemoryScript | None":
        return self._scripts.get(script)


@dataclass(slots=True)
class _MemoryResource:
    """A code-defined resource: a :class:`FunctionTool` wrapping the callable.

    Wrapping with ``tool()`` gives the resource the same FastDepends invocation
    path as a script, so a resource callable can use ``Context`` / ``Variable`` /
    ``Inject`` dependency injection when read.
    """

    name: str
    description: str
    tool: FunctionTool


@dataclass(slots=True)
class _MemoryScript:
    """A code-defined script: a :class:`FunctionTool` wrapping the callable.

    Wrapping with ``tool()`` gives one object that supplies both the parameter
    JSON Schema *and* the FastDepends-validated invocation path (``tool.model``),
    so a script is serialized and validated exactly like a regular tool.
    """

    name: str
    description: str
    tool: FunctionTool

    @property
    def parameters_schema(self) -> dict[str, Any] | None:
        params = self.tool.schema.function.parameters
        return params if params.get("properties") else None
