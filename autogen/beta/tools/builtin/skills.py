# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass, field

from autogen.beta.annotations import Context
from autogen.beta.middleware import BaseMiddleware
from autogen.beta.tools.schemas import ToolSchema
from autogen.beta.tools.tool import Tool


@dataclass(slots=True)
class Skill:
    """A reference to a provider-managed skill.

    Args:
        id: Skill identifier (such as ``"pptx"``, ``"xlsx"``, ``"skill_abc123"``).
        version: Version pin. ``None`` means the provider uses its latest version.
    """

    id: str
    version: str | None = None


@dataclass(slots=True)
class SkillsToolSchema(ToolSchema):
    """Provider-neutral capability flag for provider-side skills.

    Skills are passed to the provider via a separate API parameter
    (``container`` for Anthropic).
    They never appear in the ``tools[]`` array of a request.
    """

    type: str = field(default="skills", init=False)
    skills: list[Skill] = field(default_factory=list)


class SkillsTool(Tool):
    """Declares provider-side skills to be activated for this agent.

    Accepts skill identifiers as plain strings (shorthand for the latest
    version) or :class:`Skill` objects when a specific version is required.

    Example::

        # Strings — shorthand for Skill(id=s, version=None)
        SkillsTool("pptx", "xlsx")

        # Mix
        SkillsTool("pptx", Skill("xlsx", version="latest"))
    """

    __slots__ = ("_skills",)

    def __init__(self, *skills: str | Skill) -> None:
        self._skills: list[Skill] = [s if isinstance(s, Skill) else Skill(id=s) for s in skills]

    async def schemas(self, context: "Context") -> list[SkillsToolSchema]:
        return [SkillsToolSchema(skills=list(self._skills))]

    def register(
        self,
        stack: "ExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        pass
