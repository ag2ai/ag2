# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import AsyncExitStack, ExitStack
from dataclasses import dataclass, field

from ag2.annotations import Context
from ag2.events import BuiltinToolCallEvent, ToolCallEvent
from ag2.middleware import BaseMiddleware
from ag2.tools.schemas import ToolSchema
from ag2.tools.tool import Tool


@dataclass(slots=True)
class Skill:
    """A reference to a provider-managed skill.

    Args:
        id: Skill identifier. Provider-specific:

            - Anthropic first-party skills: ``"pptx"``, ``"xlsx"``, ``"docx"``, ``"pdf"``.
            - OpenAI system skills carry the ``openai-`` prefix
              (e.g. ``"openai-spreadsheets"``); bare ids like ``"spreadsheet"``
              are rejected with 404 "System skill not found".
            - Custom skill ids as returned by the provider's skills API
              (e.g. ``"skill_abc123"``).
        version: Version pin, always a string:

            - Anthropic: a version date such as ``"20251013"`` or ``"latest"``.
              ``None`` is sent as ``"latest"``.
            - OpenAI: a positive-integer string such as ``"2"`` or ``"latest"``.
              ``None`` omits the field, meaning the skill's ``default_version``.
    """

    id: str
    version: str | None = None


SKILLS_TOOL_NAME = "skills"


@dataclass(slots=True)
class SkillsToolSchema(ToolSchema):
    """Provider-neutral capability flag for provider-side skills.

    Skills are passed to the provider via a separate API parameter
    (``container`` for Anthropic, the hosted shell tool's
    ``environment.skills`` for OpenAI Responses).
    They never appear as standalone entries in the ``tools[]`` array.
    """

    type: str = field(default=SKILLS_TOOL_NAME, init=False)
    skills: list[Skill] = field(default_factory=list)


class SkillsTool(Tool):
    """Declares provider-side skills to be activated for this agent.

    Accepts skill identifiers as plain strings (shorthand for the provider's
    default version) or :class:`Skill` objects when a specific version is
    required. Skills are executed server-side by the provider — no local
    executor is registered.

    Provider support:

    - **Anthropic** — skills ride the ``container`` API parameter as
      ``{"type": "anthropic", "skill_id": ..., "version": ...}`` entries; they
      never appear in ``tools[]``. Skills run inside the code-execution
      container, so a :class:`~ag2.tools.CodeExecutionTool` is added
      automatically when missing. The client also injects the required
      ``code-execution-2025-08-25`` and ``skills-2025-10-02`` beta headers.

    - **OpenAI Responses API** — skills attach to the hosted shell tool as
      ``skill_reference`` entries in its ``container_auto`` environment; a
      ``{"type": "shell"}`` tool is appended automatically when missing.
      Combining with :class:`~ag2.tools.ContainerReferenceEnvironment` raises
      ``ValueError`` — skills for an existing container are configured at
      container creation (:class:`~ag2.config.ContainerManager`).

    - All other providers (including OpenAI Chat Completions) raise
      :class:`~ag2.exceptions.UnsupportedToolError`.

    Example::

        # Anthropic: first-party skills, auto-adds CodeExecutionTool
        agent = Agent(
            name="analyst",
            config=AnthropicConfig(model="claude-sonnet-4-6", ...),
            tools=[SkillsTool("pptx", Skill("xlsx", version="latest"))],
        )

        # OpenAI Responses: system skills use the "openai-" prefix,
        # auto-adds the hosted shell tool
        agent = Agent(
            name="analyst",
            config=OpenAIResponsesConfig(model="gpt-5.4", ...),
            tools=[SkillsTool("openai-spreadsheets")],
        )

    See:
    - https://platform.claude.com/docs/en/agents-and-tools/agent-skills
    - https://developers.openai.com/api/docs/guides/tools-skills
    """

    __slots__ = (
        "_skills",
        "name",
    )

    def __init__(self, *skills: str | Skill) -> None:
        self._skills: list[Skill] = [s if isinstance(s, Skill) else Skill(id=s) for s in skills]
        self.name = SKILLS_TOOL_NAME

    async def schemas(self, context: "Context") -> list[SkillsToolSchema]:
        return [SkillsToolSchema(skills=list(self._skills))]

    def register(
        self,
        stack: "ExitStack | AsyncExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        async def execute(event: "ToolCallEvent", context: "Context") -> None:
            pass

        stack.enter_context(
            context.stream.where(BuiltinToolCallEvent.name == SKILLS_TOOL_NAME).sub_scope(execute),
        )
