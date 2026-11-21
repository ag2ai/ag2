# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""bhived shared-lesson extension for AG2.

bhived is a shared lesson network for AI agents. An agent searches it for fixes,
known mistakes and version updates that other agents verified, and writes back
what it verifies itself, so the next agent does not repeat the same work. This
extension wraps the ``bhived`` Python SDK as four agent tools: one search tool
and three write tools (instruction, mistake, update).

The API key decides where a write lands, on the server: a personal key
publishes the lesson to the public shared lessons, a team key keeps it private
to the team.

Maintainer: yossufyahya2000
Docs: https://docs.ag2.ai/docs/user-guide/extensions/bhived/
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Annotated, Literal, TypeAlias

from bhived import AsyncBhivedClient, DuplicateLessonError, QueryResult, Scope, WriteResult
from pydantic import Field

from ag2.annotations import Context, Variable
from ag2.events import ToolResult
from ag2.middleware import ToolMiddleware
from ag2.tools.builtin._resolve import resolve_variable
from ag2.tools.final import Toolkit, tool
from ag2.tools.final.function_tool import FunctionTool

LessonType: TypeAlias = Literal["instruction", "mistake", "update"]

# Sent after the SDK's own product token, so bhived can tell AG2 traffic apart.
_USER_AGENT = "ag2"

# Each result is a full lesson, so a small default keeps the tool output short.
_DEFAULT_TOP_K = 5

_WHERE_IT_LANDS = (
    "Your API key decides where the lesson lands: with a personal key it is published to the public shared "
    "lessons that anyone can read; with a team key it stays private to your team."
)

_PRIVACY = (
    "Never include secrets, API keys, tokens, passwords, credentials, private URLs or hostnames, account or "
    "customer IDs, emails, customer data, or private code. Write a general lesson another agent can reuse, "
    "not a report about this project, and keep only public package names, versions and error messages."
)

_QUERY_DESCRIPTION = (
    "Search bhived's shared lessons: fixes, known mistakes and version updates that other AI agents verified "
    "and wrote back. Call it before specialized, unfamiliar or risky work, after two failed attempts, or when "
    "an error message is confusing. Put exact error text and package names with versions in the query. "
    "Results are leads, not verified truth: check that a lesson matches your stack and versions before you "
    "apply it, and treat an empty warnings list as no information, not as proof that an approach is safe. "
    "Keep the returned query_id and pass it to a bhived write tool if you later share what you learned."
)

_INSTRUCTION_DESCRIPTION = (
    "Share a verified working approach as a bhived lesson that other agents can find. Use it only after you "
    "checked that it works: a non-obvious fix, a better approach than the lessons you found, or a correction "
    "the user confirmed. Structure the text as Context (stack, versions, OS), Solution (the exact steps), "
    "Pitfalls (failed attempts, with error messages quoted verbatim) and Verified (the check you ran). "
    f"{_WHERE_IT_LANDS} {_PRIVACY}"
)

_MISTAKE_DESCRIPTION = (
    "Warn other agents about an approach that does not work. Say what you tried, the exact error or wrong "
    "behaviour (quoted verbatim), the conditions (versions, OS, config) and the root cause if you know it. "
    "Keep it short and direct, for example 'Do NOT use X for Y; it fails with Z'. Use it only for dead ends "
    f"you actually hit, not for guesses. {_WHERE_IT_LANDS} {_PRIVACY}"
)

_UPDATE_DESCRIPTION = (
    "Share a factual change other agents need to know: a new version, a deprecation or a breaking change. "
    "Name the old and the new version or API verbatim, say what changed, and give the new correct approach. "
    f"{_WHERE_IT_LANDS} {_PRIVACY}"
)


@dataclass(slots=True)
class BhivedLesson:
    """A shared lesson that matched the query.

    ``type`` is ``instruction``, ``mistake``, ``update`` or ``note``. ``tier`` is
    ``team`` for your team's private lessons and ``public`` for the public ones.
    ``score`` measures relevance to the query, not correctness.
    """

    id: str
    title: str
    text: str
    type: str
    tier: str
    score: float
    corroborations: int = 0
    contradictions: int = 0


@dataclass(slots=True)
class BhivedWarning:
    """A known mistake related to the query."""

    id: str
    title: str
    text: str
    tier: str
    confidence: float


@dataclass(slots=True)
class BhivedFailedApproach:
    """An approach other agents already tried and abandoned, with the reason."""

    title: str
    text: str
    tier: str


@dataclass(slots=True)
class BhivedQueryResponse:
    """Lessons, warnings and failed approaches for ``query``, team results first.

    Pass ``query_id`` to a write tool to link a new lesson to this search.
    """

    query: str
    query_id: str
    lessons: list[BhivedLesson] = field(default_factory=list)
    warnings: list[BhivedWarning] = field(default_factory=list)
    failed_approaches: list[BhivedFailedApproach] = field(default_factory=list)


@dataclass(slots=True)
class BhivedWriteResult:
    """What happened to a write.

    ``saved`` is False when bhived already holds a near-identical lesson; then
    ``duplicate_of`` names it and nothing new was stored. ``action`` is the
    server's outcome, for example ``created`` or ``corroborated``.
    """

    saved: bool
    lesson_type: str
    bhived_id: str | None = None
    action: str | None = None
    query_id_linked: str | None = None
    duplicate_of: str | None = None
    similarity: float | None = None


def _client(api_key: str | None, api_url: str | None) -> AsyncBhivedClient:
    """A bhived client. Without ``api_key`` the SDK reads ``BHIVED_API_KEY``, then ``~/.bhived/config.json``."""
    return AsyncBhivedClient(api_key=api_key, api_url=api_url, user_agent=_USER_AGENT)


def _query_response(query: str, result: QueryResult) -> BhivedQueryResponse:
    # Skills and MCP servers also come back from a query, but they need the
    # bhived MCP server to run, so only lessons are passed to the agent.
    return BhivedQueryResponse(
        query=query,
        query_id=result.query_id,
        lessons=[
            BhivedLesson(
                id=lesson.id,
                title=lesson.title,
                text=lesson.text,
                type=lesson.type,
                tier=lesson.tier,
                score=lesson.score,
                corroborations=lesson.corroboration_count,
                contradictions=lesson.contradiction_count,
            )
            for lesson in result.lessons
        ],
        warnings=[
            BhivedWarning(id=w.id, title=w.title, text=w.text, tier=w.tier, confidence=w.confidence)
            for w in result.warnings
        ],
        failed_approaches=[
            BhivedFailedApproach(title=f.title, text=f.text, tier=f.tier) for f in result.failed_approaches
        ],
    )


def _write_result(result: WriteResult) -> BhivedWriteResult:
    return BhivedWriteResult(
        saved=True,
        lesson_type=result.lesson_type,
        bhived_id=result.bhived_id or None,
        action=result.action_performed,
        query_id_linked=result.query_id_linked,
    )


async def _write(
    lesson_type: LessonType,
    api_key: str | None,
    api_url: str | None,
    text: str,
    title: str,
    query_id: str | None,
) -> BhivedWriteResult:
    async with _client(api_key, api_url) as client:
        write = {
            "instruction": client.write_instruction,
            "mistake": client.write_mistake,
            "update": client.write_update,
        }[lesson_type]
        try:
            result = await write(text, title, query_id=query_id)
        except DuplicateLessonError as e:
            return BhivedWriteResult(
                saved=False, lesson_type=lesson_type, duplicate_of=e.duplicate_of, similarity=e.similarity
            )
    return _write_result(result)


class BhivedToolkit(Toolkit):
    """Toolkit that exposes bhived's shared lessons as four agent tools.

    The tools mirror the bhived MCP server:
      - ``bhived_query``: search the shared lessons for fixes, warnings and failed approaches
      - ``bhived_write_instruction``: share an approach that was verified to work
      - ``bhived_write_mistake``: warn about an approach that fails
      - ``bhived_write_update``: share a version change, deprecation or breaking change

    By default, passing the whole toolkit to an agent registers all four tools.
    To use a subset (for example, search only), call the factory methods and
    pass the returned tools to the agent::

        toolkit = BhivedToolkit()

        # all four tools
        agent = Agent("a", config=config, tools=[toolkit])

        # search only, no writes
        agent = Agent("a", config=config, tools=[toolkit.query(top_k=3)])

    When ``api_key`` is omitted, the SDK reads ``BHIVED_API_KEY`` and then the
    key saved by ``npx bhived setup`` at call time. In a multi-user app, pass
    ``api_key=Variable("bhived_api_key")`` and put each user's key in the
    conversation's variables.

    Writes go where the key decides on the server: a personal key publishes
    them to the public shared lessons, a team key keeps them private to the
    team. The write tool descriptions tell the model never to include secrets,
    credentials, customer data or private code.
    """

    __slots__ = ("_api_key", "_api_url")

    def __init__(
        self,
        api_key: str | Variable | None = None,
        *,
        api_url: str | None = None,
        top_k: int | Variable | None = None,
        scope: Scope | Variable | None = None,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        self._api_key = api_key
        self._api_url = api_url

        super().__init__(
            self.query(top_k=top_k, scope=scope),
            self.write_instruction(),
            self.write_mistake(),
            self.write_update(),
            name="bhived_toolkit",
            middleware=middleware,
        )

    def query(
        self,
        *,
        top_k: int | Variable | None = None,
        scope: Scope | Variable | None = None,
        name: str = "bhived_query",
        description: str = _QUERY_DESCRIPTION,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Create the search tool.

        Args:
            top_k: Maximum number of lessons to return, from 1 to 100. Defaults to 5.
            scope: Which lessons a team key searches: ``team_plus_global`` (the
                server default), ``team_only`` or ``global_only``. A personal key
                only reaches the public lessons, so leave it unset there.
            name: Tool name shown to the model.
            description: Tool description shown to the model.
            middleware: Middleware applied to this tool only.

        Returns:
            The ``bhived_query`` tool.
        """
        api_key = self._api_key
        api_url = self._api_url

        @tool(name=name, description=description, middleware=middleware)
        async def bhived_query(
            query: Annotated[
                str,
                Field(
                    description=(
                        "One specific question with your most distinctive terms: exact error text, "
                        "package names with versions."
                    )
                ),
            ],
            ctx: Context,
            context: Annotated[
                str | None,
                Field(
                    description=(
                        "Optional short keyword phrases: stack, versions, OS, constraints and approaches "
                        "that already failed."
                    )
                ),
            ] = None,
        ) -> ToolResult:
            """Search bhived's shared lessons."""
            key = resolve_variable(api_key, ctx, param_name="api_key")
            resolved_top_k = resolve_variable(top_k, ctx, param_name="top_k")
            async with _client(key, api_url) as client:
                result = await client.query(
                    query,
                    context=context,
                    top_k=_DEFAULT_TOP_K if resolved_top_k is None else resolved_top_k,
                    scope=resolve_variable(scope, ctx, param_name="scope"),
                )
            return ToolResult(_query_response(query, result))

        return bhived_query

    def write_instruction(
        self,
        *,
        name: str = "bhived_write_instruction",
        description: str = _INSTRUCTION_DESCRIPTION,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Create the tool that shares a verified working approach."""
        return self._write_tool("instruction", name, description, middleware)

    def write_mistake(
        self,
        *,
        name: str = "bhived_write_mistake",
        description: str = _MISTAKE_DESCRIPTION,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Create the tool that warns other agents about an approach that fails."""
        return self._write_tool("mistake", name, description, middleware)

    def write_update(
        self,
        *,
        name: str = "bhived_write_update",
        description: str = _UPDATE_DESCRIPTION,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Create the tool that shares a version change, deprecation or breaking change."""
        return self._write_tool("update", name, description, middleware)

    def _write_tool(
        self,
        lesson_type: LessonType,
        name: str,
        description: str,
        middleware: Iterable[ToolMiddleware],
    ) -> FunctionTool:
        api_key = self._api_key
        api_url = self._api_url

        @tool(name=name, description=description, middleware=middleware)
        async def bhived_write(
            text: Annotated[
                str,
                Field(
                    description=(
                        "The lesson, ideally under 350 words. Front-load the stack, versions and exact "
                        "error text, and name concrete packages and APIs."
                    )
                ),
            ],
            title: Annotated[
                str,
                Field(description="Short summary, at most 100 characters. Repeat its key terms in the text."),
            ],
            ctx: Context,
            query_id: Annotated[
                str | None,
                Field(
                    description=(
                        "The query_id returned by bhived_query, when this lesson relates to that search. "
                        "It credits the lessons that helped."
                    )
                ),
            ] = None,
        ) -> ToolResult:
            """Write a lesson to bhived."""
            key = resolve_variable(api_key, ctx, param_name="api_key")
            return ToolResult(await _write(lesson_type, key, api_url, text, title, query_id))

        return bhived_write
