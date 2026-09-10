# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Shared scaffolding for the MCP suite.

Everything here is incidental to what a test asserts: the agent that answers
``"hi"``, the client-side callback that makes a capability declared, the two
lines of ceremony a modern-era round trip costs. What a test is *about* stays in
the test.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from mcp.client.session import ClientRequestContext
from mcp.server.streamable_http import CONTENT_TYPE_JSON, CONTENT_TYPE_SSE
from mcp.types import (
    CallToolResult,
    ElicitRequestParams,
    ElicitResult,
    InputRequiredResult,
    TextContent,
)
from mcp_types.version import LATEST_HANDSHAKE_VERSION
from pydantic import BaseModel
from typing_extensions import Self

from ag2 import Agent, Context
from ag2.config.client import LLMClient
from ag2.config.config import ModelConfig
from ag2.events import (
    BaseEvent,
    ModelMessage,
    ModelMessageChunk,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
)
from ag2.hitl import HumanHook
from ag2.mcp.elicitation import ANSWER_FIELD, input_request
from ag2.mcp.pause import SuspendedTurn
from ag2.mcp.sessions import CONVERSATION_META_KEY
from ag2.testing import TestConfig

ElicitationCallback = Callable[[ClientRequestContext, ElicitRequestParams], Awaitable[ElicitResult]]


class Weather(BaseModel):
    """A response schema, for the tests that need the served agent to have one."""

    city: str
    temp_c: float


def greeter(reply: str = "hi", *, name: str = "greeter", **agent_kwargs: Any) -> Agent:
    """An agent that answers ``reply`` and does nothing else."""
    return Agent(name, config=TestConfig(reply), **agent_kwargs)


@dataclass
class Asked:
    """What an :func:`asking_agent`'s tool did.

    ``answers`` is what ``context.input()`` returned, ``outcomes`` how each ask
    ended — ``"answered"``, or the name of the exception that ended it — and
    ``runs`` how many times the tool body was entered, which is how "the run
    resumed rather than restarted" is asserted: a restart enters it twice.
    """

    answers: list[str] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    runs: int = 0


@dataclass
class Gate:
    """A hold placed inside the tool, just before it asks.

    ``entered`` is set when the tool reaches the hold, which gives a
    deterministic point at which a round is mid-flight rather than parked on a
    question; nothing continues until ``release`` is set.
    """

    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)


def asking_agent(
    asked: "Asked | None" = None,
    *,
    questions: Sequence[str] = ("What colour?",),
    timeout: float | None = None,
    gate: "Gate | None" = None,
    hitl_hook: HumanHook | None = None,
) -> Agent:
    """An agent whose one tool asks the human ``questions``, in order.

    What the tool did is recorded into ``asked``, which is the assertion that
    matters: the turn completing proves only that *something* answered.
    """

    async def ask_human(ctx: Context) -> str:
        if asked is not None:
            asked.runs += 1
        collected: list[str] = []
        try:
            if gate is not None:
                gate.entered.set()
                await gate.release.wait()
            for question in questions:
                answer = await ctx.input(question, timeout=timeout)
                collected.append(answer)
                if asked is not None:
                    asked.answers.append(answer)
                    asked.outcomes.append("answered")
        except BaseException as exc:
            if asked is not None:
                asked.outcomes.append(type(exc).__name__)
            raise
        return "human said: " + ", ".join(collected)

    return Agent(
        "asker",
        config=TestConfig(ToolCallEvent(name="ask_human"), "done", raise_tool_errors=False),
        tools=[ask_human],
        hitl_hook=hitl_hook,
    )


def accepting(answer: str) -> ElicitResult:
    """The wire answer a client sends when its human replies ``answer``."""
    return ElicitResult(action="accept", content={ANSWER_FIELD: answer})


def answering(answer: str, *, seen: list[str] | None = None) -> ElicitationCallback:
    """A client that replies ``answer``, recording into ``seen`` what it was asked."""

    async def callback(context: ClientRequestContext, params: ElicitRequestParams) -> ElicitResult:
        if seen is not None:
            seen.append(params.message)
        return accepting(answer)

    return callback


def refusing(action: str = "decline", *, seen: list[str] | None = None) -> ElicitationCallback:
    """A client whose human refuses — declining the question, or dismissing it."""

    async def callback(context: ClientRequestContext, params: ElicitRequestParams) -> ElicitResult:
        if seen is not None:
            seen.append(params.message)
        return ElicitResult(action=action)  # type: ignore[arg-type]

    return callback


async def declares_elicitation(context: ClientRequestContext, params: ElicitRequestParams) -> ElicitResult:
    """A callback supplied only so the client declares that it can answer.

    A server asks nobody who has not said so, which a modern-era test driving
    the retry loop by hand still needs — and never invokes, because that era
    carries the question back as the call's result.
    """
    raise AssertionError("a modern-era question is answered by retrying, not through the callback")


async def ask(session: Any, message: str = "go", *, conversation: str | None = None, **kwargs: Any) -> Any:
    """Call the conversational tool once, ready for the modern era's paused result.

    ``allow_input_required=True`` keeps a pause a *result* rather than a raised
    error; a handshake-era server never returns one, so both eras come here.
    """
    arguments: dict[str, Any] = {"message": message}
    if conversation is not None:
        arguments["conversation"] = conversation
    return await session.call_tool("ask", arguments, allow_input_required=True, **kwargs)


def outstanding(paused: InputRequiredResult) -> tuple[str, Any]:
    """The one request a paused round came back with, and the key naming it."""
    ((key, request),) = (paused.input_requests or {}).items()
    return key, request


async def answer(
    session: Any,
    paused: InputRequiredResult,
    response: Any,
    *,
    message: str = "go",
    conversation: str | None = None,
) -> Any:
    """Retry ``paused``'s call, answering the question it came back with.

    The boundary binds state to its call's arguments, so the retry repeats the
    ``message`` (and ``conversation``) the paused call was made with.
    """
    key, _request = outstanding(paused)
    return await ask(
        session,
        message,
        conversation=conversation,
        input_responses={key: response},
        request_state=paused.request_state,
    )


def first_text(result: Any) -> str:
    """The text of a result's first content block."""
    block = result.content[0]
    assert isinstance(block, TextContent), f"expected a text block, got {block.type!r}"
    return block.text


def handle_of(result: Any) -> str:
    """The conversation handle a result carries, as a programmatic client reads it."""
    assert result.meta is not None
    return str(result.meta[CONVERSATION_META_KEY])


async def asks_twice(turn: SuspendedTurn) -> CallToolResult:
    """Stand in for a held turn with a second question behind the first."""
    await turn.ask(input_request("First?"))
    await turn.ask(input_request("Second?"))
    return _finished()


async def asks_then_works(turn: SuspendedTurn, gate: asyncio.Event) -> CallToolResult:
    """Stand in for a held turn that, once answered, is busy rather than parked."""
    await turn.ask(input_request("First?"))
    await gate.wait()
    return _finished()


async def parks_until_cancelled(closed: list[str]) -> Any:
    """Stand in for a held turn: parks forever, and records that it was closed."""
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        closed.append("closed")
        raise


def _finished() -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text="done")])


async def settle() -> None:
    """Give the loop enough turns for a held run to reach its next await."""
    for _ in range(10):
        await asyncio.sleep(0)


class Clock:
    """A monotonic clock a test advances by hand."""

    __slots__ = ("_now",)

    def __init__(self, now: float = 1000.0) -> None:
        self._now = now

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


JSON_HEADERS = {"Accept": f"{CONTENT_TYPE_JSON}, {CONTENT_TYPE_SSE}", "Content-Type": CONTENT_TYPE_JSON}
"""What a client sends on a POST to the streamable-HTTP endpoint."""


def initialize_request(*, request_id: int = 1, version: str = LATEST_HANDSHAKE_VERSION) -> dict[str, Any]:
    """The ``initialize`` handshake a handshake-era client opens a session with."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "initialize",
        "params": {
            "protocolVersion": version,
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1"},
        },
    }


class ChunkConfig(ModelConfig):
    """Test config whose client streams ``ModelMessageChunk`` events before the final reply.

    Used to exercise the executor's progress / log forwarding. The final body
    defaults to the concatenation of the chunks.
    """

    def __init__(self, *chunks: str, final: str | None = None) -> None:
        self._chunks = chunks
        self._final = final if final is not None else "".join(chunks)

    def copy(self) -> Self:
        return self

    def create(self) -> "ChunkClient":
        return ChunkClient(self._chunks, self._final)

    def create_files_client(self) -> None:
        raise NotImplementedError


class ChunkClient(LLMClient):
    def __init__(self, chunks: Sequence[str], final: str) -> None:
        self._chunks = chunks
        self._final = final

    async def __call__(self, messages: Sequence[BaseEvent], context: Context, **kwargs: Any) -> ModelResponse:
        for chunk in self._chunks:
            await context.send(ModelMessageChunk(chunk))
        message = ModelMessage(self._final)
        await context.send(message)
        return ModelResponse(message=message)


class RecordingConfig(ModelConfig):
    """Records the whole message list the framework sends the LLM on each turn.

    ``TrackingConfig`` keeps only each turn's last message, but a conversation is
    exactly what accumulates *before* it — so continuity tests read the full list
    through :attr:`prompts`.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.calls: list[list[BaseEvent]] = []

    @property
    def prompts(self) -> list[list[str]]:
        """The text inputs replayed on each turn: one list per turn, in order."""
        return [
            [
                part.content
                for m in call
                if isinstance(m, ModelRequest)
                for part in m.parts
                if isinstance(part, TextInput)
            ]
            for call in self.calls
        ]

    def copy(self) -> Self:
        return self

    def create(self) -> "RecordingClient":
        return RecordingClient(self.config.create(), self.calls)

    def create_files_client(self) -> None:
        raise NotImplementedError


class RecordingClient(LLMClient):
    def __init__(self, client: LLMClient, sink: list[list[BaseEvent]]) -> None:
        self.client = client
        self.sink = sink

    async def __call__(self, messages: Sequence[BaseEvent], context: Context, **kwargs: Any) -> ModelResponse:
        self.sink.append(list(messages))
        return await self.client(messages, context=context, **kwargs)
