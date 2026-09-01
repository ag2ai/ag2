# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``MCPToolkit`` against a live HTTP MCP server, through its real session construction.

The rest of ``test/tools/test_mcp.py`` substitutes ``_mcp_session`` for a fake,
which is right for tool proxying and content mapping but leaves the transport
itself — the client, the handshake, the wire — untouched. This module is the one
place that exercises it, so a change to how the streamable-HTTP session is built
cannot ship unexercised.

Lives apart from ``test_mcp.py`` because serving on a real socket needs
``uvicorn``, which ships with ``ag2[acp]`` and not ``ag2[mcp]``; the skip guard
below would otherwise take the fake-session tests down with it.
"""

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest

pytest.importorskip("mcp")
pytest.importorskip("uvicorn")

import uvicorn

from ag2 import Agent, Context
from ag2.events import (
    HumanInputRequest,
    HumanMessage,
    TextInput,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResultEvent,
)
from ag2.mcp import MCPServer, mcp_tool
from ag2.stream import MemoryStream
from ag2.testing import TestConfig
from ag2.tools import AnswerPolicy, MCPServerConfig, MCPToolkit
from ag2.tools.toolkits.mcp_server import toolkit as _toolkit_module


@mcp_tool
def echo(message: str) -> str:
    """Echo the message back."""
    return f"echo: {message}"


@asynccontextmanager
async def _live_mcp_server(
    headers_seen: list[dict[str, str]] | None = None,
) -> AsyncGenerator[str]:
    """Serve an AG2 ``MCPServer`` on a loopback port, yielding the MCP endpoint URL.

    The URL carries the canonical trailing slash. ``test_a_slashless_url_still_reaches
    _the_server`` strips it to cover the redirect a Starlette ``Mount`` issues.

    The served side is AG2's own public serving API rather than a hand-built
    ``mcp`` server, so this test owns no handler registration of its own and
    stays valid across changes to how handlers are registered.

    When ``headers_seen`` is supplied, every request's headers are appended to it,
    which is how a test observes what the toolkit's HTTP client actually sent.
    """
    served = MCPServer(Agent("live", config=TestConfig("unused")), tools=[echo], path="/mcp")
    app = served if headers_seen is None else _recording(served, headers_seen)
    async with _serving(app) as url:
        yield url


@asynccontextmanager
async def _serving(app: Any) -> AsyncGenerator[str]:
    """Run ``app`` under ``uvicorn`` on a loopback port, yielding the MCP endpoint URL."""
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    # Bound here rather than inside `serve()`: the socket is already listening, so
    # the port is known and a connection can be made without waiting for start-up.
    sock = config.bind_socket()
    uv = uvicorn.Server(config)
    serving = asyncio.create_task(uv.serve(sockets=[sock]))
    try:
        yield f"http://127.0.0.1:{sock.getsockname()[1]}/mcp/"
    finally:
        uv.should_exit = True
        await serving
        sock.close()


def _recording(app: Any, headers_seen: list[dict[str, str]]) -> Any:
    """Wrap an ASGI app, recording each HTTP request's headers."""

    async def recording(scope: dict[str, Any], receive: Callable[..., Any], send: Callable[..., Any]) -> None:
        if scope["type"] == "http":
            headers_seen.append({k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]})
        await app(scope, receive, send)

    return recording


@pytest.mark.asyncio
async def test_tools_are_discovered_over_the_real_transport(context: Context) -> None:
    async with _live_mcp_server() as url:
        schemas = list(await MCPToolkit(url).schemas(context))

    # ``ask`` is the served agent's own conversational tool; ``echo`` is the
    # custom tool. Both arriving means the handshake and ``tools/list`` completed.
    assert sorted(s.function.name for s in schemas) == ["ask", "echo"]


@pytest.mark.asyncio
async def test_a_tool_call_round_trips_over_the_real_transport(context: Context) -> None:
    async with _live_mcp_server() as url:
        toolkit = MCPToolkit(url)
        await toolkit.schemas(context)
        proxy = next(t for t in toolkit.tools if t.name == "echo")

        result = await proxy(ToolCallEvent(name="echo", arguments='{"message": "hi"}'), context)

    assert isinstance(result, ToolResultEvent)
    assert result.result.parts == [TextInput(content="echo: hi")]


@pytest.mark.asyncio
async def test_configured_headers_reach_the_server(context: Context) -> None:
    """A bearer-token MCP server is reached this way, so no request may skip them."""
    headers_seen: list[dict[str, str]] = []

    async with _live_mcp_server(headers_seen) as url:
        toolkit = MCPToolkit(MCPServerConfig(server_url=url, headers={"X-Tenant": "acme"}, authorization_token="t0ken"))
        await toolkit.schemas(context)

    assert headers_seen, "no HTTP request reached the server"
    assert all(h.get("x-tenant") == "acme" for h in headers_seen)
    assert all(h.get("authorization") == "Bearer t0ken" for h in headers_seen)


@pytest.mark.asyncio
async def test_a_slashless_url_still_reaches_the_server(context: Context) -> None:
    """A Starlette-mounted endpoint 307s the slashless form, and that form is what
    a caller naturally writes, so the toolkit's client has to follow the redirect.
    """
    async with _live_mcp_server() as url:
        schemas = list(await MCPToolkit(url.rstrip("/")).schemas(context))

    assert sorted(s.function.name for s in schemas) == ["ask", "echo"]


# --------------------------------------------------------------------------- #
# An AG2 agent asks, and another AG2 agent answers.
#
# Everything else tests one side against a double. This is the only place the
# two halves meet, so it is the only place a wire-level disagreement between
# them shows up — a state token round-tripped wrongly, an answer keyed to the
# wrong question, a capability advertised in one shape and read in another all
# survive both suites and fail only here.
# --------------------------------------------------------------------------- #


def _asking_agent(runs: list[str], answers: list[str] | None = None) -> Agent:
    """A served agent whose one tool asks its caller a question.

    ``runs`` records once per tool invocation: on the modern era the call is
    answered across two ``tools/call`` round trips, and a served run that
    *restarted* rather than resumed would run the body — and its LLM call —
    a second time. ``answers`` records what the tool was told, which is the
    calling agent's human's reply having crossed the wire.
    """

    async def ask_colour(ctx: Context) -> str:
        runs.append("ran")
        answer = await ctx.input("What colour?")
        if answers is not None:
            answers.append(answer)
        return "human said: " + answer

    return Agent(
        "asker",
        config=TestConfig(ToolCallEvent(name="ask_colour"), "done", raise_tool_errors=False),
        tools=[ask_colour],
    )


class _Human:
    """The calling agent's own human-input channel, answering with a fixed reply.

    Registered as a stream interrupter, which is where an agent's human lives:
    the toolkit's answerer reaches it through ``context.input()`` without
    knowing anything about MCP.
    """

    __slots__ = ("_answer", "asked")

    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.asked: list[str] = []

    async def __call__(self, event: HumanInputRequest, context: Context) -> None:
        self.asked.append(event.content)
        await context.send(HumanMessage.ensure_message(self._answer, parent_id=event.id))
        return None


def _counting_retries(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Count the toolkit's answered retries, which is what a *pause* costs.

    The handshake era answers inline over a standalone request and retries
    nothing; the modern era has no standalone request and must come back. Both
    complete, so the retry count is what tells the two paths apart from outside.
    """
    retries: list[str] = []
    original = _toolkit_module._retry_call

    async def counting(session: Any, name: str, arguments: str, responses: Any, state: Any) -> Any:
        retries.append(name)
        return await original(session, name, arguments, responses, state)

    monkeypatch.setattr(_toolkit_module, "_retry_call", counting)
    return retries


async def _ask_through_toolkit(url: str, human: "_Human | None", answering: AnswerPolicy) -> Any:
    """Call the served agent's conversational tool through the toolkit, as an agent would."""
    calling = Context(stream=MemoryStream())
    toolkit = MCPToolkit(
        MCPServerConfig(server_url=url, protocol_mode="auto"),
        answering=answering,
    )
    await toolkit.schemas(calling)
    proxy = next(t for t in toolkit.tools if t.name == "ask")
    call = ToolCallEvent(name="ask", arguments='{"message": "pick one"}')
    if human is None:
        return await proxy(call, calling)
    with calling.stream.where(HumanInputRequest).sub_scope(human, interrupt=True):
        return await proxy(call, calling)


def _text(result: ToolResultEvent) -> str:
    return "".join(p.content for p in result.result.parts if isinstance(p, TextInput))


@pytest.mark.asyncio
async def test_a_served_question_is_answered_by_the_calling_agents_human(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole feature, end to end, over a real socket and a real handshake."""
    runs: list[str] = []
    answers: list[str] = []
    human = _Human("blue")
    retries = _counting_retries(monkeypatch)

    async with _serving(MCPServer(_asking_agent(runs, answers), path="/mcp")) as url:
        result = await _ask_through_toolkit(url, human, AnswerPolicy(elicitation="ask"))

    assert human.asked == ["What colour?"], "the served agent's question never reached the calling human"
    assert answers == ["blue"], "the human's answer never reached the served tool"
    assert isinstance(result, ToolResultEvent), f"the call did not complete: {result}"
    assert "done" in _text(result)
    assert retries == ["ask"], "the modern era must answer by retrying the call"
    assert runs == ["ran"], "the served run restarted rather than resumed"


@pytest.mark.asyncio
async def test_the_handshake_era_answers_inline_with_no_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same scenario one revision back: a standalone request, nothing paused."""
    runs: list[str] = []
    answers: list[str] = []
    human = _Human("green")
    retries = _counting_retries(monkeypatch)

    async with _serving(MCPServer(_asking_agent(runs, answers), path="/mcp")) as url:
        calling = Context(stream=MemoryStream())
        # ``legacy`` is the default; named here because it is the subject.
        toolkit = MCPToolkit(
            MCPServerConfig(server_url=url, protocol_mode="legacy"),
            answering=AnswerPolicy(elicitation="ask"),
        )
        await toolkit.schemas(calling)
        proxy = next(t for t in toolkit.tools if t.name == "ask")
        with calling.stream.where(HumanInputRequest).sub_scope(human, interrupt=True):
            result = await proxy(ToolCallEvent(name="ask", arguments='{"message": "pick one"}'), calling)

    assert human.asked == ["What colour?"]
    assert answers == ["green"], "the human's answer never reached the served tool"
    assert isinstance(result, ToolResultEvent), f"the call did not complete: {result}"
    assert "done" in _text(result)
    assert retries == [], "the handshake era has a back-channel and must not pause"
    assert runs == ["ran"]


@pytest.mark.asyncio
async def test_a_calling_agent_that_will_not_answer_ends_the_served_turn_deliberately() -> None:
    """A refusal is a refusal, not a hang and not a broken connection.

    The calling operator enabled nothing, so the toolkit advertises no ability
    to answer and this server — which asks only clients that said they could —
    never puts the question. The served turn then ends through the fallback
    chain's last link, and the calling agent gets that failure as its tool
    result, with the connection intact.
    """
    runs: list[str] = []

    async with _serving(MCPServer(_asking_agent(runs), path="/mcp")) as url:
        result = await _ask_through_toolkit(url, None, AnswerPolicy())

    assert isinstance(result, ToolErrorEvent), f"expected a deliberate failure, got {result}"
    assert "Human input was requested but not provided" in str(result.error)
    assert runs == ["ran"], "the served tool never ran, so the turn failed before the question"
