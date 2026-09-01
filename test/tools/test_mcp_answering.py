# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""What an agent gives a third-party MCP server that asks it for something.

A proxied tool call can come back asking for input instead of returning a
result. Everything the agent might supply — its user's attention, its model, its
filesystem layout — is the operator's to give, so each is off until enabled and
advertised only when enabled.

The server here is a real ``mcp`` low-level server driven over the SDK's own
client, not a hand-written double of the session: the capability declaration, the
round-trip driver and the answer routing are all the SDK's, and a test that stubs
them would assert its own stub. Only the *server's* behaviour is scripted, which
is the part a third party owns.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

from mcp.client import Client
from mcp.client.session import ClientRequestContext
from mcp.server.lowlevel import Server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ClientCapabilities,
    CreateMessageRequest,
    CreateMessageRequestParams,
    ElicitRequest,
    ElicitRequestFormParams,
    ElicitResult,
    InputRequest,
    InputRequiredResult,
    ListRootsRequest,
    ListToolsResult,
    PaginatedRequestParams,
    SamplingMessage,
    TextContent,
)
from mcp.types import Tool as MCPTool
from mcp_types.version import LATEST_MODERN_VERSION

from ag2 import Agent, Context
from ag2.events import HumanInputRequest, HumanMessage, TextInput, ToolCallEvent, ToolErrorEvent, ToolResultEvent
from ag2.exceptions import HumanInputNotProvidedError
from ag2.stream import MemoryStream
from ag2.testing import TestConfig
from ag2.tools import AnswerPolicy, MCPStdioServerConfig, MCPToolkit
from ag2.tools.toolkits.mcp_server import toolkit as _toolkit_module
from ag2.tools.toolkits.mcp_server.answering import InputRequestAnswerer
from ag2.utils import MODEL_CONFIG_CONTEXT_DEPENDENCY_KEY

TOOL = "needs_input"


def elicitation(message: str, *, field: str = "answer") -> ElicitRequest:
    """A one-property form, which is the shape a free-text answer fits."""
    return ElicitRequest(
        params=ElicitRequestFormParams(
            message=message,
            requested_schema={"type": "object", "properties": {field: {"type": "string"}}, "required": [field]},
        )
    )


def sampling(prompt: str) -> CreateMessageRequest:
    return CreateMessageRequest(
        params=CreateMessageRequestParams(
            messages=[SamplingMessage(role="user", content=TextContent(type="text", text=prompt))],
            max_tokens=64,
        )
    )


class ThirdPartyServer:
    """A server that answers ``needs_input`` with input requests, then a result.

    Each entry in ``rounds`` is one ``InputRequiredResult`` worth of requests;
    once they are exhausted the call returns a result naming what came back, so a
    test can read the answers *the server actually received* rather than the ones
    the client believes it sent.
    """

    __slots__ = ("_rounds", "received", "capabilities", "server")

    def __init__(self, *rounds: dict[str, InputRequest]) -> None:
        self._rounds = list(rounds)
        self.received: list[dict[str, Any]] = []
        self.capabilities: ClientCapabilities | None = None
        self.server = Server(
            name="third-party",
            version="1.0.0",
            on_list_tools=self._on_list_tools,
            on_call_tool=self._on_call_tool,
        )

    async def _on_list_tools(self, ctx: Any, params: PaginatedRequestParams | None) -> ListToolsResult:
        self.capabilities = ctx.session.client_capabilities
        return ListToolsResult(
            tools=[MCPTool(name=TOOL, description="Asks for things", inputSchema={"type": "object"})]
        )

    async def _on_call_tool(self, ctx: Any, params: CallToolRequestParams) -> Any:
        self.capabilities = ctx.session.client_capabilities
        if params.input_responses:
            self.received.append({
                key: response.model_dump(mode="json", exclude_none=True)
                for key, response in params.input_responses.items()
            })
        if self._rounds:
            return InputRequiredResult(input_requests=self._rounds.pop(0), request_state=f"state-{len(self._rounds)}")
        return CallToolResult(content=[TextContent(type="text", text=f"served after {len(self.received)} answers")])


class Human:
    """The calling agent's own human-input channel."""

    __slots__ = ("_answer", "asked")

    def __init__(self, answer: str = "blue") -> None:
        self._answer = answer
        self.asked: list[str] = []

    async def __call__(self, event: HumanInputRequest, context: Context) -> None:
        self.asked.append(event.content)
        await context.send(HumanMessage.ensure_message(self._answer, parent_id=event.id))
        return None


@pytest.fixture
def calling_agent(monkeypatch: pytest.MonkeyPatch) -> "CallingAgent":
    return CallingAgent(monkeypatch)


class CallingAgent:
    """The AG2 side: a toolkit pointed at ``server``, called the way an agent calls it."""

    __slots__ = ("_monkeypatch",)

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._monkeypatch = monkeypatch

    def install(self, server: ThirdPartyServer) -> None:
        """Point the toolkit's session factory at ``server``, over the modern era.

        Only the transport is substituted: a real ``ClientSession`` still speaks
        to a real low-level server, so what the client advertises and how it
        routes an input request are the SDK's own behaviour.
        """

        @asynccontextmanager
        async def connected(_config: Any, **session_kwargs: Any) -> AsyncGenerator[Any]:
            async with Client(
                server.server,
                mode=LATEST_MODERN_VERSION,
                raise_exceptions=True,
                **session_kwargs,
            ) as client:
                yield client.session

        self._monkeypatch.setattr(_toolkit_module, "_mcp_session", connected)

    async def call(
        self,
        server: ThirdPartyServer,
        *,
        answering: AnswerPolicy,
        human: Human | None = None,
        config: TestConfig | None = None,
    ) -> ToolResultEvent | ToolErrorEvent:
        self.install(server)
        dependencies = {MODEL_CONFIG_CONTEXT_DEPENDENCY_KEY: config} if config is not None else {}
        context = Context(stream=MemoryStream(), dependencies=dependencies)
        toolkit = MCPToolkit(MCPStdioServerConfig(command="unused"), answering=answering)
        await toolkit.schemas(context)
        proxy = next(t for t in toolkit.tools if t.name == TOOL)
        call = ToolCallEvent(name=TOOL, arguments="{}")
        if human is None:
            return await proxy(call, context)
        with context.stream.where(HumanInputRequest).sub_scope(human, interrupt=True):
            return await proxy(call, context)

    async def ask_as_an_agent(self, server: ThirdPartyServer, *, answering: AnswerPolicy) -> str:
        """Drive the same call through a real ``Agent.ask``.

        The agent is what installs the default human-input hook, so this is the
        only setup in which "nobody could be asked" is the real failure rather
        than a context a test forgot to furnish.
        """
        self.install(server)
        agent = Agent(
            "caller",
            tools=[MCPToolkit(MCPStdioServerConfig(command="unused"), answering=answering)],
            config=TestConfig(ToolCallEvent(name=TOOL, arguments="{}"), "done"),
        )
        reply = await agent.ask("go")
        return reply.body


def text_of(result: ToolResultEvent) -> str:
    return "".join(p.content for p in result.result.parts if isinstance(p, TextInput))


@pytest.mark.asyncio
class TestAnsweringAQuestion:
    async def test_the_agents_human_answers_and_the_call_completes(self, calling_agent: CallingAgent) -> None:
        server = ThirdPartyServer({"q": elicitation("What colour?")})
        human = Human("blue")

        result = await calling_agent.call(server, answering=AnswerPolicy(elicitation="ask"), human=human)

        assert human.asked == ["What colour?"], "the server's question never reached the agent's human"
        assert server.received == [{"q": {"action": "accept", "content": {"answer": "blue"}}}]
        assert isinstance(result, ToolResultEvent), f"the call did not complete: {result}"
        assert text_of(result) == "served after 1 answers"

    async def test_a_form_of_more_than_one_field_is_declined_rather_than_invented(
        self, calling_agent: CallingAgent
    ) -> None:
        """Splitting one free-text answer across fields would be fabricating data."""
        server = ThirdPartyServer({
            "q": ElicitRequest(
                params=ElicitRequestFormParams(
                    message="Name and address?",
                    requested_schema={
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "address": {"type": "string"}},
                    },
                )
            )
        })
        human = Human()

        result = await calling_agent.call(server, answering=AnswerPolicy(elicitation="ask"), human=human)

        assert human.asked == [], "an unanswerable form still reached the human"
        assert server.received == [{"q": {"action": "decline"}}]
        assert isinstance(result, ToolResultEvent), "a decline is an answer; the call should still complete"

    async def test_a_question_answered_on_the_form_s_own_field(self, calling_agent: CallingAgent) -> None:
        """A third-party form names its property whatever it likes."""
        server = ThirdPartyServer({"q": elicitation("City?", field="city")})

        result = await calling_agent.call(server, answering=AnswerPolicy(elicitation="ask"), human=Human("Kraków"))

        assert server.received == [{"q": {"action": "accept", "content": {"city": "Kraków"}}}]
        assert isinstance(result, ToolResultEvent)

    async def test_with_no_human_hook_the_existing_failure_surfaces_rather_than_a_decline(
        self, calling_agent: CallingAgent
    ) -> None:
        """An absent channel is not a refusal, and must not be reported as one."""
        server = ThirdPartyServer({"q": elicitation("What colour?")})

        with pytest.raises(HumanInputNotProvidedError):
            await calling_agent.ask_as_an_agent(server, answering=AnswerPolicy(elicitation="ask"))

        assert server.received == [], "the server was told 'declined' by an operator who never declined"

    async def test_a_question_asked_anyway_is_declined(self) -> None:
        """A conforming server never asks — this is what a non-conforming one gets.

        Driven against the answerer directly: the whole point of the policy is
        that the callback is not wired at all, so there is no route through a
        conforming session to reach this.
        """
        answerer = InputRequestAnswerer(AnswerPolicy(elicitation="decline"), Context(stream=MemoryStream()))

        answer = await answerer.on_elicitation(
            ClientRequestContext(session=None, request_id="q", meta=None),  # type: ignore[arg-type]
            ElicitRequestFormParams(
                message="What colour?",
                requested_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
            ),
        )

        assert isinstance(answer, ElicitResult)
        assert answer.action == "decline"


@pytest.mark.asyncio
class TestWhatIsAdvertised:
    async def test_nothing_is_advertised_by_default(self, calling_agent: CallingAgent) -> None:
        """The default policy hands over nothing, so a conforming server asks for nothing."""
        server = ThirdPartyServer()

        await calling_agent.call(server, answering=AnswerPolicy())

        assert server.capabilities is not None
        assert server.capabilities.elicitation is None
        assert server.capabilities.sampling is None
        assert server.capabilities.roots is None

    async def test_only_the_enabled_capability_is_advertised(self, calling_agent: CallingAgent) -> None:
        server = ThirdPartyServer()

        await calling_agent.call(server, answering=AnswerPolicy(elicitation="ask"), human=Human())

        assert server.capabilities is not None
        assert server.capabilities.elicitation is not None
        assert server.capabilities.sampling is None
        assert server.capabilities.roots is None


@pytest.mark.asyncio
class TestLendingTheAgentsModel:
    async def test_a_sampling_request_runs_on_the_agents_own_model(self, calling_agent: CallingAgent) -> None:
        server = ThirdPartyServer({"s": sampling("Summarise this.")})

        result = await calling_agent.call(
            server,
            answering=AnswerPolicy(sampling=True),
            config=TestConfig("a summary"),
        )

        assert isinstance(result, ToolResultEvent), f"the call did not complete: {result}"
        [round_one] = server.received
        assert round_one["s"]["content"] == {"type": "text", "text": "a summary"}
        assert round_one["s"]["role"] == "assistant"

    async def test_sampling_is_advertised_only_when_enabled(self, calling_agent: CallingAgent) -> None:
        server = ThirdPartyServer()

        await calling_agent.call(server, answering=AnswerPolicy(sampling=True), config=TestConfig("unused"))

        assert server.capabilities is not None
        assert server.capabilities.sampling is not None
        assert server.capabilities.elicitation is None

    async def test_an_un_enabled_sampling_request_is_refused_rather_than_served(
        self, calling_agent: CallingAgent
    ) -> None:
        """Sampling has no ``decline`` arm, so a refusal is an error — but an answer either way."""
        server = ThirdPartyServer({"s": sampling("Summarise this.")})

        result = await calling_agent.call(server, answering=AnswerPolicy(), config=TestConfig("never used"))

        assert isinstance(result, ToolErrorEvent)
        assert server.received == []

    async def test_sampling_without_a_model_is_refused_rather_than_crashing(self, calling_agent: CallingAgent) -> None:
        server = ThirdPartyServer({"s": sampling("Summarise this.")})

        result = await calling_agent.call(server, answering=AnswerPolicy(sampling=True), config=None)

        assert isinstance(result, ToolErrorEvent)


@pytest.mark.asyncio
class TestReportingRoots:
    async def test_configured_roots_are_reported(self, calling_agent: CallingAgent, tmp_path: Path) -> None:
        server = ThirdPartyServer({"r": ListRootsRequest()})

        result = await calling_agent.call(server, answering=AnswerPolicy(roots=[str(tmp_path)]))

        assert isinstance(result, ToolResultEvent), f"the call did not complete: {result}"
        [round_one] = server.received
        assert round_one["r"]["roots"] == [{"uri": tmp_path.as_uri(), "name": tmp_path.name}]

    async def test_roots_are_advertised_only_when_configured(self, calling_agent: CallingAgent, tmp_path: Path) -> None:
        server = ThirdPartyServer()

        await calling_agent.call(server, answering=AnswerPolicy(roots=[str(tmp_path)]))

        assert server.capabilities is not None
        assert server.capabilities.roots is not None

    async def test_with_no_roots_configured_the_capability_is_absent_and_a_request_refused(
        self, calling_agent: CallingAgent
    ) -> None:
        """No roots is not an empty list of roots — the agent has nothing to report."""
        server = ThirdPartyServer({"r": ListRootsRequest()})

        result = await calling_agent.call(server, answering=AnswerPolicy())

        assert server.capabilities is not None
        assert server.capabilities.roots is None
        assert isinstance(result, ToolErrorEvent)


@pytest.mark.asyncio
class TestTheRoundBound:
    async def test_a_server_that_re_asks_forever_is_stopped(self, calling_agent: CallingAgent) -> None:
        server = ThirdPartyServer(*[{"q": elicitation("Again?")} for _ in range(5)])

        result = await calling_agent.call(
            server,
            answering=AnswerPolicy(elicitation="ask", max_rounds=2),
            human=Human(),
        )

        assert isinstance(result, ToolErrorEvent)
        assert "more than 2 rounds" in str(result.error)
        assert len(server.received) == 2, "the bound did not stop the loop where it said it would"

    async def test_the_bound_covers_sampling_and_roots_too(self, calling_agent: CallingAgent, tmp_path: Path) -> None:
        """One bound for the call, not one per request type."""
        server = ThirdPartyServer(*[{"s": sampling("again"), "r": ListRootsRequest()} for _ in range(5)])

        result = await calling_agent.call(
            server,
            answering=AnswerPolicy(sampling=True, roots=[str(tmp_path)], max_rounds=2),
            config=TestConfig(*["a summary"] * 5),
        )

        assert isinstance(result, ToolErrorEvent)
        assert "more than 2 rounds" in str(result.error)
