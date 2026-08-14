# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64
import threading
from collections.abc import Sequence
from typing import Any

import acp
import pytest
from acp import schema
from acp.exceptions import RequestError
from typing_extensions import Self

from ag2 import Agent, Context
from ag2.acp import ACPAgent
from ag2.acp.executor import META_VARIABLE, AgentExecutor, UpdateDeliveryError
from ag2.acp.sessions import SessionStore
from ag2.acp.testing import RecordingClient, connect
from ag2.config import LLMClient, ModelConfig
from ag2.events import BaseEvent, ModelMessageChunk, ModelResponse, ToolCallEvent
from ag2.testing import TestConfig


class _RecordingClient(LLMClient):
    """Wraps another client, capturing the full message list sent on each call."""

    def __init__(
        self,
        client: LLMClient,
        sink: list[list[BaseEvent]],
        variables: list[dict[Any, Any]],
        prompts: list[list[str]],
    ) -> None:
        self.client = client
        self.sink = sink
        self.variables = variables
        self.prompts = prompts

    async def __call__(self, messages: Sequence[BaseEvent], context: Context, **kwargs: Any) -> ModelResponse:
        self.sink.append(list(messages))
        self.variables.append(dict(context.variables))
        self.prompts.append(list(context.prompt))
        return await self.client(messages, context=context, **kwargs)


class _RecordingConfig(ModelConfig):
    """Records every message list the framework sends to the LLM, per turn."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.calls: list[list[BaseEvent]] = []
        self.variables: list[dict[Any, Any]] = []
        self.prompts: list[list[str]] = []

    def copy(self) -> Self:
        return self

    def create(self) -> _RecordingClient:
        return _RecordingClient(self.config.create(), self.calls, self.variables, self.prompts)

    def create_files_client(self) -> None:
        raise NotImplementedError


class _StreamingClient(LLMClient):
    """Emits the reply as chunks first, the way a streaming provider does."""

    def __init__(self, client: LLMClient, chunks: Sequence[str]) -> None:
        self.client = client
        self.chunks = chunks

    async def __call__(self, messages: Sequence[BaseEvent], context: Context, **kwargs: Any) -> ModelResponse:
        for chunk in self.chunks:
            await context.send(ModelMessageChunk(chunk))
        return await self.client(messages, context=context, **kwargs)


class _StreamingConfig(ModelConfig):
    """Streams ``chunks`` and then returns their concatenation as the final reply."""

    def __init__(self, *chunks: str) -> None:
        self.chunks = chunks
        self.config = TestConfig("".join(chunks))

    def copy(self) -> Self:
        return self

    def create(self) -> _StreamingClient:
        return _StreamingClient(self.config.create(), self.chunks)

    def create_files_client(self) -> None:
        raise NotImplementedError


def _agent(*turns: object) -> Agent:
    return Agent("workie", config=TestConfig(*(turns or ("ok",))))


def _recording_agent(*turns: object) -> tuple[Agent, _RecordingConfig]:
    config = _RecordingConfig(TestConfig(*(turns or ("ok",))))
    return Agent("workie", config=config), config


def _texts(updates: list[Any]) -> list[str]:
    """The text of every ``agent_message_chunk``, in arrival order."""
    return [u.content.text for u in updates if isinstance(u, schema.AgentMessageChunk)]


@pytest.mark.asyncio
class TestPromptTurn:
    async def test_the_reply_reaches_the_client(self) -> None:
        async with connect(ACPAgent(_agent("200"))) as (conn, recorder):
            session = await conn.new_session(cwd="/tmp")
            response = await conn.prompt(
                session_id=session.session_id,
                prompt=[acp.text_block("what's 100 + 100")],
            )

        assert response.stop_reason == "end_turn"
        assert _texts(recorder.updates_for(session.session_id)) == ["200"]

    async def test_an_unknown_session_is_a_protocol_error(self) -> None:
        async with connect(ACPAgent(_agent())) as (conn, _):
            with pytest.raises(RequestError):
                await conn.prompt(session_id="never-issued", prompt=[acp.text_block("hi")])

    async def test_prompt_text_reaches_the_agent(self) -> None:
        agent, config = _recording_agent()

        async with connect(ACPAgent(agent)) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("hello agent")])

        assert "hello agent" in str(config.calls[-1])

    async def test_multiple_text_blocks_all_reach_the_agent(self) -> None:
        agent, config = _recording_agent()

        async with connect(ACPAgent(agent)) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            await conn.prompt(
                session_id=session.session_id,
                prompt=[acp.text_block("first"), acp.text_block("second")],
            )

        rendered = str(config.calls[-1])
        assert "first" in rendered
        assert "second" in rendered

    async def test_an_embedded_text_resource_reaches_the_agent(self) -> None:
        agent, config = _recording_agent()
        resource = schema.EmbeddedResourceContentBlock(
            type="resource",
            resource=schema.TextResourceContents(uri="file:///notes.md", text="the embedded body"),
        )

        async with connect(ACPAgent(agent)) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=session.session_id, prompt=[resource])

        assert "the embedded body" in str(config.calls[-1])

    async def test_an_image_block_is_accepted(self) -> None:
        image = schema.ImageContentBlock(
            type="image",
            data=base64.b64encode(b"fake-png").decode(),
            mime_type="image/png",
        )

        async with connect(ACPAgent(_agent("saw it"))) as (conn, recorder):
            session = await conn.new_session(cwd="/tmp")
            response = await conn.prompt(session_id=session.session_id, prompt=[image])

        assert response.stop_reason == "end_turn"
        assert _texts(recorder.updates_for(session.session_id)) == ["saw it"]

    async def test_a_resource_link_is_referenced_but_never_fetched(self) -> None:
        agent, config = _recording_agent()
        link = schema.ResourceContentBlock(type="resource_link", uri="file:///etc/passwd", name="passwd")

        async with connect(ACPAgent(agent)) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=session.session_id, prompt=[link])

        rendered = str(config.calls[-1])
        assert "file:///etc/passwd" in rendered  # the reference is visible
        assert "root:" not in rendered  # the file itself was not read

    async def test_an_unhandled_turn_failure_becomes_a_protocol_error(self) -> None:
        agent = _agent(ToolCallEvent(name="boom", arguments="{}"), "recovered")

        @agent.tool
        def boom() -> str:
            """Always raises."""
            raise ValueError("kaboom")

        async with connect(ACPAgent(agent)) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            with pytest.raises(RequestError):
                await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("go")])

    async def test_a_failure_tells_the_client_what_went_wrong(self) -> None:
        """The Client is another process — an error without its cause is unusable."""
        agent = _agent(ToolCallEvent(name="boom", arguments="{}"), "recovered")

        @agent.tool
        def boom() -> str:
            """Always raises."""
            raise ValueError("the specific thing that broke")

        async with connect(ACPAgent(agent)) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            with pytest.raises(RequestError) as caught:
                await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("go")])

        assert caught.value.data == {
            "reason": "the specific thing that broke",
            "type": "ValueError",
        }

    async def test_a_config_failure_reaches_the_client(self) -> None:
        """The common real-world case: the model rejected the request."""

        class _Exploding(ModelConfig):
            def copy(self) -> Self:
                return self

            def create(self) -> LLMClient:
                raise RuntimeError("Could not resolve authentication method.")

            def create_files_client(self) -> None:
                raise NotImplementedError

        async with connect(ACPAgent(Agent("workie", config=_Exploding()))) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            with pytest.raises(RequestError) as caught:
                await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("go")])

        assert "authentication" in caught.value.data["reason"]

    async def test_a_failing_tool_is_reported_before_the_turn_dies(self) -> None:
        """The Client learns *why* a turn failed, not just that it did."""
        agent = _agent(ToolCallEvent(name="boom", arguments="{}"), "recovered")

        @agent.tool
        def boom() -> str:
            """Always raises."""
            raise ValueError("kaboom")

        async with connect(ACPAgent(agent)) as (conn, recorder):
            session = await conn.new_session(cwd="/tmp")
            with pytest.raises(RequestError):
                await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("go")])

        [progress] = [u for u in recorder.updates_for(session.session_id) if isinstance(u, schema.ToolCallProgress)]
        assert progress.status == "failed"
        assert "kaboom" in progress.content[0].content.text

    async def test_a_failed_turn_does_not_leave_the_session_locked(self) -> None:
        """A failure must release the turn lock, or the session would hang forever.

        The follow-up prompt still fails — AG2 persists the unhandled tool error
        and re-raises it on every later turn of that stream (reproducible with a
        plain ``agent.ask`` on a shared stream, nothing to do with ACP). What
        matters here is that it *returns* rather than blocking.
        """
        agent = _agent(ToolCallEvent(name="boom", arguments="{}"), "recovered")

        @agent.tool
        def boom() -> str:
            """Always raises."""
            raise ValueError("kaboom")

        async with connect(ACPAgent(agent)) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            for _ in range(2):
                with pytest.raises(RequestError):
                    await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("go")])


@pytest.mark.asyncio
class TestUpdateProjection:
    @staticmethod
    def _adding_agent() -> Agent:
        agent = _agent(ToolCallEvent(name="add", arguments='{"a": 100, "b": 100}'), "200")

        @agent.tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        return agent

    async def test_tool_activity_is_reported_in_order(self) -> None:
        async with connect(ACPAgent(self._adding_agent())) as (conn, recorder):
            session = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("100+100")])

        kinds = [u.session_update for u in recorder.updates_for(session.session_id)]
        assert kinds == ["tool_call", "tool_call_update", "agent_message_chunk"]

    async def test_a_tool_result_carries_its_value(self) -> None:
        async with connect(ACPAgent(self._adding_agent())) as (conn, recorder):
            session = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("100+100")])

        [progress] = [u for u in recorder.updates_for(session.session_id) if isinstance(u, schema.ToolCallProgress)]
        assert progress.status == "completed"
        assert progress.content[0].content.text == "200"

    async def test_a_tool_call_and_its_result_share_an_id(self) -> None:
        async with connect(ACPAgent(self._adding_agent())) as (conn, recorder):
            session = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("100+100")])

        updates = recorder.updates_for(session.session_id)
        [start] = [u for u in updates if isinstance(u, schema.ToolCallStart)]
        [progress] = [u for u in updates if isinstance(u, schema.ToolCallProgress)]
        assert start.tool_call_id == progress.tool_call_id

    async def test_a_tool_call_carries_its_arguments(self) -> None:
        async with connect(ACPAgent(self._adding_agent())) as (conn, recorder):
            session = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("100+100")])

        [start] = [u for u in recorder.updates_for(session.session_id) if isinstance(u, schema.ToolCallStart)]
        assert start.title == "add"
        assert start.raw_input == {"a": 100, "b": 100}

    async def test_a_non_streaming_reply_still_reaches_the_client(self) -> None:
        """No chunks were emitted, so the assembled reply must be sent instead."""
        async with connect(ACPAgent(_agent("just once"))) as (conn, recorder):
            session = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("hi")])

        assert _texts(recorder.updates_for(session.session_id)) == ["just once"]

    async def test_streamed_chunks_reach_the_client_in_order(self) -> None:
        agent = Agent("workie", config=_StreamingConfig("Paris is ", "the capital ", "of France."))

        async with connect(ACPAgent(agent)) as (conn, recorder):
            session = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("capital of France?")])

        assert _texts(recorder.updates_for(session.session_id)) == [
            "Paris is ",
            "the capital ",
            "of France.",
        ]

    async def test_a_streamed_reply_is_not_repeated_at_the_end(self) -> None:
        """The final response echoes the whole answer; re-sending it would double it."""
        agent = Agent("workie", config=_StreamingConfig("Paris is ", "the capital ", "of France."))

        async with connect(ACPAgent(agent)) as (conn, recorder):
            session = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("capital of France?")])

        assembled = "".join(_texts(recorder.updates_for(session.session_id)))
        assert assembled == "Paris is the capital of France."

    async def test_reasoning_is_withheld_by_default(self) -> None:
        async with connect(ACPAgent(_agent("answer"))) as (conn, recorder):
            session = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("hi")])

        thoughts = [u for u in recorder.updates_for(session.session_id) if isinstance(u, schema.AgentThoughtChunk)]
        assert thoughts == []


@pytest.mark.asyncio
class TestSessionIsolation:
    async def test_one_session_never_sees_another_prompt(self) -> None:
        agent, config = _recording_agent()

        async with connect(ACPAgent(agent)) as (conn, _):
            first = await conn.new_session(cwd="/tmp")
            second = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=first.session_id, prompt=[acp.text_block("secret to first")])
            await conn.prompt(session_id=second.session_id, prompt=[acp.text_block("hello second")])

        assert "secret to first" not in str(config.calls[-1])
        assert "hello second" in str(config.calls[-1])

    async def test_history_accumulates_within_one_session(self) -> None:
        agent, config = _recording_agent()

        async with connect(ACPAgent(agent)) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("remember this")])
            await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("and now")])

        assert "remember this" in str(config.calls[-1])

    async def test_every_update_is_tagged_with_its_own_session(self) -> None:
        async with connect(ACPAgent(_agent("reply"))) as (conn, recorder):
            first = await conn.new_session(cwd="/tmp")
            second = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=first.session_id, prompt=[acp.text_block("one")])
            await conn.prompt(session_id=second.session_id, prompt=[acp.text_block("two")])

        assert len(recorder.updates_for(first.session_id)) == 1
        assert len(recorder.updates_for(second.session_id)) == 1


@pytest.mark.asyncio
class TestSessionContext:
    async def test_client_context_is_recorded_but_not_acted_on(self) -> None:
        server = ACPAgent(_agent())

        async with connect(server) as (conn, _):
            created = await conn.new_session(cwd="/work", additional_directories=["/extra"])
            session = await server.sessions.get(created.session_id)

        assert session.cwd == "/work"
        assert session.additional_directories == ["/extra"]

    async def test_declared_mcp_servers_are_recorded_but_never_connected(self) -> None:
        server = ACPAgent(_agent())
        declared = schema.HttpMcpServer(type="http", name="evil", url="http://127.0.0.1:9/mcp", headers=[])

        async with connect(server) as (conn, _):
            created = await conn.new_session(cwd="/tmp", mcp_servers=[declared])
            session = await server.sessions.get(created.session_id)

        # Captured for an embedding application to inspect — and nothing more.
        assert session.mcp_servers == [declared]

    async def test_every_declared_transport_round_trips_as_its_own_model(self) -> None:
        """A recorded server keeps the shape the Client declared it in.

        ``session/new`` crosses the wire as JSON, so a declared server is only
        ever as good as what comes back out of it — a stdio entry that lands as a
        dict, or as the wrong union member, is recorded just as happily. Every
        shape ``NewSessionRequest.mcp_servers`` admits is pinned once, here.
        """
        server = ACPAgent(_agent())
        declared = [
            schema.HttpMcpServer(type="http", name="over-http", url="http://127.0.0.1:9/mcp", headers=[]),
            schema.SseMcpServer(type="sse", name="over-sse", url="http://127.0.0.1:9/sse", headers=[]),
            schema.AcpMcpServer(type="acp", name="over-acp", server_id="peer-1"),
            schema.McpServerStdio(name="over-stdio", command="/bin/true", args=["--serve"], env=[]),
        ]

        async with connect(server) as (conn, _):
            created = await conn.new_session(cwd="/tmp", mcp_servers=declared)
            session = await server.sessions.get(created.session_id)

        # Pydantic equality is class-sensitive, so this pins the model too.
        assert session.mcp_servers == declared


@pytest.mark.asyncio
class TestDeliveryFailures:
    """A turn must not report success for output the Client never received."""

    class _DeadClient(RecordingClient):
        async def session_update(self, *, session_id: str, update: Any, **kwargs: Any) -> None:
            raise ConnectionResetError("client went away")

    async def test_an_undeliverable_turn_is_not_reported_as_end_turn(self) -> None:
        store = SessionStore()
        session = await store.create()
        executor = AgentExecutor(_agent("the answer"))

        with pytest.raises(UpdateDeliveryError):
            await executor.run_turn(
                session=session,
                store=store,
                client=self._DeadClient(),
                blocks=[acp.text_block("hi")],
            )

    async def test_history_survives_a_delivery_failure(self) -> None:
        """The turn ran; only the telling failed."""
        store = SessionStore()
        session = await store.create()
        executor = AgentExecutor(_agent("the answer"))

        with pytest.raises(UpdateDeliveryError):
            await executor.run_turn(
                session=session,
                store=store,
                client=self._DeadClient(),
                blocks=[acp.text_block("hi")],
            )

        assert list(await store.stream(session).history.get_events())


@pytest.mark.asyncio
class TestRequestMetadata:
    """ACP `_meta` is where applications put provenance; it must survive intact."""

    @staticmethod
    def _raw(conn: Any) -> Any:
        return conn._conn if hasattr(conn, "_conn") else conn._connection

    async def test_wire_meta_is_captured_on_the_session(self) -> None:
        server = ACPAgent(_agent())

        async with connect(server) as (conn, _):
            response = await self._raw(conn).send_request(
                "session/new",
                {"cwd": "/tmp", "mcpServers": [], "_meta": {"ag2.space": {"room": "!r"}}},
            )
            session = await server.sessions.get(response["sessionId"])

        assert session.meta == {"ag2.space": {"room": "!r"}}

    async def test_a_session_without_meta_carries_none(self) -> None:
        server = ACPAgent(_agent())

        async with connect(server) as (conn, _):
            created = await conn.new_session(cwd="/tmp")
            session = await server.sessions.get(created.session_id)

        assert session.meta == {}

    async def test_prompt_meta_reaches_the_agent_as_a_context_variable(self) -> None:
        agent, config = _recording_agent()
        server = ACPAgent(agent)

        async with connect(server) as (conn, _):
            created = await conn.new_session(cwd="/tmp")
            await self._raw(conn).send_request(
                "session/prompt",
                {
                    "sessionId": created.session_id,
                    "prompt": [{"type": "text", "text": "hi"}],
                    "_meta": {"ag2.space": {"event": "$abc"}},
                },
            )

        assert config.variables[-1][META_VARIABLE] == {"ag2.space": {"event": "$abc"}}


@pytest.mark.asyncio
class TestDynamicPrompt:
    """``@agent.prompt`` hooks routinely carry per-request policy.

    An agent that resolves them under ``ask`` but not over ACP is a *less
    constrained* agent once it is served, which is the opposite of what a
    transport should do to it.
    """

    async def test_a_dynamic_prompt_is_resolved(self) -> None:
        agent, config = _recording_agent()

        @agent.prompt
        def policy(ctx: Context) -> str:
            return "POLICY: decline refunds."

        async with connect(ACPAgent(agent)) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("refund me")])

        assert config.prompts[-1] == ["POLICY: decline refunds."]

    async def test_the_static_prompt_still_comes_first(self) -> None:
        config = _RecordingConfig(TestConfig("ok"))
        agent = Agent("workie", prompt="You are workie.", config=config)

        @agent.prompt
        def policy(ctx: Context) -> str:
            return "POLICY: decline refunds."

        async with connect(ACPAgent(agent)) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("hi")])

        assert config.prompts[-1] == ["You are workie.", "POLICY: decline refunds."]

    async def test_a_dynamic_prompt_reads_a_context_variable(self) -> None:
        config = _RecordingConfig(TestConfig("ok"))
        agent = Agent("workie", config=config, variables={"tier": "enterprise"})

        @agent.prompt
        def tiered(ctx: Context) -> str:
            return f"Caller tier: {ctx.variables['tier']}."

        async with connect(ACPAgent(agent)) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("hi")])

        assert config.prompts[-1] == ["Caller tier: enterprise."]

    async def test_a_dynamic_prompt_reads_the_client_meta(self) -> None:
        """The Client's ``_meta`` is per-request, so a hook is where it gets used."""
        agent, config = _recording_agent()

        @agent.prompt
        def provenance(ctx: Context) -> str:
            return f"Room: {ctx.variables[META_VARIABLE]['ag2.space']['room']}."

        store = SessionStore()
        session = await store.create()

        await AgentExecutor(agent).run_turn(
            session=session,
            store=store,
            client=RecordingClient(),
            blocks=[acp.text_block("hi")],
            meta={"ag2.space": {"room": "!r"}},
        )

        assert config.prompts[-1] == ["Room: !r."]

    async def test_the_prompt_matches_what_ask_sends(self) -> None:
        """Parity is the whole point: one agent, one prompt, whatever the transport."""
        config = _RecordingConfig(TestConfig("ok"))
        agent = Agent("workie", prompt="You are workie.", config=config)

        @agent.prompt
        def policy(ctx: Context) -> str:
            return "POLICY: decline refunds."

        async with connect(ACPAgent(agent)) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("hi")])

        over_acp = config.prompts[-1]
        await agent.ask("hi")

        assert over_acp == config.prompts[-1]


class _MultiCallConfig(ModelConfig):
    """Streams a different script on each model call of one turn.

    A tool-using turn makes two LLM calls; a chatty model may stream text in both
    (a "let me check..." preamble, then the answer).
    """

    def __init__(self, *scripts: Sequence[str], inner: ModelConfig) -> None:
        self.scripts = scripts
        self.inner = inner

    def copy(self) -> Self:
        return self

    def create(self) -> LLMClient:
        return _MultiCallClient(self.inner.create(), self.scripts)

    def create_files_client(self) -> None:
        raise NotImplementedError


class _MultiCallClient(LLMClient):
    def __init__(self, client: LLMClient, scripts: Sequence[Sequence[str]]) -> None:
        self.client = client
        self.scripts = scripts
        self.call = 0

    async def __call__(self, messages: Sequence[BaseEvent], context: Context, **kwargs: Any) -> ModelResponse:
        for chunk in self.scripts[min(self.call, len(self.scripts) - 1)]:
            await context.send(ModelMessageChunk(chunk))
        self.call += 1
        return await self.client(messages, context=context, **kwargs)


@pytest.mark.asyncio
class TestStreamingToolTurns:
    """De-dup must compare against the final model call, not the whole turn."""

    @staticmethod
    def _agent_streaming_both_calls() -> Agent:
        config = _MultiCallConfig(
            ["Let me calculate. "],
            ["The answer is 4."],
            inner=TestConfig(ToolCallEvent(name="add", arguments='{"a": 2, "b": 2}'), "The answer is 4."),
        )
        agent = Agent("workie", config=config)

        @agent.tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        return agent

    async def test_the_answer_is_not_sent_twice(self) -> None:
        async with connect(ACPAgent(self._agent_streaming_both_calls())) as (conn, recorder):
            session = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("2+2")])

        assert _texts(recorder.updates_for(session.session_id)) == [
            "Let me calculate. ",
            "The answer is 4.",
        ]

    async def test_the_preamble_still_reaches_the_client(self) -> None:
        async with connect(ACPAgent(self._agent_streaming_both_calls())) as (conn, recorder):
            session = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("2+2")])

        assert "Let me calculate. " in _texts(recorder.updates_for(session.session_id))

    async def test_a_silent_first_call_still_delivers_the_answer(self) -> None:
        """The common shape: the tool-selection call streams nothing."""
        config = _MultiCallConfig(
            [],
            ["The answer is 4."],
            inner=TestConfig(ToolCallEvent(name="add", arguments='{"a": 2, "b": 2}'), "The answer is 4."),
        )
        agent = Agent("workie", config=config)

        @agent.tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        async with connect(ACPAgent(agent)) as (conn, recorder):
            session = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("2+2")])

        assert _texts(recorder.updates_for(session.session_id)) == ["The answer is 4."]


@pytest.mark.asyncio
class TestSessionVariables:
    """Variables belong to a conversation: they persist in it, and only in it."""

    @staticmethod
    def _counting_agent(reports: list[tuple[int, int]]) -> Agent:
        agent = Agent(
            "workie",
            variables={"seen": [], "n": 0},
            config=TestConfig(
                ToolCallEvent(name="touch", arguments="{}"),
                "ok",
                ToolCallEvent(name="touch", arguments="{}"),
                "ok",
            ),
        )

        @agent.tool
        def touch(ctx: Context) -> str:
            """Mutate one nested and one top-level variable."""
            ctx.variables["seen"].append("x")
            ctx.variables["n"] = ctx.variables["n"] + 1
            reports.append((len(ctx.variables["seen"]), ctx.variables["n"]))
            return "done"

        return agent

    async def test_writes_persist_across_turns_of_one_session(self) -> None:
        """The same continuity ``AgentReply.ask`` gives an off-protocol conversation."""
        reports: list[tuple[int, int]] = []

        async with connect(ACPAgent(self._counting_agent(reports))) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            for _ in range(2):
                await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("go")])

        assert reports == [(1, 1), (2, 2)]

    async def test_a_nested_value_is_not_shared_between_sessions(self) -> None:
        """A shallow copy would leave this list shared and the count would keep climbing."""
        reports: list[tuple[int, int]] = []

        async with connect(ACPAgent(self._counting_agent(reports))) as (conn, _):
            first = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=first.session_id, prompt=[acp.text_block("go")])
            second = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=second.session_id, prompt=[acp.text_block("go")])

        assert reports == [(1, 1), (1, 1)]

    async def test_the_agent_s_own_defaults_are_never_mutated(self) -> None:
        reports: list[tuple[int, int]] = []
        agent = self._counting_agent(reports)

        async with connect(ACPAgent(agent)) as (conn, _):
            session = await conn.new_session(cwd="/tmp")
            await conn.prompt(session_id=session.session_id, prompt=[acp.text_block("go")])

        assert agent._agent_variables == {"seen": [], "n": 0}

    async def test_an_uncopyable_default_is_shared_rather_than_failing(self) -> None:
        """Degrading loudly beats refusing to open a session over one odd value.

        ``threading.Lock`` is the shape this guards: a real handle that a deep
        copy cannot reproduce.
        """
        agent = Agent("workie", variables={"lock": threading.Lock()}, config=TestConfig("ok"))
        server = ACPAgent(agent)

        async with connect(server) as (conn, _):
            created = await conn.new_session(cwd="/tmp")
            session = await server.sessions.get(created.session_id)

        assert session.variables["lock"] is agent._agent_variables["lock"]
