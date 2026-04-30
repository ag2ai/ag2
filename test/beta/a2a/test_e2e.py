# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from uuid import uuid4

import httpx
import pytest
from a2a.server.agent_execution import AgentExecutor as A2ABaseAgentExecutor
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, InternalError, Task, TaskState, TaskStatus
from a2a.utils.errors import ServerError
from starlette.responses import JSONResponse, Response

from autogen.beta import Agent, MemoryStream
from autogen.beta.a2a import A2AConfig, A2AServer
from autogen.beta.a2a.errors import A2AAuthRequiredError, A2ATaskFailedError, A2ATaskRejectedError
from autogen.beta.a2a.utils import CONTEXT_ID_VAR_KEY, PROVIDER_NAME, TASK_ID_VAR_KEY
from autogen.beta.context import ConversationContext
from autogen.beta.events import ModelMessageChunk, ModelRequest, TextInput, ToolCallEvent
from autogen.beta.testing import TestConfig


@pytest.mark.asyncio
class TestSimpleRoundTrip:
    async def test_returns_servers_text(self, serve) -> None:
        agent = Agent("specialist", "be helpful", config=TestConfig("hello-from-remote"))
        env = serve(agent)
        client = env.config.create()
        ctx = ConversationContext(stream=MemoryStream())

        response = await client(
            [ModelRequest([TextInput("hi")])],
            ctx,
            tools=[],
            response_schema=None,
            serializer=None,  # type: ignore[arg-type]
        )

        assert response.message and response.message.content == "hello-from-remote"

    async def test_response_metadata(self, serve) -> None:
        agent = Agent("specialist", "p", config=TestConfig("ok"))
        env = serve(agent)
        client = env.config.create()
        ctx = ConversationContext(stream=MemoryStream())

        response = await client(
            [ModelRequest([TextInput("hi")])],
            ctx,
            tools=[],
            response_schema=None,
            serializer=None,  # type: ignore[arg-type]
        )

        assert response.provider == PROVIDER_NAME
        assert response.model == "specialist"
        assert response.finish_reason == "completed"

    async def test_context_and_task_ids_persisted_on_context(self, serve) -> None:
        agent = Agent("specialist", "p", config=TestConfig("ok"))
        env = serve(agent)
        client = env.config.create()
        ctx = ConversationContext(stream=MemoryStream())

        await client(
            [ModelRequest([TextInput("hi")])],
            ctx,
            tools=[],
            response_schema=None,
            serializer=None,  # type: ignore[arg-type]
        )

        assert CONTEXT_ID_VAR_KEY in ctx.variables
        assert TASK_ID_VAR_KEY in ctx.variables


@pytest.mark.asyncio
async def test_client_upgrades_to_extended_card_when_advertised() -> None:
    # Server publishes a public card with `supports_authenticated_extended_card`
    # set, plus an extended card with a different `name`. The client's
    # ModelResponse.model should reflect the extended card's name — proof that
    # the client fetched /agent/authenticatedExtendedCard.
    agent = Agent("specialist", "p", config=TestConfig("ok"))
    extended = AgentCard(
        name="specialist-extended",
        description="extra",
        url="http://test",
        version="0.2.0",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[],
    )
    server = A2AServer(agent, extended_card=extended, url="http://test")
    asgi = server.build_asgi()
    transport = httpx.ASGITransport(app=asgi)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        config = A2AConfig("http://test", client_factory=lambda: http)
        client = config.create()
        ctx = ConversationContext(stream=MemoryStream())

        response = await client(
            [ModelRequest([TextInput("hi")])],
            ctx,
            tools=[],
            response_schema=None,
            serializer=None,  # type: ignore[arg-type]
        )

    assert response.model == "specialist-extended"


@pytest.mark.asyncio
async def test_falls_back_to_prev_agent_card_path_on_404() -> None:
    # Servers running a2a-sdk 0.2.x serve the AgentCard at the legacy
    # /.well-known/agent.json path (current spec uses /.well-known/agent-card.json).
    # The client must fall back so it can still talk to those legacy servers.
    agent = Agent("specialist", "p", config=TestConfig("legacy-ok"))
    legacy_card = AgentCard(
        name="legacy-server",
        description="from old path",
        url="http://test",
        version="0.1.0",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[],
    )
    server = A2AServer(agent, card=legacy_card, url="http://test")
    inner_asgi = server.build_asgi()
    legacy_payload = legacy_card.model_dump(mode="json", exclude_none=True, by_alias=True)

    async def wrapped_asgi(scope, receive, send) -> None:
        if scope["type"] == "http":
            path = scope.get("path", "")
            if path == "/.well-known/agent-card.json":
                await Response(status_code=404)(scope, receive, send)
                return
            if path == "/.well-known/agent.json":
                await JSONResponse(legacy_payload)(scope, receive, send)
                return
        await inner_asgi(scope, receive, send)

    transport = httpx.ASGITransport(app=wrapped_asgi)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        config = A2AConfig("http://test", client_factory=lambda: http)
        client = config.create()
        ctx = ConversationContext(stream=MemoryStream())

        response = await client(
            [ModelRequest([TextInput("hi")])],
            ctx,
            tools=[],
            response_schema=None,
            serializer=None,  # type: ignore[arg-type]
        )

    assert response.model == "legacy-server"
    assert response.message and response.message.content == "legacy-ok"


@pytest.mark.asyncio
async def test_polling_fallback_for_non_streaming_card(serve) -> None:
    # Server card advertises streaming=False → A2AClient must reach the agent
    # via the polling code path (delegated to a2a-sdk via ClientConfig).
    agent = Agent("specialist", "p", config=TestConfig("via-polling"))
    env = serve(agent, streaming=False)
    client = env.config.create()
    ctx = ConversationContext(stream=MemoryStream())

    response = await client(
        [ModelRequest([TextInput("hi")])],
        ctx,
        tools=[],
        response_schema=None,
        serializer=None,  # type: ignore[arg-type]
    )

    assert response.message and response.message.content == "via-polling"
    assert response.finish_reason == "completed"


@pytest.mark.asyncio
async def test_chunks_arrive_in_context_stream(serve) -> None:
    agent = Agent("specialist", "p", config=TestConfig("hello"))
    env = serve(agent)
    client = env.config.create()

    chunks: list[str] = []
    stream = MemoryStream()

    @stream.where(ModelMessageChunk).subscribe
    async def collect(c: ModelMessageChunk) -> None:
        chunks.append(c.content)

    ctx = ConversationContext(stream=stream)
    response = await client(
        [ModelRequest([TextInput("hi")])],
        ctx,
        tools=[],
        response_schema=None,
        serializer=None,  # type: ignore[arg-type]
    )

    assert response.message is not None
    assert response.message.content == "hello"
    # Server card declares streaming=True (default), so chunks must arrive.
    assert "".join(chunks) == "hello"


def _bootstrap_task(context: RequestContext) -> Task:
    assert context.message is not None
    return context.current_task or Task(
        id=context.message.task_id or uuid4().hex,
        context_id=context.message.context_id or uuid4().hex,
        status=TaskStatus(state=TaskState.submitted, timestamp=datetime.now(timezone.utc).isoformat()),
        history=[context.message],
    )


class _RejectingExecutor(A2ABaseAgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = _bootstrap_task(context)
        if context.current_task is None:
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.reject()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=InternalError(message="not supported"))


class _AuthRequiredExecutor(A2ABaseAgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = _bootstrap_task(context)
        if context.current_task is None:
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.requires_auth()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=InternalError(message="not supported"))


@pytest.mark.asyncio
class TestTaskTerminalErrors:
    async def test_failed_task_raises_a2a_task_failed_error(self, serve) -> None:
        # Tool raises → TestClient re-raises on next round → executor wraps
        # in ServerError → task ends in failed state → client must raise.
        agent = Agent(
            "specialist",
            "p",
            config=TestConfig(ToolCallEvent(name="boom", arguments="{}")),
        )

        @agent.tool
        def boom() -> str:
            raise RuntimeError("kaboom")

        env = serve(agent)
        client = env.config.create()
        ctx = ConversationContext(stream=MemoryStream())

        with pytest.raises(A2ATaskFailedError) as exc_info:
            await client(
                [ModelRequest([TextInput("hi")])],
                ctx,
                tools=[],
                response_schema=None,
                serializer=None,  # type: ignore[arg-type]
            )

        assert exc_info.value.task.status.state.value == "failed"

    async def test_rejected_task_raises_a2a_task_rejected_error(self) -> None:
        agent = Agent("specialist", "p", config=TestConfig("never-called"))
        server = A2AServer(agent, url="http://test")
        handler = DefaultRequestHandler(agent_executor=_RejectingExecutor(), task_store=InMemoryTaskStore())
        asgi = server.build_asgi(request_handler=handler)
        transport = httpx.ASGITransport(app=asgi)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            config = A2AConfig("http://test", client_factory=lambda: http)
            client = config.create()
            ctx = ConversationContext(stream=MemoryStream())

            with pytest.raises(A2ATaskRejectedError) as exc_info:
                await client(
                    [ModelRequest([TextInput("hi")])],
                    ctx,
                    tools=[],
                    response_schema=None,
                    serializer=None,  # type: ignore[arg-type]
                )

        assert exc_info.value.task.status.state.value == "rejected"

    async def test_auth_required_task_raises_a2a_auth_required_error(self) -> None:
        # Server signals TaskState.auth_required — the client has no auth flow
        # of its own, so it must raise A2AAuthRequiredError instead of looping
        # waiting for terminal state.
        agent = Agent("specialist", "p", config=TestConfig("never-called"))
        server = A2AServer(agent, url="http://test")
        handler = DefaultRequestHandler(agent_executor=_AuthRequiredExecutor(), task_store=InMemoryTaskStore())
        asgi = server.build_asgi(request_handler=handler)
        transport = httpx.ASGITransport(app=asgi)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            config = A2AConfig("http://test", client_factory=lambda: http)
            client = config.create()
            ctx = ConversationContext(stream=MemoryStream())

            with pytest.raises(A2AAuthRequiredError) as exc_info:
                await client(
                    [ModelRequest([TextInput("hi")])],
                    ctx,
                    tools=[],
                    response_schema=None,
                    serializer=None,  # type: ignore[arg-type]
                )

        assert exc_info.value.task.status.state == TaskState.auth_required


@pytest.mark.asyncio
async def test_orchestrator_can_call_remote_via_as_tool(serve) -> None:
    remote_agent = Agent("remote", "be a helper", config=TestConfig("from-remote"))
    env = serve(remote_agent)

    remote = Agent("remote", config=env.config)

    main_agent = Agent(
        "main",
        "orchestrator",
        config=TestConfig(
            ToolCallEvent(name="task_remote", arguments='{"objective": "say hi"}'),
            "main-final",
        ),
        tools=[remote.as_tool(description="Delegate to remote")],
    )

    reply = await main_agent.ask("Hello")

    assert reply.body == "main-final"
