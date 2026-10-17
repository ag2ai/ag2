# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from functools import partial

import httpx
import pytest
from a2a.server.agent_execution import AgentExecutor as A2AAgentExecutorBase
from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryPushNotificationConfigStore, InMemoryTaskStore
from a2a.types import AgentCard, AgentExtension, Task, TaskState, TaskStatus

from ag2 import Agent
from ag2.a2a import A2AConfig, A2AServer, build_card
from ag2.a2a.errors import A2AExtensionNotSupportedError
from ag2.a2a.executor import AgentExecutor
from ag2.a2a.extension import EXTENSION_URI
from ag2.a2a.push import list_push_notification_configs
from ag2.a2a.tasks import list_tasks
from ag2.a2a.testing import make_test_client_factory, make_test_rest_client_factory, pick_free_port
from ag2.hitl import HumanHook
from ag2.testing import TestConfig

from ._helpers import PromptThenAckExecutor

ADMIN_TASK_ID = "task-1"
CUSTOM_URI = "urn:example:custom:v1"
OTHER_URI = "urn:example:other:v1"
URL = "http://test"
REPLY = "ok"


@dataclass(slots=True)
class _Activation:
    """How one incoming request declared its active extensions.

    The spec allows two channels and AG2 uses both, so they are recorded
    separately: ``header`` comes from the ``A2A-Extensions`` header (gRPC
    metadata on that transport) and is a set, ``message`` comes from the
    ordered ``Message.extensions`` field.
    """

    header: set[str]
    message: list[str]


@dataclass(slots=True)
class _ActivationRecorder(A2AAgentExecutorBase):
    """Records each request's activation, then delegates to ``inner``.

    Wrapping instead of reimplementing keeps the recording orthogonal to
    the task choreography — the same recorder sits in front of AG2's own
    one-shot executor or of a two-leg HITL one.
    """

    inner: A2AAgentExecutorBase
    seen: list[_Activation] = field(default_factory=list)

    async def execute(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        message = request_context.message
        assert message is not None
        self.seen.append(
            _Activation(
                header=set(request_context.requested_extensions),
                message=list(message.extensions),
            )
        )
        await self.inner.execute(request_context, event_queue)

    async def cancel(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        await self.inner.cancel(request_context, event_queue)


def _server_agent() -> Agent:
    return Agent("ext-server", config=TestConfig(REPLY))


def _requiring(card: AgentCard, uri: str) -> AgentCard:
    """Flip an already-declared extension on ``card`` to ``required=True``.

    Looked up by URI rather than by position, so a reordering inside
    ``build_card`` cannot silently retarget this at a different extension.
    """
    next(ext for ext in card.capabilities.extensions if ext.uri == uri).required = True
    return card


def _pair(
    *,
    declared: Sequence[AgentExtension] = (),
    activate: Sequence[str] = (),
    streaming: bool = True,
    tools: Sequence[Callable[..., object]] = (),
    hitl_hook: HumanHook | None = None,
    inner: Callable[[Agent], A2AAgentExecutorBase] = AgentExecutor,
    card: AgentCard | None = None,
) -> tuple[Agent, _ActivationRecorder]:
    """A client agent talking to a server that records how each request activated extensions.

    ``declared`` lands in the served card's ``capabilities.extensions``;
    ``activate`` lands in ``A2AConfig.extensions``. ``inner`` picks the
    server-side choreography — the default AG2 executor answers in a
    single leg, while ``PromptThenAckExecutor`` forces a HITL
    continuation so activation can be checked on both legs. ``card``
    overrides the served card outright, for the cases ``declared`` can't
    express.
    """
    agent = _server_agent()
    recorder = _ActivationRecorder(inner(agent))
    server = A2AServer(agent, executor=recorder)
    served = card if card is not None else build_card(agent, url=URL, extensions=list(declared))
    client = Agent(
        "client",
        config=A2AConfig(
            card_url=URL,
            httpx_client_factory=make_test_client_factory(server, url=URL, card=served),
            streaming=streaming,
            extensions=list(activate),
        ),
        tools=list(tools),
        hitl_hook=hitl_hook,
    )
    return client, recorder


@dataclass(slots=True)
class _HeaderProbe(httpx.AsyncBaseTransport):
    """Records the ``A2A-Extensions`` header of every outgoing request.

    Admin helpers never reach an ``AgentExecutor``, so the recorder used
    for ``ask`` can't see them; this sits under the httpx client instead.
    """

    inner: httpx.AsyncBaseTransport
    seen: list[tuple[str, str | None]] = field(default_factory=list)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.seen.append((request.url.path, request.headers.get("a2a-extensions")))
        return await self.inner.handle_async_request(request)

    @property
    def rpc_headers(self) -> list[str | None]:
        """Headers of the RPCs, dropping the card fetch."""
        return [header for path, header in self.seen if not path.startswith("/.well-known/")]


async def _admin_setup(
    *,
    declared: Sequence[AgentExtension] = (),
    activate: Sequence[str] = (),
    card: AgentCard | None = None,
) -> tuple[A2AConfig, _HeaderProbe]:
    """An ``A2AConfig`` for the admin helpers, plus a probe on its wire.

    The task store is seeded directly, not via an ``ask``: the refusal
    cases must never reach the server at all.
    """
    agent = _server_agent()
    task_store = InMemoryTaskStore()
    await task_store.save(
        Task(id=ADMIN_TASK_ID, context_id="ctx-1", status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED)),
        ServerCallContext(),
    )
    server = A2AServer(
        agent,
        task_store=task_store,
        push_config_store=InMemoryPushNotificationConfigStore(),
    )
    served = card if card is not None else build_card(agent, url=URL, extensions=list(declared))
    probe = _HeaderProbe(httpx.ASGITransport(app=server.build_jsonrpc(url=URL, card=served)))
    config = A2AConfig(
        card_url=URL,
        httpx_client_factory=lambda: httpx.AsyncClient(transport=probe, base_url=URL, timeout=30.0),
        extensions=list(activate),
    )
    return config, probe


class TestCardDeclaration:
    def test_user_extensions_are_declared_after_the_native_one(self) -> None:
        card = build_card(
            _server_agent(),
            url=URL,
            extensions=[AgentExtension(uri=CUSTOM_URI, description="custom", required=False)],
        )

        assert [ext.uri for ext in card.capabilities.extensions] == [EXTENSION_URI, CUSTOM_URI]

    def test_duplicate_uris_are_rejected(self) -> None:
        with pytest.raises(ValueError, match=CUSTOM_URI):
            build_card(
                _server_agent(),
                url=URL,
                extensions=[AgentExtension(uri=CUSTOM_URI), AgentExtension(uri=CUSTOM_URI)],
            )

    def test_redeclaring_the_native_extension_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=EXTENSION_URI):
            build_card(_server_agent(), url=URL, extensions=[AgentExtension(uri=EXTENSION_URI)])


@pytest.mark.asyncio
class TestCardReconciliation:
    """``A2AConfig.extensions`` versus the served card, checked before the first request."""

    async def test_activating_an_unadvertised_extension_is_refused(self) -> None:
        client, recorder = _pair(activate=[CUSTOM_URI])

        with pytest.raises(A2AExtensionNotSupportedError) as exc_info:
            await client.ask("hi")

        assert exc_info.value.uris == [CUSTOM_URI]
        assert exc_info.value.url == URL
        assert recorder.seen == [], "refusal must happen before anything reaches the server"

    async def test_a_required_extension_left_inactive_is_refused(self) -> None:
        client, _ = _pair(declared=[AgentExtension(uri=CUSTOM_URI, required=True)])

        with pytest.raises(A2AExtensionNotSupportedError) as exc_info:
            await client.ask("hi")

        assert exc_info.value.uris == [CUSTOM_URI]

    async def test_a_required_extension_connects_once_activated(self) -> None:
        client, _ = _pair(
            declared=[AgentExtension(uri=CUSTOM_URI, required=True)],
            activate=[CUSTOM_URI],
        )

        reply = await client.ask("hi")

        assert reply.body == REPLY

    async def test_a_required_native_extension_needs_no_activation(self) -> None:
        # A third-party server may mark ``urn:ag2:client-tools:v1`` required.
        # AG2 implements it natively, so there is nothing for the user to
        # activate and the connection must go through anyway.
        client, _ = _pair(card=_requiring(build_card(_server_agent(), url=URL), EXTENSION_URI))

        reply = await client.ask("hi")

        assert reply.body == REPLY


@pytest.mark.asyncio
class TestActivationOnTheWire:
    async def test_activation_rides_both_the_header_and_the_message(self) -> None:
        client, recorder = _pair(declared=[AgentExtension(uri=CUSTOM_URI)], activate=[CUSTOM_URI])

        reply = await client.ask("hi")

        assert reply.body == REPLY
        assert recorder.seen == [_Activation(header={CUSTOM_URI}, message=[CUSTOM_URI])]

    async def test_without_activation_neither_channel_carries_a_uri(self) -> None:
        client, recorder = _pair(declared=[AgentExtension(uri=CUSTOM_URI)])

        reply = await client.ask("hi")

        assert reply.body == REPLY
        assert recorder.seen == [_Activation(header=set(), message=[])]

    async def test_duplicate_activations_collapse_and_keep_user_order(self) -> None:
        client, recorder = _pair(
            declared=[AgentExtension(uri=CUSTOM_URI), AgentExtension(uri=OTHER_URI)],
            activate=[OTHER_URI, CUSTOM_URI, OTHER_URI],
        )

        await client.ask("hi")

        # ``Message.extensions`` is the ordered channel and preserves the
        # order the user configured; the header is a set server-side, so it
        # only witnesses the dedup.
        assert recorder.seen == [_Activation(header={OTHER_URI, CUSTOM_URI}, message=[OTHER_URI, CUSTOM_URI])]

    async def test_activating_the_native_extension_does_not_duplicate_it(self) -> None:
        # Client tools already put ``EXTENSION_URI`` on ``Message.extensions``;
        # a user who also names it in ``extensions`` must not get it twice.
        def look_up(city: str) -> str:
            return f"sunny in {city}"

        client, recorder = _pair(activate=[EXTENSION_URI], tools=[look_up])

        await client.ask("hi")

        assert recorder.seen == [_Activation(header={EXTENSION_URI}, message=[EXTENSION_URI])]

    async def test_activating_the_native_extension_without_tools_rides_both_channels(self) -> None:
        # No tools and no in-flight task, so AG2 doesn't advertise
        # ``EXTENSION_URI`` itself; the explicit activation still must.
        client, recorder = _pair(activate=[EXTENSION_URI])

        await client.ask("hi")

        assert recorder.seen == [_Activation(header={EXTENSION_URI}, message=[EXTENSION_URI])]

    @pytest.mark.parametrize("streaming", [True, False])
    async def test_activation_survives_a_continuation_leg(self, streaming: bool) -> None:
        async def hitl_hook() -> str:
            return "more input"

        client, recorder = _pair(
            declared=[AgentExtension(uri=CUSTOM_URI)],
            activate=[CUSTOM_URI],
            streaming=streaming,
            inner=lambda _: PromptThenAckExecutor("more?"),
            hitl_hook=hitl_hook,
        )

        reply = await client.ask("hi")

        assert reply.body == "echo: more input", "the HITL round-trip has to actually complete"
        assert recorder.seen == [
            _Activation(header={CUSTOM_URI}, message=[CUSTOM_URI]),
            _Activation(header={CUSTOM_URI}, message=[CUSTOM_URI]),
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call",
    [
        pytest.param(list_tasks, id="list_tasks"),
        pytest.param(
            partial(list_push_notification_configs, task_id=ADMIN_TASK_ID),
            id="list_push_notification_configs",
        ),
    ],
)
class TestActivationOnAdminHelpers:
    """``tasks`` / ``push`` connect on their own from the same ``A2AConfig``.

    Activation is per-request, so it has to reach them too. One read-only
    helper per module stands in for every RPC issued inside ``open_session``.
    """

    async def test_activation_reaches_the_helper(self, call: Callable[[A2AConfig], Awaitable[object]]) -> None:
        config, probe = await _admin_setup(declared=[AgentExtension(uri=CUSTOM_URI)], activate=[CUSTOM_URI])

        await call(config)

        assert probe.rpc_headers == [CUSTOM_URI]

    async def test_no_activation_leaves_the_header_off(self, call: Callable[[A2AConfig], Awaitable[object]]) -> None:
        config, probe = await _admin_setup(declared=[AgentExtension(uri=CUSTOM_URI)])

        await call(config)

        assert probe.rpc_headers == [None]

    async def test_activating_an_unadvertised_extension_is_refused(
        self, call: Callable[[A2AConfig], Awaitable[object]]
    ) -> None:
        config, probe = await _admin_setup(activate=[CUSTOM_URI])

        with pytest.raises(A2AExtensionNotSupportedError) as exc_info:
            await call(config)

        assert exc_info.value.uris == [CUSTOM_URI]
        assert probe.rpc_headers == [], "refusal must happen before any RPC goes out"

    async def test_a_required_extension_left_inactive_is_refused(
        self, call: Callable[[A2AConfig], Awaitable[object]]
    ) -> None:
        config, _ = await _admin_setup(declared=[AgentExtension(uri=CUSTOM_URI, required=True)])

        with pytest.raises(A2AExtensionNotSupportedError) as exc_info:
            await call(config)

        assert exc_info.value.uris == [CUSTOM_URI]

    async def test_a_required_native_extension_needs_no_activation(
        self, call: Callable[[A2AConfig], Awaitable[object]]
    ) -> None:
        config, probe = await _admin_setup(card=_requiring(build_card(_server_agent(), url=URL), EXTENSION_URI))

        await call(config)

        assert probe.rpc_headers == [None]


@pytest.mark.asyncio
class TestActivationAcrossTransports:
    """JSON-RPC is covered above; these two bindings render activation differently."""

    async def test_activation_rides_the_rest_transport(self) -> None:
        agent = _server_agent()
        recorder = _ActivationRecorder(AgentExecutor(agent))
        server = A2AServer(agent, executor=recorder)
        card = build_card(agent, url=URL, transports=("rest",), extensions=[AgentExtension(uri=CUSTOM_URI)])
        client = Agent(
            "client",
            config=A2AConfig(
                card_url=URL,
                httpx_client_factory=make_test_rest_client_factory(server, url=URL, card=card),
                prefer="rest",
                streaming=False,
                extensions=[CUSTOM_URI],
            ),
        )

        reply = await client.ask("hi")

        assert reply.body == REPLY
        assert recorder.seen == [_Activation(header={CUSTOM_URI}, message=[CUSTOM_URI])]

    async def test_activation_rides_grpc_metadata(self) -> None:
        # gRPC has no headers; the SDK renders the activation as call
        # metadata instead. Needs a real socket — there is no in-process
        # ASGITransport equivalent for gRPC.
        agent = _server_agent()
        recorder = _ActivationRecorder(AgentExecutor(agent))
        server = A2AServer(agent, executor=recorder)
        grpc_url = f"127.0.0.1:{pick_free_port()}"
        card = build_card(
            agent,
            url=grpc_url,
            transports=("grpc",),
            grpc_url=grpc_url,
            extensions=[AgentExtension(uri=CUSTOM_URI)],
        )
        grpc_server = server.build_grpc(bind=grpc_url, grpc_url=grpc_url, card=card)
        await grpc_server.start()

        try:
            client = Agent(
                "client",
                config=A2AConfig(
                    card_url=grpc_url,
                    preset_card=card,
                    prefer="grpc",
                    streaming=False,
                    extensions=[CUSTOM_URI],
                ),
            )

            reply = await client.ask("hi")

            assert reply.body == REPLY
            assert recorder.seen == [_Activation(header={CUSTOM_URI}, message=[CUSTOM_URI])]
        finally:
            await grpc_server.stop(grace=0)
