# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

import httpx
import pytest
from a2a.types import (
    Artifact,
    Part,
    StreamResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from google.protobuf import json_format

from ag2 import Agent
from ag2.a2a import A2AConfig, build_card
from ag2.a2a.errors import A2AReconnectError
from ag2.testing import TestConfig

URL = "http://test"
TASK_ID = "task-1"
CONTEXT_ID = "ctx-1"


def _task() -> StreamResponse:
    """Task snapshot — the event that first tells the client its task id."""
    return StreamResponse(
        task=Task(id=TASK_ID, context_id=CONTEXT_ID, status=TaskStatus(state=TaskState.TASK_STATE_WORKING)),
    )


def _artifact(artifact_id: str, text: str) -> StreamResponse:
    return StreamResponse(
        artifact_update=TaskArtifactUpdateEvent(
            task_id=TASK_ID,
            context_id=CONTEXT_ID,
            artifact=Artifact(artifact_id=artifact_id, parts=[Part(text=text)]),
            append=False,
            last_chunk=True,
        ),
    )


def _completed() -> StreamResponse:
    return StreamResponse(
        status_update=TaskStatusUpdateEvent(
            task_id=TASK_ID,
            context_id=CONTEXT_ID,
            status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
        ),
    )


@dataclass(slots=True)
class _Leg:
    """One scripted streaming response: emit ``events``, then maybe lose the connection."""

    events: Sequence[StreamResponse] = ()
    drop: bool = False


class _DroppingByteStream(httpx.AsyncByteStream):
    """SSE body that yields whole event blocks, then optionally dies mid-stream."""

    def __init__(self, leg: _Leg) -> None:
        self._leg = leg

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for event in self._leg.events:
            envelope = {"jsonrpc": "2.0", "id": "1", "result": json_format.MessageToDict(event)}
            yield f"data: {json.dumps(envelope)}\n\n".encode()
        if self._leg.drop:
            # What a severed connection looks like to httpx. The A2A SDK
            # funnels ``httpx.RequestError`` into ``A2AClientError``, which is
            # the signal ``A2AClient`` reconnects on.
            raise httpx.ReadError("simulated stream drop")


@dataclass
class _ScriptedServer(httpx.AsyncBaseTransport):
    """Serves scripted SSE bodies per JSON-RPC method, over the public httpx seam.

    ``send`` answers the opening ``SendStreamingMessage``. Each following
    ``SubscribeToTask`` consumes the next entry of ``resubscribes``; once
    that runs out the last entry repeats, so "every retry also drops" is a
    one-element script rather than a padded list.

    ``methods`` records the JSON-RPC method of every request in order —
    the reconnect behaviour under test is exactly *which* calls happen.
    """

    send: _Leg
    resubscribes: Sequence[_Leg] = ()
    methods: list[str] = field(default_factory=list)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        method = json.loads(await request.aread())["method"]
        self.methods.append(method)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_DroppingByteStream(self._leg_for(method)),
        )

    def _leg_for(self, method: str) -> _Leg:
        if method == "SendStreamingMessage":
            return self.send
        assert method == "SubscribeToTask", f"unscripted JSON-RPC method {method!r}"
        assert self.resubscribes, "script has no resubscribe leg"
        index = min(self.subscribe_count - 1, len(self.resubscribes) - 1)
        return self.resubscribes[index]

    @property
    def subscribe_count(self) -> int:
        return self.methods.count("SubscribeToTask")


def _client(server: _ScriptedServer, *, max_reconnects: int = 3) -> Agent:
    """Agent whose A2A transport is ``server``, with backoff disabled.

    ``preset_card`` skips card discovery, so the script only has to answer
    the RPC calls — but everything downstream (transport selection, SSE
    parsing, reconnect, artifact dedup) is the real client code path.
    """
    card = build_card(Agent("remote", config=TestConfig("unused")), url=URL)
    return Agent(
        "client",
        config=A2AConfig(
            card_url=URL,
            preset_card=card,
            httpx_client_factory=lambda: httpx.AsyncClient(transport=server, base_url=URL),
            max_reconnects=max_reconnects,
            reconnect_backoff=0.0,
        ),
    )


@pytest.mark.asyncio
class TestStreamingReconnect:
    async def test_resubscribes_and_finishes_the_task(self) -> None:
        server = _ScriptedServer(
            send=_Leg([_task(), _artifact("art-1", "hello")], drop=True),
            resubscribes=[_Leg([_artifact("art-2", " world"), _completed()])],
        )

        reply = await _client(server).ask("ping")

        assert reply.body == "hello world"
        assert server.methods == ["SendStreamingMessage", "SubscribeToTask"]

    async def test_artifact_replayed_on_resubscribe_is_not_counted_twice(self) -> None:
        # A2A spec §3.5.2 lets the server replay artifacts it already sent
        # when a client resubscribes; the text must not double up.
        server = _ScriptedServer(
            send=_Leg([_task(), _artifact("art-1", "hello")], drop=True),
            resubscribes=[_Leg([_artifact("art-1", "hello"), _completed()])],
        )

        reply = await _client(server).ask("ping")

        assert reply.body == "hello"

    async def test_gives_up_after_max_reconnects(self) -> None:
        server = _ScriptedServer(
            send=_Leg([_task(), _artifact("art-1", "hi")], drop=True),
            resubscribes=[_Leg(drop=True)],
        )

        with pytest.raises(A2AReconnectError) as exc_info:
            await _client(server, max_reconnects=2).ask("ping")

        assert exc_info.value.attempts == 2
        # Exactly two retries — one more would mean the budget is off by one,
        # one fewer that a retry was skipped.
        assert server.methods == ["SendStreamingMessage", "SubscribeToTask", "SubscribeToTask"]

    async def test_does_not_resubscribe_before_a_task_id_is_known(self) -> None:
        # The stream died before the server announced a task, so there is no
        # task to resubscribe to — failing immediately beats a doomed retry.
        server = _ScriptedServer(send=_Leg(drop=True))

        with pytest.raises(A2AReconnectError) as exc_info:
            await _client(server).ask("ping")

        assert exc_info.value.attempts == 0
        assert server.methods == ["SendStreamingMessage"]
