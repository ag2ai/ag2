# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

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

from ag2 import Agent, Context
from ag2.a2a import A2AConfig, build_card
from ag2.a2a.errors import A2AReconnectError
from ag2.a2a.extension import MIME_TOOL_CALL
from ag2.a2a.mappers.parts import data_part
from ag2.a2a.mappers.tools import call_to_payload
from ag2.events import BaseEvent, ModelResponse, ToolCallEvent
from ag2.middleware import BaseMiddleware, LLMCall
from ag2.middleware.base import MiddlewareFactory
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


def _chunk(artifact_id: str, text: str, *, append: bool = True) -> StreamResponse:
    """Open (not last) artifact chunk."""
    return StreamResponse(
        artifact_update=TaskArtifactUpdateEvent(
            task_id=TASK_ID,
            context_id=CONTEXT_ID,
            artifact=Artifact(artifact_id=artifact_id, parts=[Part(text=text)]),
            append=append,
            last_chunk=False,
        ),
    )


def _closing_chunk(artifact_id: str, text: str) -> StreamResponse:
    """Appended chunk that closes the artifact."""
    return StreamResponse(
        artifact_update=TaskArtifactUpdateEvent(
            task_id=TASK_ID,
            context_id=CONTEXT_ID,
            artifact=Artifact(artifact_id=artifact_id, parts=[Part(text=text)]),
            append=True,
            last_chunk=True,
        ),
    )


def _snapshot(artifact_id: str, text: str) -> StreamResponse:
    """Task as it stands on resubscribe: the artifact with everything written so far."""
    return StreamResponse(
        task=Task(
            id=TASK_ID,
            context_id=CONTEXT_ID,
            status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
            artifacts=[Artifact(artifact_id=artifact_id, parts=[Part(text=text)])],
        ),
    )


def _snapshot_with_call(
    artifact_id: str,
    tool: str,
    *,
    state: TaskState = TaskState.TASK_STATE_INPUT_REQUIRED,
    call_id: str | None = None,
) -> StreamResponse:
    """Task as it stands on resubscribe after the server wrote a client tool call: asking for input, or about to."""
    event = ToolCallEvent(name=tool, arguments="{}") if call_id is None else ToolCallEvent(name=tool, id=call_id)
    call = data_part(call_to_payload(event), media_type=MIME_TOOL_CALL)
    return StreamResponse(
        task=Task(
            id=TASK_ID,
            context_id=CONTEXT_ID,
            status=TaskStatus(state=state),
            artifacts=[Artifact(artifact_id=artifact_id, parts=[call])],
        ),
    )


def _closed_call(artifact_id: str, call_id: str, *, append: bool) -> StreamResponse:
    """Client tool call written to ``artifact_id`` and closed with ``last_chunk``."""
    call = data_part(call_to_payload(ToolCallEvent(name="lookup", id=call_id)), media_type=MIME_TOOL_CALL)
    return StreamResponse(
        artifact_update=TaskArtifactUpdateEvent(
            task_id=TASK_ID,
            context_id=CONTEXT_ID,
            artifact=Artifact(artifact_id=artifact_id, parts=[call]),
            append=append,
            last_chunk=True,
        ),
    )


def _input_required() -> StreamResponse:
    return StreamResponse(
        status_update=TaskStatusUpdateEvent(
            task_id=TASK_ID,
            context_id=CONTEXT_ID,
            status=TaskStatus(state=TaskState.TASK_STATE_INPUT_REQUIRED),
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

    ``finished`` models a task that ended while the stream was down: a
    resubscribe is refused as ``UnsupportedOperation`` (spec §3.1.6) and
    ``GetTask`` returns it.

    ``continuations`` answer the ``SendStreamingMessage`` calls after the first,
    the turns a client sends once it has run a tool, in order; the last entry
    repeats.

    ``get_task_fails`` makes every ``GetTask`` answer with a JSON-RPC internal error.

    ``methods`` records the JSON-RPC method of every request in order —
    the reconnect behaviour under test is exactly *which* calls happen.
    """

    send: _Leg
    resubscribes: Sequence[_Leg] = ()
    continuations: Sequence[_Leg] = ()
    finished: Task | None = None
    get_task_fails: bool = False
    methods: list[str] = field(default_factory=list)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(await request.aread())
        method = payload["method"]
        self.methods.append(method)
        if self.finished is not None and method == "SubscribeToTask":
            error = {"code": -32004, "message": f"Task {TASK_ID} is in terminal state: 3"}
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "error": error})
        if self.get_task_fails and method == "GetTask":
            error = {"code": -32603, "message": "store unavailable"}
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "error": error})
        if self.finished is not None and method == "GetTask":
            result = json_format.MessageToDict(self.finished)
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result})
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_DroppingByteStream(self._leg_for(method)),
        )

    def _leg_for(self, method: str) -> _Leg:
        if method == "SendStreamingMessage":
            sends = self.methods.count(method)
            if self.continuations and sends > 1:
                return self.continuations[min(sends - 2, len(self.continuations) - 1)]
            return self.send
        assert method == "SubscribeToTask", f"unscripted JSON-RPC method {method!r}"
        assert self.resubscribes, "script has no resubscribe leg"
        index = min(self.subscribe_count - 1, len(self.resubscribes) - 1)
        return self.resubscribes[index]

    @property
    def subscribe_count(self) -> int:
        return self.methods.count("SubscribeToTask")


def lookup() -> str:
    return "result"


class _RetryEveryFailure(BaseMiddleware):
    """Retries a failed model call once with the same messages, even after it published output."""

    async def on_llm_call(self, call_next: LLMCall, events: Sequence[BaseEvent], context: Context) -> ModelResponse:
        try:
            return await call_next(events, context)
        except A2AReconnectError:
            return await call_next(events, context)


def _client(
    server: _ScriptedServer,
    *,
    max_reconnects: int = 3,
    tools: Sequence[Callable[..., Any]] = (),
    middleware: Sequence[MiddlewareFactory] = (),
) -> Agent:
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
        tools=list(tools),
        middleware=list(middleware),
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
        assert server.methods == ["SendStreamingMessage", "SubscribeToTask"]

    async def test_text_appended_after_the_closing_chunk_is_read_back_from_the_task(self) -> None:
        # Spec §4.2.2: ``lastChunk`` marks the final chunk, so a later update for that artifact is
        # either a replay or a server breaking that rule. The stored task decides, as for a polling client.
        server = _ScriptedServer(
            send=_Leg([_task(), _artifact("art-1", "x"), _closing_chunk("art-1", "y"), _completed()]),
            finished=Task(
                id=TASK_ID,
                context_id=CONTEXT_ID,
                status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
                artifacts=[Artifact(artifact_id="art-1", parts=[Part(text="xy")])],
            ),
        )

        reply = await _client(server).ask("ping")

        assert reply.body == "xy"
        assert server.methods == ["SendStreamingMessage", "GetTask"]

    async def test_reply_survives_a_failed_read_back_of_a_closed_artifact(self) -> None:
        # The stream already settled the turn; the read-back only reconciles, so its failure keeps
        # what the stream delivered instead of failing the turn.
        server = _ScriptedServer(
            send=_Leg([_task(), _artifact("art-1", "x"), _closing_chunk("art-1", "y"), _completed()]),
            get_task_fails=True,
        )

        reply = await _client(server).ask("ping")

        assert reply.body == "x"
        assert server.methods == ["SendStreamingMessage", "GetTask"]

    async def test_tool_call_added_after_the_closing_chunk_is_read_back_from_the_task(self) -> None:
        # A polling client sees both calls in the stored task, so a streaming client must run both.
        ran: list[str] = []

        def lookup() -> str:
            ran.append("lookup")
            return "result"

        stored_calls = [
            data_part(call_to_payload(ToolCallEvent(name="lookup", id=call_id)), media_type=MIME_TOOL_CALL)
            for call_id in ("c1", "c2")
        ]
        server = _ScriptedServer(
            send=_Leg([
                _task(),
                _closed_call("calls", "c1", append=False),
                _closed_call("calls", "c2", append=True),
                _input_required(),
            ]),
            finished=Task(
                id=TASK_ID,
                context_id=CONTEXT_ID,
                status=TaskStatus(state=TaskState.TASK_STATE_INPUT_REQUIRED),
                artifacts=[Artifact(artifact_id="calls", parts=stored_calls)],
            ),
            continuations=[_Leg([_artifact("art-1", "done"), _completed()])],
        )

        reply = await _client(server, tools=[lookup]).ask("ping")

        assert reply.body == "done"
        assert ran == ["lookup", "lookup"]
        assert server.methods == ["SendStreamingMessage", "GetTask", "SendStreamingMessage"]

    async def test_task_finished_while_disconnected_is_read_back_instead_of_failing(self) -> None:
        server = _ScriptedServer(
            send=_Leg([_task(), _chunk("art-1", "he", append=False)], drop=True),
            finished=Task(
                id=TASK_ID,
                context_id=CONTEXT_ID,
                status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
                artifacts=[Artifact(artifact_id="art-1", parts=[Part(text="hello world")])],
            ),
        )

        reply = await _client(server).ask("ping")

        assert reply.body == "hello world"
        assert server.methods == ["SendStreamingMessage", "SubscribeToTask", "GetTask"]

    async def test_text_written_while_disconnected_is_recovered_from_the_snapshot(self) -> None:
        server = _ScriptedServer(
            send=_Leg([_task(), _chunk("art-1", "he", append=False)], drop=True),
            resubscribes=[_Leg([_snapshot("art-1", "hello wor"), _chunk("art-1", "ld"), _completed()])],
        )

        reply = await _client(server).ask("ping")

        assert reply.body == "hello world"

    async def test_tool_call_requested_while_disconnected_is_run_from_the_snapshot(self) -> None:
        server = _ScriptedServer(
            send=_Leg([_task(), _chunk("art-1", "he", append=False)], drop=True),
            resubscribes=[_Leg([_snapshot_with_call("calls", "lookup")])],
            continuations=[_Leg([_artifact("art-2", "found"), _completed()])],
        )

        reply = await _client(server, tools=[lookup]).ask("ping")

        assert reply.body == "found"
        assert server.methods == ["SendStreamingMessage", "SubscribeToTask", "SendStreamingMessage"]

    async def test_tool_call_written_while_disconnected_before_input_required_is_run(self) -> None:
        # The snapshot still shows the task working; the live INPUT_REQUIRED that follows carries no artifacts.
        server = _ScriptedServer(
            send=_Leg([_task(), _chunk("art-1", "he", append=False)], drop=True),
            resubscribes=[
                _Leg([_snapshot_with_call("calls", "lookup", state=TaskState.TASK_STATE_WORKING), _input_required()])
            ],
            continuations=[_Leg([_artifact("art-2", "found"), _completed()])],
        )

        reply = await _client(server, tools=[lookup]).ask("ping")

        assert reply.body == "found"
        assert server.methods == ["SendStreamingMessage", "SubscribeToTask", "SendStreamingMessage"]

    @pytest.mark.parametrize(
        ("snapshot", "expected"),
        [
            pytest.param("pre-end", "end", id="unchanged"),
            pytest.param("pre-end!", "end!", id="written-while-disconnected"),
        ],
    )
    async def test_snapshot_of_an_artifact_reused_across_a_tool_round_trip_keeps_only_this_turn(
        self, snapshot: str, expected: str
    ) -> None:
        # The continuation appends to the artifact the tool turn opened, so a snapshot restates
        # the earlier turn's text too; that text was consumed by the tool turn, not this reply.
        server = _ScriptedServer(
            send=_Leg([_task(), _chunk("art-1", "pre-", append=False), _snapshot_with_call("calls", "lookup")]),
            continuations=[_Leg([_chunk("art-1", "end")], drop=True)],
            resubscribes=[_Leg([_snapshot("art-1", snapshot), _completed()])],
        )

        reply = await _client(server, tools=[lookup]).ask("ping")

        assert reply.body == expected

    async def test_retried_tool_turn_runs_the_call_its_failed_attempt_received(self) -> None:
        # The server asks the retried turn for the call the failed attempt already received.
        server = _ScriptedServer(
            send=_Leg([_task(), _snapshot_with_call("calls", "lookup")]),
            continuations=[
                _Leg(
                    [_snapshot_with_call("calls-2", "lookup", call_id="again", state=TaskState.TASK_STATE_WORKING)],
                    drop=True,
                ),
                _Leg([_snapshot_with_call("calls-2", "lookup", call_id="again")]),
                _Leg([_artifact("art-2", "found"), _completed()]),
            ],
        )

        client = _client(server, max_reconnects=0, tools=[lookup], middleware=[_RetryEveryFailure])
        reply = await client.ask("ping")

        assert reply.body == "found"
        assert server.methods == ["SendStreamingMessage"] * 4

    async def test_retried_tool_turn_keeps_the_text_its_failed_attempt_received(self) -> None:
        server = _ScriptedServer(
            send=_Leg([_task(), _chunk("art-1", "pre-", append=False), _snapshot_with_call("calls", "lookup")]),
            continuations=[
                _Leg([_chunk("art-1", "mid")], drop=True),
                _Leg([_snapshot("art-1", "pre-mid-end"), _completed()]),
            ],
        )

        client = _client(server, max_reconnects=0, tools=[lookup], middleware=[_RetryEveryFailure])
        reply = await client.ask("ping")

        assert reply.body == "mid-end"

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
