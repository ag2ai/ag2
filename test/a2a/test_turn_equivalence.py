# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Streaming and polling return the same reply when a server reuses an artifact id across turns.

Every script runs two turns on one task. The first writes artifact ``art`` and requests a client
tool; the continuation, a new turn on the same task, updates ``art`` again and may write a second
artifact ``b``. Updates append, replace with text that extends the content, replace it with other
text, or restate it unchanged, and any artifact may be closed with ``last_chunk`` after its last
update in a turn.

The polling client is shown every intermediate snapshot: after each update the executor waits until
the task store holds it and a ``GetTask`` has read it. So both modes see the same sequence of
contents, and the test can compute the reply and the chunks from the rule alone:

- an artifact's reply is its content past its baseline (its content when the turn began) when it
  extends it, its whole content otherwise;
- the reply joins the non-empty artifact replies in the order the artifacts first appeared on the task;
- a chunk is the reply text not yet sent in this turn, and nothing when the reply no longer starts
  with what was sent, since sent text cannot be retracted.
"""

import asyncio
import json
from dataclasses import dataclass
from enum import Enum

import httpx
import pytest
from a2a.server.context import ServerCallContext
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import Part, Task

from ag2 import Agent
from ag2.a2a import A2AConfig, A2AServer
from ag2.a2a.extension import MIME_TOOL_CALL
from ag2.a2a.mappers.parts import data_part
from ag2.a2a.mappers.tools import call_to_payload
from ag2.events import ModelMessageChunk, ToolCallEvent
from ag2.stream import MemoryStream
from ag2.testing import TestConfig

from ._helpers import EventCollector, ToolRoundTripExecutor

_URL = "http://test"


def step() -> str:
    return "done"


class _Kind(Enum):
    CREATE = "create"
    APPEND = "append"
    EXTEND = "extend"
    OTHER = "other"
    RESTATE = "restate"


_CREATED = {"art": "pre-", "b": "X"}


@dataclass(frozen=True)
class _Op:
    artifact_id: str
    kind: _Kind
    last_chunk: bool = False

    def apply(self, content: str) -> tuple[str, bool, str]:
        """Return the update's text, its ``append`` flag, and the artifact's content after it."""
        tag = self.artifact_id[0]
        if self.kind is _Kind.CREATE:
            return _CREATED[self.artifact_id], False, _CREATED[self.artifact_id]
        if self.kind is _Kind.APPEND:
            return f"+{tag}", True, f"{content}+{tag}"
        if self.kind is _Kind.EXTEND:
            return f"{content}*{tag}", False, f"{content}*{tag}"
        if self.kind is _Kind.OTHER:
            return f"!{tag}", False, f"!{tag}"
        return content, False, content

    @property
    def id(self) -> str:
        return f"{self.artifact_id}.{self.kind.value}{'.last' if self.last_chunk else ''}"


@dataclass(frozen=True)
class _Script:
    first: tuple[_Op, ...]
    second: tuple[_Op, ...]

    @property
    def id(self) -> str:
        return f"{'+'.join(op.id for op in self.first)}|{'+'.join(op.id for op in self.second)}"

    def expected(self) -> tuple[str, list[str]]:
        """The second turn's reply and the chunks of both turns, from the rule alone."""
        content: dict[str, str] = {}
        chunks: list[str] = []
        reply = ""
        for ops in (self.first, self.second):
            baseline = dict(content)
            sent: dict[str, str] = {}
            for op in ops:
                *_, content[op.artifact_id] = op.apply(content.get(op.artifact_id, ""))
                current = _reply(baseline.get(op.artifact_id, ""), content[op.artifact_id])
                previous = sent.get(op.artifact_id, "")
                if current.startswith(previous) and (chunk := current[len(previous) :]):
                    chunks.append(chunk)
                    sent[op.artifact_id] = current
            # Dicts keep insertion order: the order of first appearance on the task.
            reply = "".join(_reply(baseline.get(artifact_id, ""), text) for artifact_id, text in content.items())
        return reply, chunks


def _reply(baseline: str, content: str) -> str:
    return content[len(baseline) :] if content.startswith(baseline) else content


def _closing_variants(ops: tuple[_Op, ...]) -> list[tuple[_Op, ...]]:
    """``ops`` with every combination of ``last_chunk`` on each artifact's last update in the turn."""
    last_index = {op.artifact_id: index for index, op in enumerate(ops)}
    closable = sorted(last_index.values())
    variants = []
    for mask in range(2 ** len(closable)):
        closing = {index for bit, index in enumerate(closable) if mask >> bit & 1}
        variants.append(tuple(_Op(op.artifact_id, op.kind, i in closing) for i, op in enumerate(ops)))
    return variants


def _turns(prefix: tuple[_Op, ...], ids: tuple[str, ...], length: int, seen: frozenset[str]) -> list[tuple[_Op, ...]]:
    """Every sequence of ``length`` updates over ``ids``; an artifact not yet on the task is created first."""
    if length == 0:
        return [prefix]
    sequences = []
    for artifact_id in ids:
        kinds = [_Kind.CREATE] if artifact_id not in seen else [k for k in _Kind if k is not _Kind.CREATE]
        for kind in kinds:
            sequences += _turns((*prefix, _Op(artifact_id, kind)), ids, length - 1, seen | {artifact_id})
    return sequences


def _scripts() -> list[_Script]:
    """Up to 3 updates over 2 turns: ``art`` is created in the first and reused in the second, ``b`` is new."""
    scripts = []
    for first_length, second_length in ((1, 1), (1, 2), (2, 1)):
        for first in _turns((), ("art",), first_length, frozenset()):
            for second in _turns((), ("art", "b"), second_length, frozenset({"art"})):
                scripts += [
                    _Script(first_variant, second_variant)
                    for first_variant in _closing_variants(first)
                    for second_variant in _closing_variants(second)
                ]
    return scripts


_SCRIPTS = _scripts()


class _SavedStore(InMemoryTaskStore):
    """Task store that lets a waiter know when it holds an artifact with a given text."""

    def __init__(self) -> None:
        super().__init__()
        self._changed = asyncio.Condition()
        self._artifacts: dict[str, str] = {}

    async def save(self, task: Task, context: ServerCallContext) -> None:
        await super().save(task, context)
        async with self._changed:
            self._artifacts = {a.artifact_id: "".join(p.text for p in a.parts) for a in task.artifacts}
            self._changed.notify_all()

    async def holds(self, artifact_id: str, text: str) -> None:
        async with self._changed:
            await self._changed.wait_for(lambda: self._artifacts.get(artifact_id) == text)


class _PollGate(httpx.AsyncBaseTransport):
    """In-process transport that tells a waiter once a ``GetTask`` sent after it began waiting has been answered."""

    def __init__(self, transport: httpx.AsyncBaseTransport) -> None:
        self._transport = transport
        self._waiters: list[asyncio.Future[None]] = []

    async def next_poll(self) -> None:
        waiter = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        await waiter

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        waiters: list[asyncio.Future[None]] = []
        if json.loads(request.content or b"{}").get("method") == "GetTask":
            waiters, self._waiters = self._waiters, []
        response = await self._transport.handle_async_request(request)
        await response.aread()
        for waiter in waiters:
            waiter.set_result(None)
        return response


class _ScriptedTurns:
    """Plays a script's turns; when ``gate`` is set, waits for the polling client to read each update."""

    def __init__(self, script: _Script, store: _SavedStore) -> None:
        self._script = script
        self._store = store
        self._content: dict[str, str] = {}
        self.gate: _PollGate | None = None

    async def first_turn(self, updater: TaskUpdater) -> None:
        await self._play(updater, self._script.first)
        call = data_part(call_to_payload(ToolCallEvent(name="step", arguments="{}")), media_type=MIME_TOOL_CALL)
        await updater.add_artifact(parts=[call], artifact_id="call", last_chunk=True)
        await updater.requires_input()

    async def continuation(self, updater: TaskUpdater) -> None:
        await self._play(updater, self._script.second)
        await updater.complete()

    async def _play(self, updater: TaskUpdater, ops: tuple[_Op, ...]) -> None:
        for op in ops:
            text, append, content = op.apply(self._content.get(op.artifact_id, ""))
            self._content[op.artifact_id] = content
            await updater.add_artifact(
                parts=[Part(text=text)],
                artifact_id=op.artifact_id,
                append=append or None,
                last_chunk=op.last_chunk or None,
            )
            if self.gate is not None:
                await self._store.holds(op.artifact_id, content)
                await self.gate.next_poll()


async def _ask(script: _Script, *, streaming: bool) -> tuple[str | None, list[str]]:
    store = _SavedStore()
    turns = _ScriptedTurns(script, store)
    executor = ToolRoundTripExecutor(turns.first_turn, turns.continuation)
    server = A2AServer(Agent("server-stub", config=TestConfig("unused")), executor=executor, task_store=store)
    gate = _PollGate(httpx.ASGITransport(app=server.build_jsonrpc(url=_URL)))
    if not streaming:
        turns.gate = gate
    config = A2AConfig(
        card_url=_URL,
        httpx_client_factory=lambda: httpx.AsyncClient(transport=gate, base_url=_URL, timeout=30.0),
        streaming=streaming,
        polling_interval=0.001,
    )
    client = Agent("client", config=config, tools=[step])
    stream = MemoryStream()
    collector = EventCollector()
    stream.where(ModelMessageChunk).subscribe(collector.on_event)

    reply = await asyncio.wait_for(client.ask("go", stream=stream), timeout=10.0)

    return reply.body, collector.chunks


@pytest.mark.asyncio
@pytest.mark.parametrize("script", _SCRIPTS, ids=[script.id for script in _SCRIPTS])
async def test_streaming_and_polling_follow_the_reply_rule(script: _Script) -> None:
    expected_reply, expected_chunks = script.expected()

    streamed = await _ask(script, streaming=True)
    polled = await _ask(script, streaming=False)

    assert polled == streamed
    assert streamed == (expected_reply or None, expected_chunks)
