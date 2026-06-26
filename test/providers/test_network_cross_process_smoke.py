# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Cross-process network smoke tests with real, mixed-provider LLMs.

A single ``Hub`` runs ``serve_ws`` on a loopback port. Every agent is
its own node: a ``HubClient(WsLink(url))`` with **no in-process hub
reference**, so every control-plane call (register, open a channel,
send, discovery, task ops) crosses the wire as a ``RequestFrame`` RPC
and every delivery arrives as a ``NotifyFrame`` — the genuinely
distributed path the cross-process work added. The agents are backed by
different providers in the same network (Anthropic ``claude-haiku-4-5``,
OpenAI ``gpt-5.4-mini``, Gemini ``gemini-3.5-flash``), mirroring the
common real-world setting where one hub coordinates heterogeneous
agents.

Each adapter family is exercised end to end through the public
``HubClient`` / ``AgentClient`` API — the initiator opens the channel
and sends; the default notify handler folds adapter state locally and
drives each turn over the wire:

* ``consulting`` — 1-question/1-reply auto-close (Anthropic ↔ OpenAI).
* ``conversation`` — free two-party exchange (Anthropic ↔ Gemini).
* ``discussion`` — three-way round-robin, one provider each.
* ``workflow`` — a handoff routed across providers via a transition graph.
* reconnect-by-name — a node drops mid-turn before acking; a fresh node
  attaches under the same name and the unacked delivery replays.
* ``RemoteAgentProxy`` — a tenant federation proxy (LLM-backed) answers
  on behalf of a ``kind="remote_agent"`` participant.

Each test skips unless the API keys for the providers it uses are
present (loaded from ``.env`` at repo root).
"""

import asyncio
import contextlib
import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

pytest.importorskip("websockets")
pytest.importorskip("anthropic")
pytest.importorskip("openai")
pytest.importorskip("google.genai")

from ag2 import Agent
from ag2.config import AnthropicConfig, GeminiConfig, OpenAIConfig
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_PACKET,
    EV_TEXT,
    AuthBlock,
    Envelope,
    Handoff,
    Hub,
    HubClient,
    Passport,
    Resume,
    WsLink,
    serve_ws,
)
from ag2.network.adapters.consulting import CONSULTING_TYPE
from ag2.network.adapters.conversation import CONVERSATION_TYPE
from ag2.network.adapters.discussion import DISCUSSION_TYPE, ORDERING_ROUND_ROBIN
from ag2.network.adapters.workflow import WORKFLOW_TYPE
from ag2.network.envelope import EV_CHANNEL_INVITE, EV_CHANNEL_INVITE_ACK
from ag2.network.transitions import (
    AgentTarget,
    FromSpeaker,
    RevertToInitiatorTarget,
    TerminateTarget,
    ToolCalled,
    Transition,
    TransitionGraph,
)

try:
    from dotenv import load_dotenv

    _REPO_ROOT = Path(__file__).resolve().parents[3]
    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass


ANTHROPIC_MODEL = "claude-haiku-4-5"
OPENAI_MODEL = "gpt-5.4-mini"
GEMINI_MODEL = "gemini-3.5-flash"


def _require(*names: str) -> dict[str, str]:
    """Return the named API keys, or skip if any are missing."""
    values = {name: os.getenv(name) for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        pytest.skip(f"cross-process smoke needs {', '.join(missing)}")
    return {name: value for name, value in values.items() if value}


def _anthropic(keys: dict[str, str]) -> AnthropicConfig:
    return AnthropicConfig(model=ANTHROPIC_MODEL, api_key=keys["ANTHROPIC_API_KEY"], temperature=0)


def _openai(keys: dict[str, str]) -> OpenAIConfig:
    return OpenAIConfig(model=OPENAI_MODEL, api_key=keys["OPENAI_API_KEY"], temperature=0)


def _gemini(keys: dict[str, str]) -> GeminiConfig:
    return GeminiConfig(model=GEMINI_MODEL, api_key=keys["GEMINI_API_KEY"], temperature=0)


@contextlib.asynccontextmanager
async def _network() -> AsyncIterator[tuple[Hub, str, list[HubClient]]]:
    """Start a hub on a loopback ``serve_ws`` port.

    Yields ``(hub, ws_url, clients)``. The hub object is held by the
    test for read-only assertions (``read_wal`` / ``adapter_state``);
    the agents only ever reach it over ``ws_url``. ``clients`` is the
    cleanup registry — append every ``HubClient`` so it is closed on
    exit.
    """
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    clients: list[HubClient] = []
    try:
        async with serve_ws(hub, "127.0.0.1", 0) as server:
            url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            yield hub, url, clients
    finally:
        for hc in clients:
            with contextlib.suppress(Exception):
                await hc.close()
        await hub.close()


async def _join(
    url: str,
    clients: list[HubClient],
    agent: Agent,
    *,
    resume: Resume | None = None,
):
    """Connect a fresh node (its own ``WsLink`` / ``HubClient``) and
    register ``agent`` over the wire. Returns the ``AgentClient``."""
    hc = HubClient(WsLink(url))
    clients.append(hc)
    return await hc.register(agent, Passport(name=agent.name), resume or Resume())


async def _wait_for_text_count(hub: Hub, channel_id: str, expected: int, *, timeout: float = 120.0) -> int:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        wal = await hub.read_wal(channel_id)
        count = sum(1 for e in wal if e.event_type == EV_TEXT)
        if count >= expected:
            return count
        await asyncio.sleep(0.2)
    return sum(1 for e in (await hub.read_wal(channel_id)) if e.event_type == EV_TEXT)


@pytest.mark.anthropic
@pytest.mark.openai
@pytest.mark.asyncio
async def test_consulting_anthropic_to_openai_over_wire() -> None:
    """Consulting 1Q1R across processes and providers: alice (Anthropic)
    opens a consulting channel with bob (OpenAI), asks one question, and
    bob's notify handler answers over the wire before the adapter
    auto-closes."""
    keys = _require("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
    async with _network() as (hub, url, clients):
        alice = await _join(
            url,
            clients,
            Agent(name="alice", prompt="You are alice, a coordinator.", config=_anthropic(keys)),
            resume=Resume(summary="coordinator"),
        )
        await _join(
            url,
            clients,
            Agent(
                name="bob",
                prompt="You are an arithmetic specialist. Reply with just the integer result, no words.",
                config=_openai(keys),
            ),
            resume=Resume(claimed_capabilities=["math"], summary="math specialist"),
        )

        channel = await alice.open(type=CONSULTING_TYPE, target=["bob"], intent="ask bob to compute a small product")
        await channel.send("What is 12 times 11? Reply with just the integer.")

        count = await _wait_for_text_count(hub, channel.channel_id, 2, timeout=90.0)
        assert count == 2, f"expected alice's question + bob's reply, got {count} text envelopes"

        wal = await hub.read_wal(channel.channel_id)
        texts = [e for e in wal if e.event_type == EV_TEXT]
        assert "132" in texts[1].event_data.get("text", ""), (
            f"bob (OpenAI) should answer 132 for 12*11; got {texts[1].event_data.get('text')!r}"
        )


@pytest.mark.anthropic
@pytest.mark.gemini
@pytest.mark.asyncio
async def test_conversation_anthropic_with_gemini_over_wire() -> None:
    """Free two-party conversation across providers: alice (Anthropic)
    and bob (Gemini) exchange at least one turn each over the wire. The
    channel is closed explicitly once both have spoken (a free
    conversation has no auto-terminator)."""
    keys = _require("ANTHROPIC_API_KEY", "GEMINI_API_KEY")
    async with _network() as (hub, url, clients):
        alice = await _join(
            url,
            clients,
            Agent(
                name="alice",
                prompt="You are alice. Keep replies to one short, friendly sentence.",
                config=_anthropic(keys),
            ),
        )
        await _join(
            url,
            clients,
            Agent(
                name="bob",
                prompt="You are bob. Keep replies to one short, friendly sentence.",
                config=_gemini(keys),
            ),
        )

        channel = await alice.open(type=CONVERSATION_TYPE, target=["bob"], intent="quick hello")
        await channel.send("Hi Bob — in one sentence, what's your favorite kind of weather?")

        count = await _wait_for_text_count(hub, channel.channel_id, 2, timeout=90.0)
        assert count >= 2, f"expected at least alice + bob turns, got {count}"

        wal = await hub.read_wal(channel.channel_id)
        speakers = {e.sender_id for e in wal if e.event_type == EV_TEXT}
        assert len(speakers) == 2, f"both providers should contribute a turn; speakers={speakers}"
        await channel.close(reason="smoke_done")


@pytest.mark.anthropic
@pytest.mark.openai
@pytest.mark.gemini
@pytest.mark.asyncio
async def test_discussion_three_providers_round_robin_over_wire() -> None:
    """Three nodes, one provider each (Anthropic, OpenAI, Gemini), take
    round-robin turns in one discussion over the wire. The adapter gates
    turns by ``can_send``, so the speaking order holds regardless of how
    concurrent deliveries interleave across connections."""
    keys = _require("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")
    async with _network() as (hub, url, clients):
        prompt = (
            "You are {name} in a 3-way discussion. When it is your turn, reply "
            "with exactly one short opinion (one sentence) as plain text. Do "
            "not ask questions and do not call any tools."
        )
        alice = await _join(
            url, clients, Agent(name="alice", prompt=prompt.format(name="alice"), config=_anthropic(keys))
        )
        bob = await _join(url, clients, Agent(name="bob", prompt=prompt.format(name="bob"), config=_openai(keys)))
        carol = await _join(url, clients, Agent(name="carol", prompt=prompt.format(name="carol"), config=_gemini(keys)))

        channel = await alice.open(
            type=DISCUSSION_TYPE,
            target=["bob", "carol"],
            knobs={"ordering": ORDERING_ROUND_ROBIN},
            intent="mixed-provider debate",
        )
        await channel.send("Quick debate: should new services default to typed languages?")

        count = await _wait_for_text_count(hub, channel.channel_id, 3, timeout=180.0)
        assert count >= 3, f"expected one turn per provider, got {count}"

        wal = await hub.read_wal(channel.channel_id)
        speakers = [e.sender_id for e in wal if e.event_type == EV_TEXT][:3]
        assert speakers == [alice.agent_id, bob.agent_id, carol.agent_id], (
            f"round-robin order broken across providers/processes; got {speakers}"
        )


@pytest.mark.anthropic
@pytest.mark.openai
@pytest.mark.asyncio
async def test_workflow_handoff_across_providers_over_wire() -> None:
    """Workflow handoff over the wire: a human seeds the graph; triage
    (Anthropic) routes the question to eng (OpenAI) via a handoff tool;
    the transition graph then reverts to the initiator. Verifies the
    routing ``EV_PACKET`` and the receiving provider's notify handler
    both work across processes."""
    keys = _require("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
    async with _network() as (hub, url, clients):
        triage = await _join(
            url,
            clients,
            Agent(
                name="triage",
                prompt=(
                    "You are the triage coordinator. When the user asks an engineering "
                    "question, call the transfer_to_eng tool with a one-line reason. Do "
                    "not answer it yourself."
                ),
                config=_anthropic(keys),
            ),
            resume=Resume(claimed_capabilities=["triage"]),
        )
        eng = await _join(
            url,
            clients,
            Agent(
                name="eng",
                prompt="You are a senior engineer. Answer concisely in one sentence.",
                config=_openai(keys),
            ),
            resume=Resume(claimed_capabilities=["engineering"]),
        )
        user_hc = HubClient(WsLink(url))
        clients.append(user_hc)
        user = await user_hc.register_human(Passport(name="user"))

        eng_agent_id = eng.agent_id

        @triage.agent.tool
        async def transfer_to_eng(reason: str = "") -> Handoff:
            """Transfer the conversation to the engineering specialist."""
            return Handoff(target=eng_agent_id, reason=reason)

        graph = TransitionGraph(
            initial_speaker=user.agent_id,
            transitions=[
                Transition(when=FromSpeaker(user.agent_id), then=AgentTarget(triage.agent_id)),
                Transition(when=ToolCalled("transfer_to_eng"), then=AgentTarget(eng.agent_id)),
                Transition(when=FromSpeaker(eng.agent_id), then=RevertToInitiatorTarget()),
            ],
            default_target=TerminateTarget(reason="triage_done"),
            max_turns=8,
        )

        channel = await triage.open(
            type=WORKFLOW_TYPE,
            target=[user.agent_id, eng.agent_id],
            knobs={"graph": graph.to_dict()},
            intent="triage routes deep questions to engineering",
        )
        await user.send(
            channel.channel_id,
            "What's the practical difference between processes and threads in Python?",
        )

        # triage hands off to eng (an EV_PACKET with routing.kind == handoff).
        async def _handoff_seen() -> bool:
            wal = await hub.read_wal(channel.channel_id)
            return any(
                e.event_type == EV_PACKET and (e.event_data.get("routing", {}) or {}).get("kind") == "handoff"
                for e in wal
            )

        deadline = asyncio.get_event_loop().time() + 90.0
        handed_off = False
        while asyncio.get_event_loop().time() < deadline:
            if await _handoff_seen():
                handed_off = True
                break
            await asyncio.sleep(0.3)
        assert handed_off, "triage (Anthropic) did not hand off to eng within 90s"

        # eng (OpenAI) replies via its notify handler, then the graph
        # reverts to the initiator (triage).
        deadline = asyncio.get_event_loop().time() + 90.0
        eng_replied = False
        while asyncio.get_event_loop().time() < deadline:
            state = hub.adapter_state(channel.channel_id)
            if state is not None and getattr(state, "last_speaker_id", None) == eng.agent_id:
                eng_replied = True
                break
            await asyncio.sleep(0.2)
        assert eng_replied, "eng (OpenAI) did not reply to triage's (Anthropic) handoff within 90s"


@pytest.mark.anthropic
@pytest.mark.openai
@pytest.mark.asyncio
async def test_reconnect_by_name_replays_unacked_turn_over_wire() -> None:
    """A node drops mid-turn before acking, then a fresh node attaches
    under the same name and the hub replays the unacked delivery.

    bob's first connection receives alice's seed but its connection is
    closed while the (slow, real-LLM) turn is still in flight — so the
    seed is never acked and bob never replies. A second ``bob`` node
    ``attach``es over the wire with ``since_envelope_id=""``; the hub
    replays the unacked seed past bob's cursor and the fresh node
    answers."""
    keys = _require("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
    async with _network() as (hub, url, clients):
        alice = await _join(url, clients, Agent(name="alice", prompt="You are alice.", config=_anthropic(keys)))

        # First bob node: connects + auto-acks the invite, but we drop its
        # connection right after the seed is dispatched, before the
        # in-flight LLM turn can post a reply or ack the delivery.
        bob_hc1 = HubClient(WsLink(url))
        clients.append(bob_hc1)
        bob1 = await bob_hc1.register(
            Agent(
                name="bob",
                prompt="You are bob. Reply with just the single integer answer.",
                config=_openai(keys),
            ),
            Passport(name="bob"),
            Resume(),
        )
        bob_id = bob1.agent_id

        channel = await alice.open(type=DISCUSSION_TYPE, target=["bob"], knobs={"ordering": ORDERING_ROUND_ROBIN})
        await channel.send("Bob, what is 7 plus 8? Reply with just the integer.")
        # Let the seed reach bob1 and start his (multi-second) turn, then
        # crash the connection before it can ack.
        await asyncio.sleep(0.3)
        await bob_hc1.close()

        # Only alice's seed is in the WAL; bob never replied.
        before = await hub.read_wal(channel.channel_id)
        assert sum(1 for e in before if e.event_type == EV_TEXT) == 1, "bob should not have replied before reconnect"

        # Fresh bob node attaches under the same name and asks for replay.
        bob_hc2 = HubClient(WsLink(url))
        clients.append(bob_hc2)
        bob2 = await bob_hc2.attach(
            Agent(
                name="bob",
                prompt="You are bob. Reply with just the single integer answer.",
                config=_openai(keys),
            ),
            name="bob",
            passport=Passport(name="bob"),
            resume=Resume(),
            since_envelope_id="",
        )
        assert bob2.agent_id == bob_id, "attach should re-bind the same identity"

        count = await _wait_for_text_count(hub, channel.channel_id, 2, timeout=90.0)
        assert count == 2, f"replay should let the fresh bob answer; got {count} text envelopes"

        wal = await hub.read_wal(channel.channel_id)
        bob_texts = [e for e in wal if e.event_type == EV_TEXT and e.sender_id == bob_id]
        assert len(bob_texts) == 1, f"bob should reply exactly once after replay, got {len(bob_texts)}"
        assert "15" in bob_texts[0].event_data.get("text", ""), (
            f"bob (OpenAI) should answer 15 for 7+8; got {bob_texts[0].event_data.get('text')!r}"
        )


class _LLMRemoteProxy:
    """Tenant ``RemoteAgentProxy`` standing in for a federated peer.

    Bridges envelopes the hub addresses to a ``kind="remote_agent"``
    passport: it acks invites on the remote's behalf, runs a real
    (Gemini) agent on inbound text, and reposts the reply into the
    channel via ``Hub.post_envelope`` — the same shape an A2A or gRPC
    bridge would take, with a local LLM standing in for the remote
    endpoint so the federation seam is exercised end to end.
    """

    scheme = "rpc"

    def __init__(self, hub: Hub, agent: Agent, agent_id: str) -> None:
        self._hub = hub
        self._agent = agent
        self._agent_id = agent_id
        self.dispatched = 0

    async def dispatch(self, envelope: Envelope, recipient: Passport) -> None:
        if envelope.event_type == EV_CHANNEL_INVITE:
            await self._hub.post_envelope(
                Envelope(
                    channel_id=envelope.channel_id,
                    sender_id=self._agent_id,
                    audience=None,
                    event_type=EV_CHANNEL_INVITE_ACK,
                    event_data={"channel_id": envelope.channel_id},
                    causation_id=envelope.envelope_id,
                )
            )
            return
        if envelope.event_type != EV_TEXT:
            return
        reply = await self._agent.ask(envelope.event_data.get("text", ""))
        await self._hub.post_envelope(
            Envelope(
                channel_id=envelope.channel_id,
                sender_id=self._agent_id,
                audience=[envelope.sender_id],
                event_type=EV_TEXT,
                event_data={"text": reply.body or ""},
                causation_id=envelope.envelope_id,
            )
        )
        self.dispatched += 1

    async def close(self) -> None:
        return None


@pytest.mark.anthropic
@pytest.mark.gemini
@pytest.mark.asyncio
async def test_remote_agent_proxy_federation_over_wire() -> None:
    """The ``RemoteAgentProxy`` federation seam: alice (Anthropic, a
    remote ``WsLink`` node) consults a ``kind="remote_agent"`` peer whose
    passport routes to a tenant proxy. The proxy answers with a real
    Gemini turn and reposts it; alice sees the reply over the wire as a
    normal consulting answer."""
    keys = _require("ANTHROPIC_API_KEY", "GEMINI_API_KEY")
    async with _network() as (hub, url, clients):
        # Register the remote-agent identity + its proxy directly on the
        # hub (this is the federation operator's side of the seam).
        echo = await hub.register_identity(
            Passport(name="echo", kind="remote_agent", auth=AuthBlock(scheme="rpc", claim={})),
            Resume(claimed_capabilities=["math"]),
        )
        proxy = _LLMRemoteProxy(
            hub,
            Agent(
                name="echo-llm",
                prompt="You are an arithmetic specialist. Reply with just the integer result.",
                config=_gemini(keys),
            ),
            echo.agent_id,
        )
        hub.register_remote_proxy(proxy)

        alice = await _join(url, clients, Agent(name="alice", prompt="You are alice.", config=_anthropic(keys)))

        channel = await alice.open(
            type=CONSULTING_TYPE, target=["echo"], intent="delegate a computation to a remote peer"
        )
        await channel.send("What is 9 times 13? Reply with just the integer.")

        count = await _wait_for_text_count(hub, channel.channel_id, 2, timeout=90.0)
        assert count == 2, f"expected alice's question + the remote's reply, got {count}"
        assert proxy.dispatched >= 1, "proxy.dispatch should have fired for the remote recipient"

        wal = await hub.read_wal(channel.channel_id)
        remote_texts = [e for e in wal if e.event_type == EV_TEXT and e.sender_id == echo.agent_id]
        assert remote_texts and "117" in remote_texts[0].event_data.get("text", ""), (
            f"remote peer (Gemini via proxy) should answer 117 for 9*13; got "
            f"{[e.event_data.get('text') for e in remote_texts]}"
        )
