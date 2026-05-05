# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Network tool surface smoke tests against a real OpenAI model.

Mirrors ``test/beta/providers/anthropic/test_network_smoke.py`` to
prove the network surface is provider-neutral end-to-end. Uses
``gpt-5.4-nano`` for cost.

Loads ``.env`` from the repo root; skips if ``OPENAI_API_KEY`` is
absent. Marked ``@pytest.mark.openai`` so the default unit run skips
them.
"""

import asyncio
import os
from pathlib import Path

import pytest

from autogen.beta import Agent
from autogen.beta.config import OpenAIConfig
from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    EV_TEXT,
    Hub,
    HubClient,
    LocalLink,
    Passport,
    Resume,
)
from autogen.beta.network.adapters.discussion import (
    DISCUSSION_TYPE,
    ORDERING_ROUND_ROBIN,
)

try:
    from dotenv import load_dotenv

    _REPO_ROOT = Path(__file__).resolve().parents[4]
    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass


@pytest.fixture()
def openai_config() -> OpenAIConfig:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return OpenAIConfig(model="gpt-5.4-nano", api_key=api_key, temperature=0)


async def _wait_for_text_count(
    hub: Hub,
    session_id: str,
    expected: int,
    *,
    timeout: float = 90.0,
) -> int:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        wal = await hub.read_wal(session_id)
        count = sum(1 for e in wal if e.event_type == EV_TEXT)
        if count >= expected:
            return count
        await asyncio.sleep(0.2)
    return sum(
        1 for e in (await hub.read_wal(session_id)) if e.event_type == EV_TEXT
    )


@pytest.mark.openai
@pytest.mark.asyncio()
async def test_peers_then_delegate_consults_a_specialist(
    openai_config: OpenAIConfig,
) -> None:
    """alice's LLM discovers bob via ``peers`` then consults via ``delegate``."""
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )
    link = LocalLink(hub)

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    alice_agent = Agent(
        name="alice",
        prompt=(
            "You are a coordinator on a multi-agent network. When given a "
            "question, FIRST call peers(action='find', capability=<topic>) "
            "to discover a specialist. Then call delegate(target=<peer_name>, "
            "prompt=<question>) to ask them. Reply to the user with the "
            "specialist's answer verbatim."
        ),
        config=openai_config,
    )
    bob_agent = Agent(
        name="bob",
        prompt=(
            "You are a math specialist. Answer with just the numeric result, "
            "no explanation."
        ),
        config=openai_config,
    )

    alice = await alice_hc.register(
        alice_agent,
        Passport(name="alice"),
        Resume(summary="multi-agent coordinator"),
    )
    await bob_hc.register(
        bob_agent,
        Passport(name="bob"),
        Resume(claimed_capabilities=["math"], summary="math specialist"),
    )

    reply = await alice.agent.ask(
        "Find a math specialist on the network and ask them: what is 12 times 17?"
    )

    assert reply.body is not None
    assert "204" in reply.body, f"expected 204 in alice's reply, got: {reply.body!r}"

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.openai
@pytest.mark.asyncio()
async def test_3way_discussion_round_robin_via_say_tool(
    openai_config: OpenAIConfig,
) -> None:
    """Three OpenAI agents take round-robin turns via the ``say`` tool.

    Three (vs five for Anthropic) keeps cost low while still exercising
    multi-party N>2 dispatch + ``expected_next_speaker`` rotation.
    """
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )
    link = LocalLink(hub)

    names = ["alice", "bob", "carol"]
    clients = []
    for name in names:
        agent = Agent(
            name=name,
            prompt=(
                f"You are {name}, a participant in a 3-way discussion on a "
                "topic. When it is your turn, contribute exactly one short "
                "opinion (one sentence) by calling say(content=<your "
                "sentence>). Do not ask questions. Do not call any other tool."
            ),
            config=openai_config,
        )
        hc = HubClient(link, hub=hub)
        client = await hc.register(agent, Passport(name=name), Resume())
        clients.append(client)

    alice = clients[0]

    session = await alice.open(
        type=DISCUSSION_TYPE,
        target=[c.agent_id for c in clients[1:]],
        knobs={"ordering": ORDERING_ROUND_ROBIN},
        intent="discuss whether type hints should be mandatory in new Python projects",
    )

    await session.send(
        "Quick debate: should type hints be mandatory in new Python projects?"
    )

    count = await _wait_for_text_count(
        hub, session.session_id, expected=3, timeout=120.0
    )
    assert count >= 3, f"expected 3 turns, got {count}"

    wal = await hub.read_wal(session.session_id)
    speakers = [e.sender_id for e in wal if e.event_type == EV_TEXT][:3]
    expected_order = [c.agent_id for c in clients]
    assert speakers == expected_order, (
        f"round-robin order broken; expected {expected_order}, got {speakers}"
    )

    contributions = [
        e.event_data.get("text", "") for e in wal if e.event_type == EV_TEXT
    ][:3]
    for i, text in enumerate(contributions):
        assert len(text) > 5, f"{names[i]}'s turn was empty/trivial: {text!r}"

    for c in clients:
        await c._hub_client.close()
    await hub.close()
