# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Network tool surface smoke tests against a real LLM.

LLM-driven agents call ``peers`` / ``delegate`` / ``say`` /
``sessions`` to coordinate via the network. They use
``claude-haiku-4-5`` for cost.

The tests load ``.env`` from the repo root so they run without shell
env setup; they ``pytest.skip`` if ``ANTHROPIC_API_KEY`` is absent.
Mark ``@pytest.mark.anthropic`` so the default unit run skips them.
"""

import asyncio
import os
from pathlib import Path

import pytest

from autogen.beta import Agent
from autogen.beta.config import AnthropicConfig
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

# Load .env from repo root so smoke tests run without shell setup.
try:
    from dotenv import load_dotenv

    _REPO_ROOT = Path(__file__).resolve().parents[4]
    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass


@pytest.fixture()
def anthropic_config() -> AnthropicConfig:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return AnthropicConfig(model="claude-haiku-4-5", api_key=api_key, temperature=0)


async def _wait_for_text_count(
    hub: Hub,
    session_id: str,
    expected: int,
    *,
    timeout: float = 60.0,
) -> int:
    """Poll until at least ``expected`` ``EV_TEXT`` envelopes appear, return the count."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        wal = await hub.read_wal(session_id)
        count = sum(1 for e in wal if e.event_type == EV_TEXT)
        if count >= expected:
            return count
        await asyncio.sleep(0.2)
    return sum(1 for e in (await hub.read_wal(session_id)) if e.event_type == EV_TEXT)


@pytest.mark.anthropic
@pytest.mark.asyncio()
async def test_peers_then_delegate_consults_a_specialist(
    anthropic_config: AnthropicConfig,
) -> None:
    """alice's LLM is told to discover a specialist via ``peers`` and
    consult them via ``delegate``. Verifies the discovery → consult chain
    works end-to-end with a real model."""
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
        config=anthropic_config,
    )
    bob_agent = Agent(
        name="bob",
        prompt=("You are a math specialist. Answer with just the numeric result, no explanation."),
        config=anthropic_config,
    )

    alice = await alice_hc.register(
        alice_agent,
        Passport(name="alice"),
        Resume(summary="multi-agent coordinator"),
    )
    bob = await bob_hc.register(
        bob_agent,
        Passport(name="bob"),
        Resume(claimed_capabilities=["math"], summary="math specialist"),
    )

    reply = await alice.agent.ask("Find a math specialist on the network and ask them: what is 12 times 17?")

    assert reply.body is not None
    assert "204" in reply.body, f"expected 204 in alice's reply, got: {reply.body!r}"

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.anthropic
@pytest.mark.asyncio()
async def test_5way_discussion_round_robin_via_say_tool(
    anthropic_config: AnthropicConfig,
) -> None:
    """Five LLM agents in a round-robin discussion. Each speaker takes
    one turn via the ``say`` tool; the adapter rotates ``expected_next_speaker``
    after every accepted envelope. Verifies that a multi-party
    LLM-driven session works end-to-end with bounded prompt size."""
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )
    link = LocalLink(hub)

    names = ["alice", "bob", "carol", "dave", "erin"]
    clients = []
    for name in names:
        agent = Agent(
            name=name,
            prompt=(
                f"You are {name}, a participant in a 5-way discussion on the "
                "topic of Python's adoption in scientific computing. When it "
                "is your turn, contribute exactly one short opinion (one "
                "sentence) by calling say(content=<your sentence>). Do not "
                "ask questions. Do not call any other tool."
            ),
            config=anthropic_config,
        )
        hc = HubClient(link, hub=hub)
        client = await hc.register(agent, Passport(name=name), Resume())
        clients.append(client)

    alice = clients[0]

    session = await alice.open(
        type=DISCUSSION_TYPE,
        target=[c.agent_id for c in clients[1:]],
        knobs={"ordering": ORDERING_ROUND_ROBIN},
        intent="discuss Python's adoption in scientific computing",
    )

    # Alice kicks off with a manual send to seed the conversation;
    # the adapter then rotates through bob → carol → dave → erin.
    await session.send("Let's debate: has Python overtaken R as the lingua franca of scientific computing?")

    # Wait for 5 EV_TEXT envelopes total (alice's seed + 4 LLM responses).
    count = await _wait_for_text_count(hub, session.session_id, expected=5, timeout=90.0)
    assert count >= 5, f"expected 5 turns, got {count}"

    # Verify round-robin order: speakers in WAL match participant order.
    wal = await hub.read_wal(session.session_id)
    speakers = [e.sender_id for e in wal if e.event_type == EV_TEXT][:5]
    expected_order = [c.agent_id for c in clients]
    assert speakers == expected_order, f"round-robin order broken; expected {expected_order}, got {speakers}"

    # Each speaker actually contributed substantive text (>10 chars).
    contributions = [e.event_data.get("text", "") for e in wal if e.event_type == EV_TEXT][:5]
    for i, text in enumerate(contributions):
        assert len(text) > 10, f"{names[i]}'s turn was empty/trivial: {text!r}"

    for hc in [c._hub_client for c in clients]:
        await hc.close()
    await hub.close()
