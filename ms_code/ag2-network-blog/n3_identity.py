"""N3 · Identity records — Passport, Resume, and SKILL.md.

Every participant carries three records:
- Passport: hub-stamped, immutable — name, agent_id, created_at.
- Resume: tenant-maintained, hub-observed — capabilities and track record.
- SKILL.md: free-form markdown indexed for peer discovery.

No LLM calls — this is a pure identity and peer-discovery demo.
"""

import asyncio

from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    Hub,
    HubClient,
    LocalLink,
    Passport,
    Resume,
)

ALICE_SKILL_MD = """\
## Alice — Risk Analyst

Senior quantitative risk analyst specialising in scenario synthesis,
stress testing, and counterparty exposure modelling.

### Capabilities
- Exposure modelling
- Scenario synthesis
- Regulatory reporting
"""


async def main() -> None:
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)
    link = LocalLink(hub)

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    alice = await alice_hc.register_human(
        Passport(name="alice", owner="acme", kind="human"),
        resume=Resume(
            claimed_capabilities=["analysis", "risk", "exposure-modelling"],
            domains=["finance"],
            summary="Senior risk analyst — scenario synthesis and stress testing.",
        ),
    )
    await hub.set_skill(alice.agent_id, ALICE_SKILL_MD)

    await bob_hc.register_human(
        Passport(name="bob", owner="acme", kind="human"),
        resume=Resume(
            claimed_capabilities=["writing", "summarisation"],
            summary="Technical writer and summariser.",
        ),
    )

    # ── Passport ──────────────────────────────────────────────────────────────
    passport = await hub.get_agent("alice")
    print("=== Passport (hub-stamped) ===")
    print(f"  name       : {passport.name}")
    print(f"  agent_id   : {passport.agent_id}")
    print(f"  owner      : {passport.owner}")
    print(f"  created_at : {passport.created_at}")

    # ── Resume ────────────────────────────────────────────────────────────────
    resume = await hub.get_resume(alice.agent_id)
    print()
    print("=== Resume ===")
    print(f"  claimed_capabilities : {resume.claimed_capabilities}")
    print(f"  domains              : {resume.domains}")
    print(f"  summary              : {resume.summary!r}")

    # ── SKILL.md ──────────────────────────────────────────────────────────────
    skill = await hub.get_skill(alice.agent_id)
    print()
    print("=== SKILL.md ===")
    print(skill)

    # ── Peer discovery ────────────────────────────────────────────────────────
    analysts = await hub.list_agents(capability="analysis")
    print("=== list_agents(capability='analysis') ===")
    for p in analysts:
        print(f"  {p.name}  (id={p.agent_id})")

    writers = await hub.list_agents(capability="writing")
    print()
    print("=== list_agents(capability='writing') ===")
    for p in writers:
        print(f"  {p.name}  (id={p.agent_id})")

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


asyncio.run(main())
