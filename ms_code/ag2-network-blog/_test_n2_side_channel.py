"""N2 §3 — audience addressing and private side-channels.

Three participants share a discussion channel (alice, bob, carol).
alice and bob exchange a message there.
Then alice opens a private consulting side-channel with carol — bob never sees it.

The WAL for each channel is independent: bob can read all of the discussion WAL
but cannot see the side-channel WAL at all.
"""

import asyncio

from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    EV_CHANNEL_CLOSED,
    EV_TEXT,
    Hub,
    HubClient,
    LocalLink,
    Passport,
)


async def main() -> None:
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)
    link = LocalLink(hub)

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)
    carol_hc = HubClient(link, hub=hub)

    alice = await alice_hc.register_human(Passport(name="alice", kind="human"))
    bob = await bob_hc.register_human(Passport(name="bob", kind="human"))
    carol = await carol_hc.register_human(Passport(name="carol", kind="human"))

    # ── Main discussion channel: alice, bob, carol ──────────────────────────
    main_ch = await alice.open(
        type="discussion",
        target=[bob.agent_id, carol.agent_id],
    )
    # alice's turn (she is initial_speaker)
    await main_ch.send("alice (main): topic — should we adopt the new infra?")

    # bob's turn
    await bob.next_envelope(
        predicate=lambda e: e.channel_id == main_ch.channel_id and e.event_type == EV_TEXT,
        timeout=10.0,
    )
    await bob.send(main_ch.channel_id, "bob (main): I need more context before deciding.")

    # carol's turn — she'll reply on the main channel after the side-channel
    await carol.next_envelope(
        predicate=lambda e: (
            e.channel_id == main_ch.channel_id and e.event_type == EV_TEXT and e.sender_id == bob.agent_id
        ),
        timeout=10.0,
    )

    # ── Private side-channel: alice ↔ carol (bob excluded) ─────────────────
    side_ch = await alice.open(
        type="consulting",
        target=carol.agent_id,
    )
    await side_ch.send("carol (private): off the record — do you trust the vendor?")

    # carol receives the side-channel invite and replies
    await carol.next_envelope(
        predicate=lambda e: e.channel_id == side_ch.channel_id and e.event_type == EV_TEXT,
        timeout=10.0,
    )
    await carol.send(side_ch.channel_id, "alice (private reply): between us — not yet.")

    # wait for alice to receive carol's private reply and the channel to close
    await alice.next_envelope(
        predicate=lambda e: e.channel_id == side_ch.channel_id and e.event_type == EV_CHANNEL_CLOSED,
        timeout=10.0,
    )

    # carol now weighs in on the main discussion
    await carol.send(main_ch.channel_id, "carol (main): let's request a reference check first.")

    # ── Show both WALs ──────────────────────────────────────────────────────
    print("=== MAIN CHANNEL WAL (alice + bob + carol) ===")
    for env in await hub.read_wal(main_ch.channel_id):
        if env.event_type == EV_TEXT:
            print(f"  {env.event_data.get('text', '')!r}")

    print("\n=== SIDE-CHANNEL WAL (alice + carol only — bob cannot see this) ===")
    for env in await hub.read_wal(side_ch.channel_id):
        if env.event_type == EV_TEXT:
            print(f"  {env.event_data.get('text', '')!r}")

    print(
        "\n(bob sees",
        len([e for e in await hub.read_wal(main_ch.channel_id) if e.event_type == EV_TEXT]),
        "messages on the main channel and 0 on the side-channel)",
    )

    await main_ch.close()
    await alice_hc.close()
    await bob_hc.close()
    await carol_hc.close()
    await hub.close()


if __name__ == "__main__":
    asyncio.run(main())
