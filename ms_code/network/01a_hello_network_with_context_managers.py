"""01a - Hello, Multi-Agent Network (using async-with).

Same scenario as ``01_hello_network.py`` — one Hub, two agents, one
``consulting`` session — but lifecycle is handled with chained
``async with`` blocks instead of explicit ``close()`` / ``unregister()``
calls.

The four context managers in play:

* ``async with await Hub.open(...) as hub:`` — Hub.__aexit__ calls
  hub.close() (cancels sweepers, closes the store).
* ``async with HubClient(link, hub=hub) as alice_hc:`` — closes the
  link's receive task on exit.
* ``async with await alice_hc.register(...) as alice_client:`` —
  unregisters the agent from the hub on exit.

If anything in the body raises, every cleanup still runs (because
__aexit__ is exception-safe). No ``try/finally`` boilerplate.

Run::

    python ms_code/network/01a_hello_network_with_context_managers.py
"""

import asyncio

from dotenv import load_dotenv

from autogen.beta import Agent
from autogen.beta.config import AnthropicConfig
from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    EV_SESSION_CLOSED,
    EV_TEXT,
    Hub,
    HubClient,
    LocalLink,
    Passport,
    Resume,
)
from autogen.beta.network.adapters import CONSULTING_TYPE

load_dotenv()


async def main() -> None:
    config = AnthropicConfig(model="claude-sonnet-4-6")

    alice_agent = Agent(
        "alice",
        prompt="You are alice. Ask one focused question and stop.",
        config=config,
    )
    bob_agent = Agent(
        "bob",
        prompt="You are bob. Answer in one short sentence.",
        config=config,
    )

    async with await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0) as hub:
        link = LocalLink(hub)
        async with (  # noqa: SIM117
            HubClient(link, hub=hub) as alice_hc,
            HubClient(link, hub=hub) as bob_hc,
        ):
            async with (
                await alice_hc.register(alice_agent, Passport(name="alice"), Resume()) as alice_client,
                await bob_hc.register(bob_agent, Passport(name="bob"), Resume()) as bob_client,
            ):
                print(f"alice agent_id: {alice_client.agent_id}")
                print(f"bob   agent_id: {bob_client.agent_id}")

                session = await alice_client.open(
                    type=CONSULTING_TYPE,
                    target=[bob_client.agent_id],
                )
                print(f"session opened: {session.session_id}  state={session.state}")

                await session.send(
                    "In one sentence: what's the single most important property of a distributed system?",
                    audience=[bob_client.agent_id],
                )

                await asyncio.sleep(5)  # wait for the reply to be processed and the session to auto-close

                close_env = await alice_client.wait_for_session_event(
                    session_id=session.session_id,
                    predicate=lambda e: e.event_type == EV_SESSION_CLOSED,
                    timeout=60.0,
                )
                print(f"session closed: reason={close_env.event_data.get('reason')!r}")

                wal = await hub.read_wal(session.session_id)
                print("\n-- conversation --")
                for env in wal:
                    if env.event_type == EV_TEXT:
                        speaker = "alice" if env.sender_id == alice_client.agent_id else "bob"
                        print(f"{speaker}: {env.event_data['text']}")
            # ↑ each AgentClient unregisters here
        # ↑ each HubClient closes its link here
    # ↑ Hub closes (cancels sweepers, closes store) here


if __name__ == "__main__":
    asyncio.run(main())
