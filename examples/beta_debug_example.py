"""
AG2 Beta Debug Server — example script
=======================================

This script demonstrates the debug server in action using a fake LLM client
(no API key required) so it runs fully offline.

What it shows
-------------
- An agent makes one tool call followed by a final text reply.
- The debug middleware intercepts every TURN_START and every TOOL_CALL,
  posting a breakpoint to the server and blocking until you resume it.
- A "driver" coroutine polls the session and automatically resumes each
  breakpoint so the whole demo runs without human interaction.
- At the end the full event + breakpoint history is printed.

To run
------
    python examples/beta_debug_example.py

You can also drive it manually from a second terminal while the agent is
blocked:

    # list sessions
    curl http://localhost:8765/sessions

    # inspect a session
    curl http://localhost:8765/sessions/<session_id>

    # resume a breakpoint
    curl -X POST http://localhost:8765/sessions/<session_id>/breakpoints/<bp_id>/resume
"""

import asyncio
import os
import time

import httpx

from autogen.beta.events import ToolCallEvent
from autogen.beta.events.types import ModelMessage, ModelResponse
from autogen.beta.stream import MemoryStream
from autogen.beta.testing import TestConfig

# ---------------------------------------------------------------------------
# 1. Start the debug server in a background thread
# ---------------------------------------------------------------------------
DEBUG_HOST = "localhost"
DEBUG_PORT = 8765
DEBUG_URL = f"http://{DEBUG_HOST}:{DEBUG_PORT}"


# Give the server a moment to boot
time.sleep(0.5)

# ---------------------------------------------------------------------------
# 2. Set the env var so Agent.ask() auto-injects DebugMiddleware
# ---------------------------------------------------------------------------
os.environ["AG2_DEBUG_SERVER_URL"] = DEBUG_URL

# ---------------------------------------------------------------------------
# 3. Define the agent (fake LLM: tool call → text reply)
# ---------------------------------------------------------------------------
from autogen.beta import Agent
from autogen.beta.tools import tool


@tool
def get_weather(city: str) -> str:
    """Return a fake weather report for *city*."""
    return f"Sunny, 22 °C in {city}"


agent = Agent(
    "weather-agent",
    prompt="You are a helpful weather assistant.",
    config=TestConfig(
        # Turn 1: model asks for the tool
        ToolCallEvent(name="get_weather", arguments='{"city": "Paris"}'),
        # Turn 2: model produces a text reply after seeing the tool result
        ModelResponse(message=ModelMessage(content="It's sunny and 22 °C in Paris — great day!")),
    ),
    tools=[get_weather],
)


# ---------------------------------------------------------------------------
async def main() -> None:
    # We need the session_id before the agent creates it.  The session_id is
    # set to str(stream.id) inside Agent.ask(), which we can't know in advance.
    # Instead we start the driver after the session is registered by polling
    # /sessions until one appears.

    stop = asyncio.Event()

    async def wait_and_drive() -> None:
        """Wait for the first session to appear, then start driving it."""
        async with httpx.AsyncClient(base_url=DEBUG_URL) as client:
            while True:
                resp = await client.get("/sessions", timeout=2.0)
                sessions = resp.json()
                if sessions:
                    session_id = sessions[0]["id"]
                    break
                await asyncio.sleep(5)
        print(f"[driver] found session {session_id[:8]}…")

    driver_task = asyncio.create_task(wait_and_drive())

    memory = MemoryStream()

    print("[main] running agent.ask() …")
    reply = await agent.ask("What's the weather in Paris?", stream=memory)
    print(f"[main] agent replied: {reply.content!r}")

    stop.set()
    await driver_task

    print("memory session id:", memory.id)

    # Print the full session summary
    async with httpx.AsyncClient(base_url=DEBUG_URL) as client:
        session_id = memory.id
        session = (await client.get(f"/sessions/{session_id}")).json()

    print(f"\n{'=' * 60}")
    print(f"Session: {session_id}")
    print(f"Status:  {session['status']}")

    print(f"\nEvents ({len(session['events'])}):")
    for ev in session["events"]:
        print(f"  [{ev['event_type']}]  {ev['event_data']}")

    print(f"\nBreakpoints ({len(session['breakpoints'])}):")
    for bp in session["breakpoints"]:
        print(f"  [{bp['type']}] on {bp['event_type']}  resumed={bp['resumed']}")


if __name__ == "__main__":
    asyncio.run(main())
