"""An agent served over AG-UI stops to ask a human, and a client answers it.

Two exchanges, one turn. The first ends with ``RUN_FINISHED`` carrying an
*interrupt* while the agent's own function is still suspended inside
``context.input``; the second — a new run id on the **same thread** — delivers
the answer into that suspended call and carries the rest of the turn.

Both halves run here: the server is the ASGI app, and the client posts at it
over in-process HTTP, so there is no port to pick and nothing to start first.

    OPENAI_API_KEY=... python -m examples.ag_ui.human_input
"""

import asyncio
import json
import os
from typing import Any
from uuid import uuid4

import httpx
from starlette.applications import Starlette
from starlette.routing import Route

from ag2 import Agent, Context
from ag2.ag_ui import AGUIStream
from ag2.config import OpenAIConfig

QUESTION = "Book LH441 (dep 09:40, arr 11:05)? Reply yes or no."
THREAD = "thread-1"


def booking_agent() -> Agent:
    agent = Agent(
        "travel_desk",
        prompt="You book flights. Always use book_flight; never claim to have booked anything yourself.",
        config=OpenAIConfig(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"]),
    )

    @agent.tool
    async def book_flight(context: Context, flight: str) -> str:
        """Book a flight, once the traveller has confirmed it."""
        print(f"  [agent] suspending inside book_flight({flight!r})")
        answer = await context.input(QUESTION)
        print(f"  [agent] resumed with {answer!r}")
        return "Booked." if answer.strip().lower().startswith("y") else "Not booked."

    return agent


def run_body(*, text: str | None = None, resume: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """One RunAgentInput, spelled the way a browser client spells it."""
    body: dict[str, Any] = {
        "threadId": THREAD,
        "runId": str(uuid4()),
        "state": {},
        "messages": [{"id": str(uuid4()), "role": "user", "content": text}] if text else [],
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }
    if resume is not None:
        body["resume"] = resume
    return body


async def post_run(app: Starlette, body: dict[str, Any]) -> list[dict[str, Any]]:
    """Drive one exchange and decode the AG-UI frames it streamed back."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ag-ui.example") as client:
        response = await client.post("/", json=body)
        response.raise_for_status()
        frames = [
            json.loads(line.removeprefix("data: ")) for line in response.text.splitlines() if line.startswith("data: ")
        ]
    print("  frames:", " → ".join(f["type"] for f in frames))
    return frames


async def main() -> None:
    stream = AGUIStream(booking_agent())
    app = Starlette(routes=[Route("/", stream.build_asgi())])

    async with stream:  # cancels anything still held, on the way out
        print("POST #1 — the agent asks")
        first = await post_run(app, run_body(text="Book me LH441 tomorrow morning."))
        [interrupt] = first[-1]["outcome"]["interrupts"]
        print(f"  question: {interrupt['message']}")
        print(f"  expires:  {interrupt['expiresAt']}")

        # A compliant client carries the interrupt's metadata back untouched:
        # it is the proof that this answer comes from whoever was asked.
        print("POST #2 — the client answers, under a new run id on the same thread")
        answered = await post_run(
            app,
            run_body(
                resume=[
                    {
                        "interruptId": interrupt["id"],
                        "status": "resolved",
                        "payload": "yes",
                        "metadata": interrupt.get("metadata"),
                    }
                ]
            ),
        )
        print(f"  outcome:  {json.dumps(answered[-1]['outcome'])}")


if __name__ == "__main__":
    asyncio.run(main())
