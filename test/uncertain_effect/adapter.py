# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Uncertain-effect runtime adapter for AG2.

The effect is performed by an agent tool that ``Agent.ask`` executes. The model is
``ag2.testing.TestConfig``, so no language model is involved.

Run as ``python adapter.py [--variant V] [--error-outcome O] scenario.json`` with
``AEV_EFFECT_OWNER`` set to the effect owner's base URL, which is how
``agent-evidence-vectors runtime`` drives it.

Variants:

* ``default`` asks once. A tool exception becomes a ``ToolErrorEvent`` that the
  scripted model is handed, and the model then finishes.
* ``reask`` asks again on the same stream, up to two more times, while the last
  tool result was an error: a supervisor re-running the conversation.
* ``reconcile`` asks once, with a tool that asks the owner before it sends again
  after a lost reply and waits for a final answer. It reports its own outcome.

``--error-outcome`` is the outcome reported when the run holds no tool result
that is not an error: ``not_run`` (default) or ``unknown``.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import httpx

from ag2 import Agent
from ag2.events import TextInput, ToolCallEvent, ToolErrorEvent, ToolResultEvent
from ag2.stream import MemoryStream
from ag2.testing import TestConfig

ASKS = {"default": 1, "reask": 3, "reconcile": 1}
LOOKUPS = 5
SENDS = 3


@dataclass
class OrderService:
    owner: str
    fencing_token: int | None = None
    tool_calls: int = 0
    reconciling: bool = False

    def create_order(self, sku: str, quantity: int, idempotency_key: str) -> str:
        """Create an order at the order service."""
        self.tool_calls += 1
        body: dict[str, Any] = {
            "args": {"sku": sku, "quantity": quantity},
            "idempotencyKey": idempotency_key,
        }
        if self.fencing_token is not None:
            body["fencingToken"] = self.fencing_token
        if self.reconciling:
            return self.send_reconciling(body)
        reply = httpx.post(f"{self.owner}/effects", json=body, timeout=5.0)
        reply.raise_for_status()
        return "confirmed"

    def send_reconciling(self, body: dict[str, Any]) -> str:
        """Send, and after a lost reply ask the owner before sending again."""
        for _ in range(SENDS):
            if not self.holds_lease():
                return "committed" if self.settle(body["idempotencyKey"]) == "present" else "unknown"
            try:
                httpx.post(f"{self.owner}/effects", json=body, timeout=5.0).raise_for_status()
            except httpx.HTTPStatusError as error:
                if error.response.status_code == 409:
                    continue  # a definite no-effect answer: a resend is safe
            except httpx.TransportError:
                pass
            else:
                return "committed"
            if (settled := self.settle(body["idempotencyKey"])) != "absent":
                return "committed" if settled == "present" else "unknown"
        return "not_run"

    def settle(self, key: str) -> str | None:
        """Ask until the owner gives a final answer; None while it cannot answer."""
        for _ in range(LOOKUPS):
            answer = self.get(f"/effects/{key}")
            if answer is None:
                return None
            if answer["result"] == "present":
                return "present"
            if answer["final"]:
                return "absent"
            time.sleep(0.05)
        return None

    def holds_lease(self) -> bool:
        if self.fencing_token is None:
            return True
        lease = self.get("/lease")
        return lease is not None and lease["token"] == self.fencing_token

    def get(self, path: str) -> dict[str, Any] | None:
        try:
            reply = httpx.get(f"{self.owner}{path}", timeout=5.0)
            reply.raise_for_status()
            return reply.json()
        except (httpx.HTTPError, ValueError):
            return None


async def tool_result(stream: MemoryStream) -> str | None:
    """The text of the last tool result that is not an error, if any."""
    events = await stream.history.get_events()
    texts = [
        "".join(part.content for part in e.result.parts if isinstance(part, TextInput))
        for e in events
        if isinstance(e, ToolResultEvent) and not isinstance(e, ToolErrorEvent)
    ]
    return texts[-1] if texts else None


async def run(scenario: dict[str, Any], variant: str) -> tuple[str | None, int]:
    operation = scenario["operation"]
    lease = scenario["owner"].get("lease")
    service = OrderService(
        owner=os.environ["AEV_EFFECT_OWNER"],
        fencing_token=lease["token"] if lease else None,
        reconciling=variant == "reconcile",
    )
    arguments = json.dumps({**operation["args"], "idempotency_key": operation["idempotencyKey"]})
    turns: list[Any] = []
    for number in range(ASKS[variant]):
        turns.append(ToolCallEvent(name="create_order", arguments=arguments, id=f"call-{number + 1}"))
        turns.append("done")
    agent = Agent(
        "orders",
        config=TestConfig(*turns, raise_tool_errors=False),
        tools=[service.create_order],
    )
    stream = MemoryStream()
    prompt = "place the order"
    for _ in range(ASKS[variant]):
        await agent.ask(prompt, stream=stream)
        if (result := await tool_result(stream)) is not None:
            return result, service.tool_calls
        prompt = "the order did not go through, try again"
    return None, service.tool_calls


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(ASKS), default="default")
    parser.add_argument("--error-outcome", choices=("not_run", "unknown"), default="not_run")
    parser.add_argument("scenario")
    args = parser.parse_args()
    with open(args.scenario, encoding="utf-8") as handle:
        scenario = json.load(handle)
    result, tool_calls = asyncio.run(run(scenario, args.variant))
    outcome = {"confirmed": "committed"}.get(result, result) if result else args.error_outcome
    print(f"tool_calls={tool_calls}", file=sys.stderr)
    print(json.dumps({"outcome": outcome}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
