# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Example: retry a model call with the same status snapshot in both modes.

Status middleware wraps RetryMiddleware: retries neither evaluate the callback
again nor persist another copy. A new user input produces a new snapshot.
"""

import asyncio
from dataclasses import dataclass
from typing import Annotated

from ag2 import Agent, DataInput, Inject
from ag2.events import ModelRequest
from ag2.middleware import RetryMiddleware, statusbar_middleware
from ag2.testing import TestConfig, TrackingConfig


@dataclass
class ComputationCounter:
    computations: int = 0


def render_snapshot(counter: ComputationCounter) -> DataInput:
    counter.computations += 1
    return DataInput({"computation": counter.computations}, metadata={"case": "retry-snapshot"})


@statusbar_middleware(persist=False)
def dynamic_status(counter: Annotated[ComputationCounter, Inject()]) -> DataInput:
    return render_snapshot(counter)


@statusbar_middleware(persist=True)
def persistent_status(counter: Annotated[ComputationCounter, Inject()]) -> DataInput:
    return render_snapshot(counter)


async def run_case(persist: bool) -> None:
    counter = ComputationCounter()
    config = TrackingConfig(
        TestConfig(
            TimeoutError("Temporary provider failure"),
            "Request recovered.",
            "Next input handled.",
        )
    )
    status = persistent_status if persist else dynamic_status
    agent = Agent(
        "retrying_assistant",
        config=config,
        dependencies={"counter": counter},
        middleware=[
            status,
            RetryMiddleware(max_retries=1, retry_on=(TimeoutError,)),
        ],
    )

    reply = await agent.ask("Respond, retrying a temporary provider failure.")
    reply = await reply.ask("Handle the next input.")

    expected = [(1,), (1,), (1, 2)] if persist else [(1,), (1,), (2,)]
    snapshots = [
        tuple(
            part.data["computation"]
            for event in call.messages
            for part in getattr(event, "parts", ())
            if isinstance(part, DataInput) and part.metadata.get("case") == "retry-snapshot"
        )
        for call in config.calls
    ]
    assert snapshots == expected
    assert counter.computations == 2
    assert await reply.history.where(ModelRequest).count() == 2
    retained = [
        part.data["computation"]
        for event in await reply.history.get_events()
        for part in getattr(event, "parts", ())
        if isinstance(part, DataInput) and part.metadata.get("case") == "retry-snapshot"
    ]
    assert retained == ([1, 2] if persist else [])
    print(f"persist={persist}: three provider attempts, two status computations:", snapshots)


async def main() -> None:
    await run_case(persist=False)
    await run_case(persist=True)


if __name__ == "__main__":
    asyncio.run(main())
