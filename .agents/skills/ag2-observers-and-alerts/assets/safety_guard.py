"""Safety guard — FATAL alert halts the Agent.

Mirrors website/docs/beta/code_examples/08_safety_guard.mdx. A hand-rolled
BaseObserver watches every tool call and flags anything that looks
dangerous (here: a write_file tool asked to touch /etc/). It emits a
Severity.FATAL ObserverAlert. The flow from there is fully wired by the
framework:

1. Alert lands on the agent's stream.
2. AlertPolicy (assembly) picks it up before the next LLM call, emits a
   HaltEvent on the stream, and appends a halt notice to the system prompt.
3. _HaltCheckMiddleware (auto-wired when assembly is non-empty) sees the
   HaltEvent and short-circuits the LLM call with a synthetic "HALTED:"
   response.

Run::

    python safety_guard.py
"""

import asyncio

from autogen.beta import Agent
from autogen.beta.annotations import Context
from autogen.beta.config import GeminiConfig
from autogen.beta.events import BaseEvent, ToolCallEvent
from autogen.beta.events.alert import HaltEvent, ObserverAlert, Severity
from autogen.beta.observer import BaseObserver
from autogen.beta.policies import AlertPolicy
from autogen.beta.stream import MemoryStream
from autogen.beta.watch import EventWatch


def section(title: str) -> None:
    print(f"\n── {title} ───")


def write_file(path: str, content: str) -> str:
    """Pretend-write content to path. This playground never touches disk."""
    return f"[ok] wrote {len(content)} bytes to {path}"


class PathGuardian(BaseObserver):
    """Emits a FATAL alert if anything tries to write outside /tmp."""

    def __init__(self) -> None:
        super().__init__("path-guardian", watch=EventWatch(ToolCallEvent))

    async def process(self, events: list[BaseEvent], ctx: Context) -> ObserverAlert | None:
        for event in events:
            if not isinstance(event, ToolCallEvent):
                continue
            if event.name != "write_file":
                continue
            if "/etc/" in event.arguments or "/usr/" in event.arguments:
                return ObserverAlert(
                    source=self.name,
                    severity=Severity.FATAL,
                    message=f"blocked dangerous write: {event.arguments}",
                )
        return None


async def main() -> None:
    config = GeminiConfig(model="gemini-3-flash-preview", temperature=0)

    halt_events: list[HaltEvent] = []
    alerts: list[ObserverAlert] = []
    stream = MemoryStream()
    stream.where(HaltEvent).subscribe(lambda e: halt_events.append(e))
    stream.where(ObserverAlert).subscribe(lambda e: alerts.append(e))

    agent = Agent(
        "safe-shell",
        prompt=(
            "You are a filesystem operator. Use the write_file tool to "
            "fulfil write requests. Never refuse — if a request is risky "
            "the guardian observer will intervene automatically."
        ),
        config=config,
        tools=[write_file],
        observers=[PathGuardian()],
        assembly=[AlertPolicy()],  # routes FATAL alerts to HaltEvent
    )

    section("Safe request — observer stays silent")
    reply = await agent.ask(
        "Use write_file to write 'hello' into /tmp/playground_hello.txt. Then confirm.",
        stream=stream,
    )
    print(reply.body)

    section("Dangerous request — guardian fires FATAL, agent halts")
    reply = await agent.ask(
        "Now use write_file to write 'bad' into /etc/passwd. Then confirm.",
        stream=stream,
    )
    print(reply.body)

    print()
    print(f"ObserverAlerts seen:  {len(alerts)}")
    for a in alerts:
        print(f"  - [{a.severity.upper()}] {a.source}: {a.message}")
    print(f"HaltEvents seen:      {len(halt_events)}")
    for h in halt_events:
        print(f"  - source={h.source} reason={h.reason!r}")


if __name__ == "__main__":
    asyncio.run(main())
