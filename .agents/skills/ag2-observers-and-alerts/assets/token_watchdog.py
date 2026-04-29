"""Token watchdog — observers and alerts.

Mirrors website/docs/beta/code_examples/04_token_watchdog.mdx. Three observer
patterns running against a single Agent:

1. TokenMonitor — built-in, tallies usage and warns above a threshold.
2. LoopDetector — built-in, spots repetitive tool calls.
3. A hand-written BaseObserver that subscribes to ObserverAlert and prints
   a formatted dashboard line every time anything alerts.

Run::

    python token_watchdog.py
"""

import asyncio

from autogen.beta import Agent
from autogen.beta.annotations import Context
from autogen.beta.config import GeminiConfig
from autogen.beta.events import BaseEvent
from autogen.beta.events.alert import ObserverAlert
from autogen.beta.observer import BaseObserver, LoopDetector, TokenMonitor
from autogen.beta.stream import MemoryStream
from autogen.beta.watch import EventWatch


def section(title: str) -> None:
    print(f"\n── {title} ───")


class AlertConsole(BaseObserver):
    """Watches the stream for ObserverAlerts and prints them to stdout."""

    def __init__(self) -> None:
        super().__init__("alert-console", watch=EventWatch(ObserverAlert))
        self.seen: list[ObserverAlert] = []

    async def process(self, events: list[BaseEvent], ctx: Context) -> None:
        for event in events:
            if isinstance(event, ObserverAlert):
                self.seen.append(event)
                print(f"    [{event.severity.upper():<8}] {event.source}: {event.message}")
        return None


async def main() -> None:
    config = GeminiConfig(model="gemini-3-flash-preview", temperature=0)

    section("Watchdog — low thresholds so observers trip on a single ask")

    token_monitor = TokenMonitor(warn_threshold=50, alert_threshold=5_000)
    loop_detector = LoopDetector(window_size=5, repeat_threshold=2)
    console = AlertConsole()

    stream = MemoryStream()

    agent = Agent(
        "writer",
        prompt=("Write prose the user asks for. Favour variety — never repeat the same sentence twice."),
        config=config,
        observers=[token_monitor, loop_detector, console],
    )

    reply = await agent.ask(
        "Write three distinct 30-word paragraphs about springtime in Kyoto.",
        stream=stream,
    )

    print()
    print("Final reply (truncated):")
    print("   ", (reply.body or "")[:240], "...")
    print()
    print(f"Total tokens tracked by TokenMonitor: {token_monitor.total_tokens}")
    print(f"Alerts emitted this run:              {len(console.seen)}")


if __name__ == "__main__":
    asyncio.run(main())
