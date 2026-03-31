"""Unified agent execution with event capture.

Provides a single execution path for running discovered agents across
all CLI commands (run, chat, serve, test), using the modern run() and
run_group_chat() APIs.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .discovery import DiscoveredAgent


@dataclass
class RunResult:
    """Structured result from agent execution."""

    output: str = ""
    turns: int = 0
    cost: Any = None
    elapsed: float = 0.0
    errors: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    agent_names: list[str] = field(default_factory=list)
    last_speaker: str | None = None


class CliIOStream:
    """Custom IOStream that routes AG2 agent events to CLI callbacks.

    Used only for the main() execution path where user code may use
    IOStream directly.
    """

    def __init__(
        self,
        on_print: Callable[[str], None] | None = None,
        on_event: Callable[[Any], None] | None = None,
    ):
        self._on_print = on_print
        self._on_event = on_event

    def print(self, *objects: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
        text = sep.join(str(o) for o in objects)
        if self._on_print:
            self._on_print(text)

    def send(self, message: Any) -> None:
        if self._on_event:
            self._on_event(message)

    def input(self, prompt: str = "", *, password: bool = False) -> str:
        return ""


def _drain_events(response: Any, result: RunResult, on_event: Callable[[Any], None] | None) -> None:
    """Iterate over a RunResponse's events, forwarding to the callback and auto-exiting input requests."""
    for event in response.events:
        if on_event:
            on_event(event)
        if getattr(event, "type", None) == "input_request":
            event.content.respond("exit")

    result.output = response.summary or ""
    result.turns = len(list(response.messages))
    if response.cost:
        result.cost = response.cost
    if response.last_speaker:
        result.last_speaker = response.last_speaker


def _extract_chat_result(ret: Any, result: RunResult) -> None:
    """Extract output, history, and cost from an AG2 ChatResult (main() path only)."""
    if hasattr(ret, "chat_history"):
        result.history = ret.chat_history or []
        result.turns = len(result.history)

        if hasattr(ret, "summary") and ret.summary:
            result.output = ret.summary
        else:
            agent_msgs = [m for m in result.history if m.get("name", "").lower() != "user" and m.get("content")]
            result.output = agent_msgs[-1]["content"] if agent_msgs else ""

        if hasattr(ret, "cost"):
            result.cost = ret.cost
    elif isinstance(ret, str):
        result.output = ret
        result.turns = 1


def execute(
    discovered: DiscoveredAgent,
    message: str,
    *,
    max_turns: int = 10,
    on_print: Callable[[str], None] | None = None,
    on_event: Callable[[Any], None] | None = None,
    clear_history: bool = True,
) -> RunResult:
    """Execute a discovered agent with optional live event callbacks.

    Args:
        discovered: A DiscoveredAgent from the discovery module.
        message: The input message to send.
        max_turns: Maximum conversation turns.
        on_print: Callback for agent print output (main() path only).
        on_event: Callback for structured AG2 events (live rendering).
        clear_history: Whether to clear chat history before this turn.
            Set to False for multi-turn conversations to preserve context.

    Returns:
        RunResult with output, history, cost, timing, and errors.
    """
    d = discovered
    start = time.time()
    result = RunResult(agent_names=d.agent_names)

    try:
        if d.kind == "main":
            iostream = CliIOStream(on_print=on_print, on_event=on_event)

            def _set_iostream() -> Any:
                try:
                    from autogen.io.base import IOStream

                    return IOStream.set_default(iostream)
                except ImportError:
                    from contextlib import nullcontext

                    return nullcontext()

            fn = d.main_fn
            if fn is None:
                result.errors.append("No main function found")
                result.elapsed = time.time() - start
                return result

            kwargs: dict[str, Any] = {}
            sig = inspect.signature(fn)
            if "message" in sig.parameters and message:
                kwargs["message"] = message

            with _set_iostream():
                ret = fn(**kwargs)
                if asyncio.iscoroutine(ret):
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = None
                    if loop is not None and loop.is_running():
                        import concurrent.futures

                        def _run_with_iostream(coro):
                            with _set_iostream():
                                return asyncio.run(coro)

                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            ret = pool.submit(_run_with_iostream, ret).result()
                    else:
                        ret = asyncio.run(ret)

            _extract_chat_result(ret, result)

        elif d.kind == "agent":
            response = d.agent.run(
                message=message,
                max_turns=max_turns,
                clear_history=clear_history,
            )
            _drain_events(response, result, on_event)

        elif d.kind == "agents":
            from autogen.agentchat import run_group_chat
            from autogen.agentchat.group.patterns.auto import AutoPattern
            from autogen.agentchat.group.patterns.round_robin import RoundRobinPattern

            if len(d.agents) <= 2:
                pattern = RoundRobinPattern(initial_agent=d.agents[0], agents=d.agents)
            else:
                pattern = AutoPattern(
                    initial_agent=d.agents[0],
                    agents=d.agents,
                    group_manager_args={"llm_config": d.agents[0].llm_config},
                )

            response = run_group_chat(
                pattern=pattern,
                messages=message,
                max_rounds=max_turns,
            )
            _drain_events(response, result, on_event)

        else:
            result.errors.append(f"Unknown discovery kind: {d.kind}")

    except Exception as exc:
        result.errors.append(str(exc))

    result.elapsed = time.time() - start
    return result
