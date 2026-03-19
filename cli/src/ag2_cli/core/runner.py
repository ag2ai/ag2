"""Unified agent execution with event capture.

Provides a single execution path for running discovered agents across
all CLI commands (run, chat, serve, test), eliminating duplication and
enabling IOStream-based live event rendering.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable


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

    Implements the IOStreamProtocol expected by AG2's IOStream.set_default().
    Captures both print() calls (from initiate_chat path) and send() events
    (from the newer event system).
    """

    def __init__(
        self,
        on_print: Callable[[str], None] | None = None,
        on_event: Callable[[Any], None] | None = None,
    ):
        self._on_print = on_print
        self._on_event = on_event

    def print(
        self, *objects: Any, sep: str = " ", end: str = "\n", flush: bool = False
    ) -> None:
        text = sep.join(str(o) for o in objects)
        if self._on_print:
            self._on_print(text)

    def send(self, message: Any) -> None:
        if self._on_event:
            self._on_event(message)

    def input(self, prompt: str = "", *, password: bool = False) -> str:
        # Non-interactive by default; chat command overrides this
        return ""


def _extract_chat_result(ret: Any, result: RunResult) -> None:
    """Extract output, history, and cost from an AG2 ChatResult."""
    if hasattr(ret, "chat_history"):
        result.history = ret.chat_history or []
        result.turns = len(result.history)

        # Prefer ChatResult.summary — AG2 sets this to the last agent reply
        if hasattr(ret, "summary") and ret.summary:
            result.output = ret.summary
        else:
            # Fallback: get last message not from the "user" proxy (by name, not role).
            # In AG2 chat history, role is relative to the initiator, so we filter
            # by the 'name' field instead.
            agent_msgs = [
                m
                for m in result.history
                if m.get("name", "").lower() != "user" and m.get("content")
            ]
            result.output = agent_msgs[-1]["content"] if agent_msgs else ""

        if hasattr(ret, "cost"):
            result.cost = ret.cost
    elif isinstance(ret, str):
        result.output = ret
        result.turns = 1


def _create_user_proxy(ag2: Any) -> Any:
    """Create a non-interactive UserProxyAgent for CLI execution."""
    return ag2.UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )


def execute(
    discovered: Any,
    message: str,
    *,
    max_turns: int = 10,
    on_print: Callable[[str], None] | None = None,
    on_event: Callable[[Any], None] | None = None,
    user_proxy: Any = None,
    clear_history: bool = True,
) -> RunResult:
    """Execute a discovered agent with optional live event callbacks.

    Args:
        discovered: A DiscoveredAgent from the discovery module.
        message: The input message to send.
        max_turns: Maximum conversation turns.
        on_print: Callback for agent print output (live rendering).
        on_event: Callback for structured AG2 events (live rendering).
        user_proxy: Reuse an existing UserProxyAgent (for multi-turn chat).
        clear_history: Whether to clear chat history before this turn.
            Set to False for multi-turn conversations to preserve context.

    Returns:
        RunResult with output, history, cost, timing, and errors.
    """
    from .discovery import DiscoveredAgent

    d: DiscoveredAgent = discovered
    start = time.time()
    result = RunResult(agent_names=d.agent_names)
    iostream = CliIOStream(on_print=on_print, on_event=on_event)

    def _set_iostream() -> Any:
        """Import IOStream and return context manager. Returns nullcontext if unavailable."""
        try:
            from autogen.io.base import IOStream

            return IOStream.set_default(iostream)
        except ImportError:
            from contextlib import nullcontext

            return nullcontext()

    try:
        if d.kind == "main":
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
                    ret = asyncio.run(ret)

            _extract_chat_result(ret, result)

        elif d.kind == "agent":
            import autogen

            if user_proxy is None:
                user_proxy = _create_user_proxy(autogen)

            with _set_iostream():
                ret = user_proxy.initiate_chat(
                    d.agent,
                    message=message,
                    max_turns=max_turns,
                    clear_history=clear_history,
                )

            _extract_chat_result(ret, result)

        elif d.kind == "agents":
            try:
                from autogen.agentchat.group import run_group_chat
                from autogen.agentchat.group.patterns.pattern import AutoPattern

                pattern = AutoPattern(
                    initial_agent=d.agents[0],
                    agents=d.agents,
                )
                response = run_group_chat(
                    pattern=pattern,
                    messages=message,
                    max_rounds=max_turns,
                )
                # Iterate events for live rendering
                for event in response.events:
                    if on_event:
                        on_event(event)

                result.output = response.summary or ""
                if hasattr(response, "cost"):
                    result.cost = response.cost
                if hasattr(response, "last_speaker"):
                    result.last_speaker = response.last_speaker

            except ImportError:
                # Fallback to classic GroupChat
                import autogen

                groupchat = autogen.GroupChat(
                    agents=d.agents, messages=[], max_round=max_turns
                )
                manager = autogen.GroupChatManager(groupchat=groupchat)
                if user_proxy is None:
                    user_proxy = _create_user_proxy(autogen)

                with _set_iostream():
                    ret = user_proxy.initiate_chat(manager, message=message)

                _extract_chat_result(ret, result)
        else:
            result.errors.append(f"Unknown discovery kind: {d.kind}")

    except Exception as exc:
        result.errors.append(str(exc))

    result.elapsed = time.time() - start
    return result
