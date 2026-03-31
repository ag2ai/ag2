# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tool policy middleware -- blocks disallowed tool calls before execution."""

import threading
from dataclasses import dataclass, field

from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, ToolCallEvent, ToolErrorEvent
from autogen.beta.middleware.base import BaseMiddleware, MiddlewareFactory, ToolExecution, ToolResultType
from autogen.beta.tools import ToolResult


@dataclass
class ToolPolicyConfig:
    """Configuration for ToolPolicyMiddleware.

    Args:
        blocked_tools: Tool names that are always denied.
        allowed_tools: If not None, only these tools are allowed.
            An empty list means no tools are permitted.
            None means no restriction based on an allowlist.
    """

    blocked_tools: list[str] = field(default_factory=list)
    allowed_tools: list[str] | None = None

    def __post_init__(self) -> None:
        # Freeze to tuple so the config is immutable after construction.
        self.blocked_tools = tuple(self.blocked_tools)  # type: ignore[assignment]
        if self.allowed_tools is not None:
            self.allowed_tools = tuple(self.allowed_tools)  # type: ignore[assignment]


def _make_tool_error(event: ToolCallEvent, reason: str) -> ToolErrorEvent:
    """Build a ToolErrorEvent with an explicit content string."""
    err = ToolErrorEvent(
        parent_id=event.id,
        name=event.name,
        result=ToolResult(content=None),
        error=PermissionError(reason),
    )
    # Override content so it shows the reason, not a traceback.
    err.content = reason
    return err


class _ToolPolicy:
    """Stateless policy check for a single tool name."""

    def __init__(self, config: ToolPolicyConfig) -> None:
        self._blocked: frozenset[str] = frozenset(config.blocked_tools)
        self._allowed: frozenset[str] | None = (
            frozenset(config.allowed_tools) if config.allowed_tools is not None else None
        )

    def check(self, tool_name: str) -> tuple[bool, str]:
        """Return (allowed, reason).

        blocked_tools always takes precedence over allowed_tools.
        """
        if tool_name in self._blocked:
            return False, f"tool '{tool_name}' is blocked"
        if self._allowed is not None and tool_name not in self._allowed:
            return False, f"tool '{tool_name}' is not in the allowed list"
        return True, ""


class ToolPolicyMiddleware(MiddlewareFactory):
    """Factory that creates per-invocation tool policy middleware instances.

    A single ToolPolicyMiddleware shares counters across all instances it
    creates, so statistics accumulate over the lifetime of the factory.

    Example::

        config = ToolPolicyConfig(
            blocked_tools=["delete_all"],
            allowed_tools=["search", "calc"],
        )
        mw = ToolPolicyMiddleware(config)
        agent = MyAgent(middleware=[mw])
    """

    def __init__(self, config: ToolPolicyConfig | None = None) -> None:
        self._config = config or ToolPolicyConfig()
        self._policy = _ToolPolicy(self._config)
        self._total_tool_calls: int = 0
        self._total_blocked: int = 0
        self._lock = threading.Lock()

    @property
    def total_tool_calls(self) -> int:
        """Number of tool calls that passed the policy check and were forwarded."""
        with self._lock:
            return self._total_tool_calls

    @property
    def total_blocked(self) -> int:
        """Number of tool calls that were blocked."""
        with self._lock:
            return self._total_blocked

    def __call__(self, event: "BaseEvent", context: "Context") -> "BaseMiddleware":
        return _ToolPolicyInstance(event, context, self._policy, self)


class _ToolPolicyInstance(BaseMiddleware):
    """Per-invocation middleware instance that enforces the tool policy."""

    def __init__(
        self,
        event: "BaseEvent",
        context: "Context",
        policy: _ToolPolicy,
        factory: ToolPolicyMiddleware,
    ) -> None:
        super().__init__(event, context)
        self._policy = policy
        self._factory = factory

    async def on_tool_execution(
        self,
        call_next: ToolExecution,
        event: "ToolCallEvent",
        context: "Context",
    ) -> ToolResultType:
        allowed, reason = self._policy.check(event.name)
        if not allowed:
            with self._factory._lock:
                self._factory._total_blocked += 1
            return _make_tool_error(event, reason)

        with self._factory._lock:
            self._factory._total_tool_calls += 1
        return await call_next(event, context)
