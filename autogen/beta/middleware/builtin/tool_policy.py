# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tool policy middleware -- blocks disallowed tool calls before execution."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, ToolCallEvent, ToolErrorEvent
from autogen.beta.middleware.base import BaseMiddleware, MiddlewareFactory, ToolExecution, ToolResultType
from autogen.beta.tools import ToolResult

OnBlockedCallback = Callable[[ToolCallEvent, str], None]


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

    The middleware is stateless: no counters or shared mutable state are
    retained across invocations. To observe denied calls, pass an
    ``on_blocked`` callback when constructing the middleware.

    Example::

        def audit(call: ToolCallEvent, reason: str) -> None:
            logger.info("blocked %s: %s", call.name, reason)


        mw = ToolPolicyMiddleware(
            blocked_tools=["delete_all"],
            allowed_tools=["search", "calc"],
            on_blocked=audit,
        )
        agent = MyAgent(middleware=[mw])
    """

    def __init__(
        self,
        blocked_tools: Sequence[str] | None = None,
        allowed_tools: Sequence[str] | None = None,
        on_blocked: OnBlockedCallback | None = None,
    ) -> None:
        self._config = ToolPolicyConfig(
            blocked_tools=list(blocked_tools) if blocked_tools else [],
            allowed_tools=list(allowed_tools) if allowed_tools is not None else None,
        )
        self._policy = _ToolPolicy(self._config)
        self._on_blocked = on_blocked

    def __call__(self, event: "BaseEvent", context: "Context") -> "BaseMiddleware":
        return _ToolPolicyInstance(event, context, self._policy, self._on_blocked)


class _ToolPolicyInstance(BaseMiddleware):
    """Per-invocation middleware instance that enforces the tool policy."""

    def __init__(
        self,
        event: "BaseEvent",
        context: "Context",
        policy: _ToolPolicy,
        on_blocked: OnBlockedCallback | None,
    ) -> None:
        super().__init__(event, context)
        self._policy = policy
        self._on_blocked = on_blocked

    async def on_tool_execution(
        self,
        call_next: ToolExecution,
        event: "ToolCallEvent",
        context: "Context",
    ) -> ToolResultType:
        allowed, reason = self._policy.check(event.name)
        if not allowed:
            if self._on_blocked is not None:
                self._on_blocked(event, reason)
            return _make_tool_error(event, reason)

        return await call_next(event, context)
