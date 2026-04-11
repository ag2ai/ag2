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

    The policy can be replaced at runtime via :meth:`update_policy` --
    useful for rate-limit kill switches, per-tenant rules driven by an
    ops dashboard, or role changes that land mid-session without
    rebuilding the agent.

    Example::

        def audit(call: ToolCallEvent, reason: str) -> None:
            logger.info("blocked %s: %s", call.name, reason)


        mw = ToolPolicyMiddleware(
            blocked_tools=["delete_all"],
            allowed_tools=["search", "calc"],
            on_blocked=audit,
        )
        agent = MyAgent(middleware=[mw])

        # Later, swap the policy without rebuilding the agent:
        mw.update_policy(ToolPolicyConfig(blocked_tools=["delete_all", "shell_exec"]))
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

    @property
    def config(self) -> ToolPolicyConfig:
        """Return the currently active policy configuration."""
        return self._config

    def update_policy(self, config: ToolPolicyConfig) -> None:
        """Atomically replace the tool policy at runtime.

        Subsequent tool calls use the new policy. Calls that have already
        entered ``on_tool_execution`` continue with the policy snapshot
        their per-invocation instance captured at construction time, so
        in-flight executions are never torn across a configuration change.

        This is the intended hook for dynamic policy sources: rate-limit
        kill switches, per-tenant rule updates from an ops dashboard,
        role changes during a long-running session, etc.

        Under CPython's GIL each individual attribute assignment is atomic
        at the bytecode level.  ``_policy`` is assigned last so that
        :meth:`__call__` readers never observe a new policy paired with a
        stale :attr:`config`.

        Args:
            config: The new :class:`ToolPolicyConfig` to enforce. Passing
                a fresh config with defaults (empty blocklist, no allowlist)
                effectively clears all restrictions.
        """
        new_policy = _ToolPolicy(config)
        # Assign _config first, _policy last.  __call__ reads only _policy,
        # so the window where _config and _policy disagree is one bytecode
        # instruction -- no reader in __call__ can observe the mismatch.
        self._config = config
        self._policy = new_policy

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
