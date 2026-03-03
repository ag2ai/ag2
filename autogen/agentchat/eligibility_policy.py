from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from autogen.agentchat.agent import Agent


@dataclass(frozen=True)
class SelectionContext:
    """Minimal context passed to AgentEligibilityPolicy.is_eligible().

    Intentionally minimal to avoid coupling policies to GroupChat internals.
    Do NOT add the GroupChat object itself — policies should be stateless
    with respect to the chat object.
    """

    round: int
    """Current round index (0-based)."""

    last_speaker: str | None
    """Name of the last speaker, or None if this is the first round."""

    participants: list[str]
    """Names of all registered participants in the GroupChat."""


@runtime_checkable
class AgentEligibilityPolicy(Protocol):
    """Protocol for runtime eligibility filters in GroupChat speaker selection.

    Implement this protocol to remove agents from the candidate set before
    the speaker selection method (auto/manual/random/round_robin) runs.

    Example - always allow all agents (no-op, equivalent to no policy)::

        class AllowAll:
            def is_eligible(self, agent: Agent, ctx: SelectionContext) -> bool:
                return True

    Example - exclude a specific agent by name::

        class ExcludeAgent:
            def __init__(self, name: str) -> None:
                self.name = name

            def is_eligible(self, agent: Agent, ctx: SelectionContext) -> bool:
                return agent.name != self.name

    Multiple policies can be registered on a GroupChat; all must return True
    for an agent to be considered eligible (AND semantics).
    """

    def is_eligible(self, agent: "Agent", ctx: SelectionContext) -> bool:
        """Return True if agent should be included in the candidate set.

        Args:
            agent: The candidate agent being evaluated.
            ctx: Minimal context about the current selection round.

        Returns:
            True to keep the agent in the candidate set; False to exclude it.
        """
        ...


_UNAVAILABLE_PREFIX = "[UNAVAILABLE] "


class DescriptionMutationMixin:
    """Manages soft-signal description mutation for LLM-based speaker selection.

    When an agent becomes unavailable (e.g. circuit breaker opens), prepend
    [UNAVAILABLE] to the agent description so LLM-based auto-selection
    is less likely to choose it.  Restore on recovery.
    """

    def __init__(self, agent: "Agent") -> None:
        self._agent = agent
        self._original_description: str | None = None

    def mark_unavailable(self) -> None:
        """Prepend [UNAVAILABLE] to agent.description (idempotent)."""
        current = self._agent.description or ""
        if current.startswith(_UNAVAILABLE_PREFIX):
            return
        self._original_description = current
        self._agent.description = _UNAVAILABLE_PREFIX + current

    def mark_available(self) -> None:
        """Restore original description (no-op if not marked)."""
        if self._original_description is not None:
            self._agent.description = self._original_description
            self._original_description = None
