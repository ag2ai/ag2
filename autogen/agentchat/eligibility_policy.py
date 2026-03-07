# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..doc_utils import export_module

if TYPE_CHECKING:
    from autogen.agentchat.agent import Agent


@export_module("autogen")
@dataclass(frozen=True)
class SelectionContext:
    """Minimal context passed to AgentEligibilityPolicy.is_eligible().

    Intentionally minimal to avoid coupling policies to GroupChat internals.
    Do NOT add the GroupChat object itself -- policies should be stateless
    with respect to the chat object.
    """

    round: int
    """Number of messages spoken so far.

    In normal ``GroupChatManager`` flow, equals 1 during the first speaker
    selection (the initial message is appended before ``select_speaker`` is
    called). Use ``round == 1`` to detect the first selection in that context.

    Example::

        def is_eligible(self, agent, ctx):
            if ctx.round == 1:
                return agent.name != "expensive_agent"  # skip on opening turn
            return True
    """

    last_speaker: str | None
    """Name of the last speaker, or None if this is the first round."""

    participants: tuple[str, ...]
    """Names of all registered participants in the GroupChat (the full roster,
    not the filtered candidate set at the point of this call)."""


@export_module("autogen")
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

    Note:
        ``isinstance(obj, AgentEligibilityPolicy)`` only checks that ``is_eligible``
        exists as an attribute -- it does **not** verify the method signature.
        A class with the wrong arity will fail at call time, not at registration.
    """

    def is_eligible(self, agent: Agent, ctx: SelectionContext) -> bool:
        """Return True if agent should be included in the candidate set.

        Args:
            agent: The candidate agent being evaluated.
            ctx: Minimal context about the current selection round.

        Returns:
            True to keep the agent in the candidate set; False to exclude it.
        """
        ...


_UNAVAILABLE_PREFIX = "[UNAVAILABLE] "


@export_module("autogen")
class AgentDescriptionGuard:
    """Manages soft-signal description mutation for LLM-based speaker selection.

    Wraps an agent via composition and toggles an ``[UNAVAILABLE]`` prefix on
    its description so that LLM-based auto-selection is less likely to choose it
    when it is unavailable (e.g. circuit breaker open).  Restore on recovery.

    Warning:
        This guard owns the ``[UNAVAILABLE]`` prefix on ``agent.description``.
        External code may modify the description between ``mark_unavailable``
        and ``mark_available`` calls -- those changes are preserved.  However,
        if external code adds or removes the ``[UNAVAILABLE]`` prefix itself,
        the guard's state becomes inconsistent.
    """

    def __init__(self, agent: Agent) -> None:
        self._agent = agent
        self._lock = threading.Lock()

    def mark_unavailable(self) -> None:
        """Prepend [UNAVAILABLE] to agent.description (idempotent, thread-safe).

        Note:
            If ``agent.description`` is ``None``, it is treated as ``""`` internally.
        """
        with self._lock:
            current = self._agent.description or ""
            if current.startswith(_UNAVAILABLE_PREFIX):
                return
            self._agent.description = _UNAVAILABLE_PREFIX + current

    def mark_available(self) -> None:
        """Strip [UNAVAILABLE] prefix from description (no-op if not marked, thread-safe).

        Any modifications made to the description after the prefix (e.g. by
        external code appending text) are preserved.
        """
        with self._lock:
            current = self._agent.description or ""
            if current.startswith(_UNAVAILABLE_PREFIX):
                self._agent.description = current[len(_UNAVAILABLE_PREFIX) :]
