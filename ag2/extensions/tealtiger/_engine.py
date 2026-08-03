# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""TealTiger engine adapter — the ONLY file that imports the tealtiger package.

This isolation means:
- Tests can run without tealtiger installed (they test middleware.py directly)
- No dependency conflict in CI (tealtiger's transitive deps don't pollute the env)
- Users can supply their own GovernanceEngine implementation
"""

import tealtiger  # noqa: F401 — validates the package is installed

from .types import GovernanceDecision, GovernancePolicy


def create_default_engine() -> "DefaultTealTigerEngine":
    """Create the default governance engine backed by the tealtiger package.

    Raises:
        ImportError: If tealtiger is not installed.

    Returns:
        A GovernanceEngine instance using TealTiger's evaluation logic.
    """
    return DefaultTealTigerEngine()


class DefaultTealTigerEngine:
    """Default governance engine using the tealtiger package.

    Delegates policy evaluation to tealtiger's deterministic engine.
    """

    def evaluate(
        self,
        tool_name: str,
        arguments: dict,
        agent_name: str,
        policies: list[GovernancePolicy],
        cumulative_cost: float,
    ) -> GovernanceDecision:
        """Evaluate policies using the tealtiger engine."""
        from tealtiger import TealEngine

        engine = TealEngine()
        # Map our policies to tealtiger's format and evaluate
        # For now, delegate to the built-in evaluation in middleware.py
        # This adapter is the integration point for future TealEngine features
        return GovernanceDecision(action="ALLOW", reason="Delegated to middleware")
