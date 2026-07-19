# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
# SPDX-License-Identifier: Apache-2.0
#
# Maintainer: @nagasatish007
# Additional dependency: tealtiger>=1.3.0

"""TealTiger deterministic governance middleware for AG2.

Provides policy enforcement, PII detection, tool allowlisting, cost tracking,
and per-agent kill switches — all evaluated deterministically in <5ms with no
LLM in the governance path.

Example:
    from ag2 import Agent
    from ag2.extensions.tealtiger import TealTigerMiddleware, GovernancePolicy

    governance = TealTigerMiddleware(
        policies=[
            GovernancePolicy.tool_allowlist(["search", "read_file"]),
            GovernancePolicy.pii_block(["ssn", "credit_card"]),
            GovernancePolicy.cost_limit(max_per_session=5.0),
        ],
        mode="ENFORCE",
    )

    agent = Agent(
        "assistant",
        config=OpenAIConfig("gpt-4o-mini"),
        tools=[search, read_file],
        middleware=[governance],
    )

    # Access governance state on the factory
    governance.freeze("assistant")
    print(governance.decisions)
    print(governance.receipts)
"""

try:
    from ag2.extensions.tealtiger.middleware import TealTigerMiddleware
    from ag2.extensions.tealtiger.types import (
        GovernanceDecision,
        GovernanceMode,
        GovernancePolicy,
        TEECReceipt,
    )
except ImportError as e:
    from ag2.exceptions import missing_additional_dependency

    TealTigerMiddleware = missing_additional_dependency("TealTigerMiddleware", "tealtiger>=1.3.0", e)  # type: ignore[misc]
    GovernanceDecision = missing_additional_dependency("GovernanceDecision", "tealtiger>=1.3.0", e)  # type: ignore[misc]
    GovernanceMode = missing_additional_dependency("GovernanceMode", "tealtiger>=1.3.0", e)  # type: ignore[misc]
    GovernancePolicy = missing_additional_dependency("GovernancePolicy", "tealtiger>=1.3.0", e)  # type: ignore[misc]
    TEECReceipt = missing_additional_dependency("TEECReceipt", "tealtiger>=1.3.0", e)  # type: ignore[misc]

__all__ = [
    "GovernanceDecision",
    "GovernanceMode",
    "GovernancePolicy",
    "TEECReceipt",
    "TealTigerMiddleware",
]
