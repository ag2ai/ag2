# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
# SPDX-License-Identifier: Apache-2.0

"""Type definitions for TealTiger governance middleware."""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GovernanceMode(str, Enum):
    """Governance enforcement mode."""

    OBSERVE = "OBSERVE"  # Log decisions, never block
    MONITOR = "MONITOR"  # Log decisions with warnings, never block
    ENFORCE = "ENFORCE"  # Log decisions AND block denied tool calls


@dataclass
class GovernancePolicy:
    """A governance policy definition.

    Use the class methods to create specific policy types.
    """

    type: str
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def tool_allowlist(cls, allowed: list[str]) -> GovernancePolicy:
        """Only allow tools matching these patterns (supports glob-style '*' suffix).

        Args:
            allowed: List of tool name patterns. Use '*' suffix for prefix matching.
        """
        return cls(type="tool_allowlist", config={"allowed": allowed})

    @classmethod
    def pii_block(cls, categories: list[str] | None = None) -> GovernancePolicy:
        """Block tool calls containing PII in arguments.

        Args:
            categories: PII categories to detect. Default: ["ssn", "credit_card", "email", "phone"].
        """
        return cls(
            type="pii_block",
            config={"categories": categories or ["ssn", "credit_card", "email", "phone"]},
        )

    @classmethod
    def secret_detection(cls) -> GovernancePolicy:
        """Block tool calls containing API keys/tokens in arguments."""
        return cls(type="secret_detection", config={})

    @classmethod
    def cost_limit(cls, max_per_session: float) -> GovernancePolicy:
        """Deny tool calls once cumulative session cost exceeds the limit.

        Args:
            max_per_session: Maximum allowed cost in USD per session.
        """
        return cls(type="cost_limit", config={"max_per_session": max_per_session})


@dataclass
class GovernanceDecision:
    """A governance evaluation result."""

    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action: str = "ALLOW"  # ALLOW, DENY, MONITOR
    mode: str = "OBSERVE"
    agent_name: str = ""
    tool_name: str = ""
    reason_codes: list[str] = field(default_factory=list)
    risk_score: int = 0
    evaluation_time_ms: float = 0.0
    cost_tracked: float = 0.0
    cumulative_cost: float = 0.0
    timestamp_ms: float = field(default_factory=lambda: time.time() * 1000)


@dataclass
class TEECReceipt:
    """Typed Evidence & Evidence Contract receipt — tamper-evident governance record."""

    receipt_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    decision_id: str = ""
    agent_name: str = ""
    tool_name: str = ""
    action: str = "ALLOW"
    execution_outcome: str = "executed"  # executed, blocked, pending
    reason_codes: list[str] = field(default_factory=list)
    risk_score: int = 0
    policy_digest: str = ""
    timestamp_ms: float = field(default_factory=lambda: time.time() * 1000)
