# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Type definitions for TealTiger governance middleware."""

import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class GovernanceMode(str, Enum):
    """Governance enforcement mode."""

    ENFORCE = "ENFORCE"
    MONITOR = "MONITOR"
    OBSERVE = "OBSERVE"


@dataclass
class GovernancePolicy:
    """A governance policy rule.

    Attributes:
        type: Policy type identifier.
        config: Policy-specific configuration.
    """

    type: str
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def tool_allowlist(cls, allowed: list[str]) -> "GovernancePolicy":
        """Create a tool allowlist policy."""
        return cls(type="tool_allowlist", config={"allowed": allowed})

    @classmethod
    def pii_block(cls, categories: list[str] | None = None) -> "GovernancePolicy":
        """Create a PII blocking policy."""
        return cls(
            type="pii_block",
            config={"categories": categories or ["ssn", "credit_card", "email", "phone"]},
        )

    @classmethod
    def secret_block(cls, categories: list[str] | None = None) -> "GovernancePolicy":
        """Create a secret detection policy."""
        return cls(
            type="secret_block",
            config={"categories": categories or ["api_key", "password", "token", "private_key"]},
        )

    @classmethod
    def cost_limit(cls, max_per_session: float = 5.0) -> "GovernancePolicy":
        """Create a cost limit policy."""
        return cls(type="cost_limit", config={"max_per_session": max_per_session})


@dataclass
class GovernanceDecision:
    """Structured governance decision."""

    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action: str = "ALLOW"
    reason: str = ""
    reason_codes: list[str] = field(default_factory=list)
    risk_score: float = 0.0
    tool_name: str = ""
    agent_name: str = ""
    evaluation_time_ms: float = 0.0
    findings: list[dict[str, Any]] = field(default_factory=list)
    cumulative_cost: float = 0.0


@dataclass
class TEECReceipt:
    """Typed Evidence & Evidence Contract receipt for audit."""

    receipt_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    decision_id: str = ""
    agent_name: str = ""
    tool_name: str = ""
    action: str = ""
    policy_id: str = ""
    timestamp_ms: float = 0.0
    decision_source: str = "policy_engine"
    execution_outcome: str = ""


# --- PII and Secret patterns (used by middleware directly) ---

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
}

SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)(?:api[_-]?key|apikey)\s*[:=]\s*['\"]?[a-zA-Z0-9_\-]{20,}"),
    re.compile(r"(?i)(?:password|passwd|pwd)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
    re.compile(r"(?i)(?:token|bearer|auth)\s*[:=]\s*['\"]?[a-zA-Z0-9_\-.]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----"),
    re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}"),
]


@runtime_checkable
class GovernanceEngine(Protocol):
    """Protocol for pluggable governance engines.

    The default implementation wraps the tealtiger package.
    Users can supply their own engine matching this protocol.
    """

    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        agent_name: str,
        policies: list["GovernancePolicy"],
        cumulative_cost: float,
    ) -> "GovernanceDecision":
        """Evaluate governance policies for a tool call."""
        ...
