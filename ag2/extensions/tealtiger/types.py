# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
# SPDX-License-Identifier: Apache-2.0

"""Type definitions for TealTiger governance middleware."""

import re
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


# Minimum pattern confidence a prompt-injection finding needs to trigger a block.
DEFAULT_INJECTION_CONFIDENCE_THRESHOLD = 0.7


@dataclass
class GovernancePolicy:
    """A governance policy definition.

    Use the class methods to create specific policy types.
    """

    type: str
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def tool_allowlist(cls, allowed: list[str]) -> "GovernancePolicy":
        """Only allow tools matching these patterns (supports glob-style '*' suffix).

        Args:
            allowed: List of tool name patterns. Use '*' suffix for prefix matching.
        """
        return cls(type="tool_allowlist", config={"allowed": allowed})

    @classmethod
    def pii_block(cls, categories: list[str] | None = None) -> "GovernancePolicy":
        """Block tool calls containing PII in arguments.

        Args:
            categories: PII categories to detect. Default: ["ssn", "credit_card", "email", "phone"].
        """
        return cls(
            type="pii_block",
            config={"categories": categories or ["ssn", "credit_card", "email", "phone"]},
        )

    @classmethod
    def secret_detection(cls) -> "GovernancePolicy":
        """Block tool calls containing API keys/tokens in arguments."""
        return cls(type="secret_detection", config={})

    @classmethod
    def cost_limit(cls, max_per_session: float) -> "GovernancePolicy":
        """Deny tool calls once cumulative session cost exceeds the limit.

        Args:
            max_per_session: Maximum allowed cost in USD per session.
        """
        return cls(type="cost_limit", config={"max_per_session": max_per_session})

    @classmethod
    def prompt_injection_block(
        cls,
        techniques: list[str] | None = None,
        confidence_threshold: float = DEFAULT_INJECTION_CONFIDENCE_THRESHOLD,
    ) -> "GovernancePolicy":
        """Block tool calls containing prompt injection patterns in arguments.

        Args:
            techniques: Technique categories to detect. Default: all of
                `INJECTION_TECHNIQUES` (instruction_override, role_manipulation,
                context_manipulation, encoding_evasion, multi_turn_assembly).
            confidence_threshold: Minimum pattern confidence (0.0-1.0) for a finding to
                trigger a block. This selects *which patterns run* — each pattern carries a
                fixed confidence — it is not a score computed per match. Default: 0.7.

        Raises:
            ValueError: If `techniques` is empty or names an unknown technique, or if
                `confidence_threshold` falls outside 0.0-1.0. Both would otherwise leave
                the policy silently detecting nothing.
        """
        selected = list(INJECTION_TECHNIQUES) if techniques is None else list(techniques)
        if not selected:
            raise ValueError("`techniques` must not be empty; omit it to enable every technique.")
        unknown = [technique for technique in selected if technique not in INJECTION_PATTERNS]
        if unknown:
            raise ValueError(
                f"Unknown prompt injection technique(s): {', '.join(unknown)}. "
                f"Valid techniques: {', '.join(INJECTION_TECHNIQUES)}."
            )
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(f"`confidence_threshold` must be between 0.0 and 1.0, got {confidence_threshold}.")

        return cls(
            type="prompt_injection_block",
            config={"techniques": selected, "confidence_threshold": confidence_threshold},
        )


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


# ── Prompt injection patterns ────────────────────────────────────────────────


@dataclass(frozen=True)
class InjectionPattern:
    """A compiled prompt-injection regex together with its technique and confidence."""

    technique: str
    name: str
    pattern: re.Pattern[str]
    confidence: float


@dataclass(frozen=True)
class InjectionFinding:
    """One prompt-injection pattern match found in scanned text."""

    technique: str
    pattern_name: str
    matched_text: str
    confidence: float
    start: int
    end: int


# Pattern definitions organized by technique category.
# Each tuple: (name, regex, confidence)
#
# Patterns are deliberately anchored on injection *framing* (imperatives, chat-template
# markers, possessive references to the model's own context) rather than on keywords
# alone: a keyword-only pattern such as r"\bDAN\b" matches every person named Dan and
# makes the policy unusable in ENFORCE mode. Case-scoped groups `(?i:...)` are used where
# only part of a pattern may vary in case — `DAN` is the literal jailbreak acronym and is
# matched case-sensitively.
_INJECTION_PATTERN_DEFS: dict[str, list[tuple[str, str, float]]] = {
    "instruction_override": [
        (
            "ignore_previous",
            r"(?i)\b(?:ignore|disregard|forget|override|bypass)\b.{0,30}\b(?:previous|above|prior|earlier|all|system)\b.{0,20}\b(?:instructions?|prompts?|rules?|guidelines?|constraints?)",
            0.95,
        ),
        (
            "new_instructions",
            r"(?i)\b(?:new|updated|revised|real|actual|true)\s+(?:system\s+)?(?:instructions?|rules?|directives?|prompts?)\s*[:=]"
            r"|\byour\s+(?:new|real|actual|true)\s+(?:instructions?|rules?|directives?|system\s*prompt)\b"
            r"|\b(?:here\s+(?:are|is)|these\s+are)\s+(?:your\s+)?(?:new|updated|revised|real|actual|true)\s+(?:instructions?|rules?|directives?)\b",
            0.85,
        ),
        (
            "do_not_follow",
            r"(?i)\bdo\s+not\s+follow\b.{0,30}\b(?:instructions?|rules?|guidelines?|prompts?)",
            0.90,
        ),
        (
            "reset_context",
            r"(?i)\b(?:reset|clear|wipe|erase|flush)\b\s+(?:all\s+(?:of\s+)?)?your\s+(?:previous\s+|prior\s+|entire\s+)?(?:context|memory|conversation|chat|history|instructions?|system\s*prompt)\b",
            0.80,
        ),
        (
            "system_prompt_override",
            r"(?i)\b(?:system\s*prompt|system\s*message)\s*[:=]\s*",
            0.90,
        ),
    ],
    "role_manipulation": [
        (
            "dan_jailbreak",
            r"(?i:\b(?:you\s+are|act\s+as|acting\s+as|pretend\s+(?:to\s+be|you(?:'re|\s+are))|roleplay\s+as|enable|activate|enter)\b.{0,20})\bDAN\b"
            r"|\bDAN\b(?i:.{0,40}\b(?:mode|jailbreak|prompt|anything\s+now|no\s+restrictions?)\b)"
            r"|(?i:\bdo\s+anything\s+now\b.{0,25}\b(?:mode|prompt|jailbreak|no\s+restrictions?)\b)",
            0.95,
        ),
        (
            "developer_mode",
            r"(?i)\b(?:developer|dev)\s+mode\b.{0,20}\b(?:enabled?|activated?|on|unlocked?)",
            0.90,
        ),
        (
            "persona_switch",
            r"(?i)\b(?:you\s+are(?:\s+now)?|act\s+as|pretend\s+(?:to\s+be|you(?:'re|\s+are))|roleplay\s+as|assume\s+the\s+role\s+of)\s+(?:an?\s+|the\s+)?(?:unrestricted|uncensored|unfiltered|evil|malicious|amoral|lawless)\b"
            r"|\b(?:you\s+are\s+now|act\s+as|pretend\s+(?:to\s+be|you(?:'re|\s+are))|roleplay\s+as|assume\s+the\s+role)\b.{0,40}\b(?:unrestricted|uncensored|unfiltered|evil|malicious|without\s+(?:restrictions?|limits?|rules?|safety|filters?))\b.{0,25}\b(?:ai|assistant|model|chatbot|bot|persona|mode|version|gpt|llm)\b",
            0.90,
        ),
        (
            "character_override",
            r"(?i)\b(?:from\s+now\s+on|henceforth|going\s+forward)\b.{0,30}\b(?:you\s+(?:are|will|must|should)|your\s+(?:new|real)\s+(?:role|purpose|personality))",
            0.85,
        ),
        (
            "jailbreak_keyword",
            r"(?i)\b(?:jailbreak|jail\s*break|uncensor|unleash|unshackle|liberate)\b.{0,30}\b(?:mode|yourself|ai|model|assistant|restrictions?)",
            0.95,
        ),
    ],
    "context_manipulation": [
        (
            "delimiter_injection",
            r"(?i)(?:---+\s*(?:END|BEGIN)\s*(?:SYSTEM|USER|ASSISTANT|CONTEXT|INSTRUCTIONS?)?\s*---+)",
            0.85,
        ),
        (
            # A bare <system>…</system> element is ordinary data in many tool arguments, so
            # the tag only counts as injection when instruction-like content follows it.
            "xml_tag_injection",
            r"</?(?:system|instructions?|context|rules?|prompt|override|admin)(?:\s[^>]*)?>\s*"
            r"(?=[^<]{0,120}(?i:\b(?:ignore|disregard|forget|override|reveal|jailbreak|evil|malicious|instructions?|rules?|new|you\s+are|you\s+must|no\s+restrictions?)\b))"
            r"|<\|(?:im_start|im_end|system|endoftext)\|>",
            0.85,
        ),
        (
            "markdown_fence_injection",
            r"```(?:system|instructions?|rules?|prompt|override)\b",
            0.80,
        ),
        (
            "fake_system_message",
            r"(?i)\[(?:SYSTEM|ADMIN|INTERNAL|HIDDEN)\]\s*[:>]",
            0.90,
        ),
    ],
    "encoding_evasion": [
        (
            "base64_payload",
            r"(?i)\b(?:decode|base64|b64decode|atob)\s*\(\s*['\"]?[A-Za-z0-9+/=]{40,}",
            0.75,
        ),
        (
            # `\xNN` escapes repeat the backslash, so they need their own alternative —
            # the contiguous-hex form only ever matches `hex`/`0x` prefixed blobs.
            "hex_encoding",
            r"(?i)(?:\\{1,2}x[0-9a-f]{2}){8,}|\b(?:hex|0x)(?:[0-9a-f]{2}){10,}",
            0.70,
        ),
        (
            "unicode_escape",
            r"(?:\\{1,2}u[0-9a-fA-F]{4}){5,}",
            0.70,
        ),
        (
            # ROT13 ciphertext is word-shaped, so this looks for a run of cipher-looking
            # words rather than one long letter run. Confidence 0.65 is below
            # DEFAULT_INJECTION_CONFIDENCE_THRESHOLD: inactive unless the caller lowers it.
            "rot13_reference",
            r"(?i)\b(?:rot13|caesar\s+cipher|decode\s+this)\b.{0,50}(?:\b[a-zA-Z]{4,}\b\W+){2,}",
            0.65,
        ),
    ],
    "multi_turn_assembly": [
        (
            "payload_split_instruction",
            r"(?i)\b(?:combine|concatenate|join|merge|assemble)\b.{0,30}\b(?:previous|above|parts?|pieces?|fragments?|segments?)\b.{0,20}\b(?:instructions?|messages?|responses?|outputs?)",
            0.80,
        ),
        (
            "continuation_attack",
            r"(?i)\b(?:continue|resume|proceed)\b.{0,20}\b(?:from\s+where|the\s+previous|my\s+earlier|last\s+(?:message|response))\b.{0,40}\b(?:ignore|without|skip)\b.{0,20}\b(?:safety|filters?|restrictions?|rules?)",
            0.85,
        ),
    ],
}

# Compiled once at import; a comprehension so no loop variables leak into the module.
INJECTION_PATTERNS: dict[str, list[InjectionPattern]] = {
    technique: [
        InjectionPattern(technique=technique, name=name, pattern=re.compile(regex), confidence=confidence)
        for name, regex, confidence in pattern_defs
    ]
    for technique, pattern_defs in _INJECTION_PATTERN_DEFS.items()
}

INJECTION_TECHNIQUES: tuple[str, ...] = tuple(INJECTION_PATTERNS)
