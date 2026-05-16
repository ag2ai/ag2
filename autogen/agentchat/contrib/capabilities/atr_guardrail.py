# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""ATR (Agent Threat Rules) guardrail capability.

Hooks into an agent's ``safeguard_tool_outputs`` and ``safeguard_llm_inputs``
hookable methods to scan tool results and outgoing LLM messages against the
ATR open detection ruleset. ATR is an MIT-licensed community ruleset
(https://github.com/Agent-Threat-Rule/agent-threat-rules); the ``pyatr``
package is an optional dependency and this module degrades gracefully if it
is not installed.

Usage::

    from autogen import AssistantAgent
    from autogen.agentchat.contrib.capabilities.atr_guardrail import ATRGuardrail

    agent = AssistantAgent(name="assistant", llm_config=llm_config)
    ATRGuardrail(action="warn", min_severity="medium").add_to_agent(agent)
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ....import_utils import optional_import_block
from ...assistant_agent import ConversableAgent
from .agent_capability import AgentCapability

with optional_import_block():
    import pyatr  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

# Severity ordering used for the optional threshold filter.
_SEVERITY_ORDER: dict[str, int] = {
    "info": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

_DEFAULT_ACTION = "warn"
_VALID_ACTIONS = ("allow", "warn", "block")


@dataclass(frozen=True)
class ATRMatch:
    """A single rule match emitted by the guardrail."""

    rule_id: str
    severity: str
    category: str
    hook: str  # "tool_output" or "llm_input"
    snippet: str
    action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "category": self.category,
            "hook": self.hook,
            "snippet": self.snippet,
            "action": self.action,
        }


@dataclass
class _CompiledRule:
    """Internal representation of an ATR rule reduced to a regex matcher."""

    rule_id: str
    severity: str
    category: str
    pattern: re.Pattern[str]


def _default_rule_loader() -> list[dict[str, Any]]:
    """Load rules from the optional ``pyatr`` package.

    Returns an empty list if ``pyatr`` is not installed, so the capability is
    a no-op rather than a hard failure when the optional dependency is
    missing.
    """
    try:
        # pyatr exposes a load_rules() helper in the published distribution.
        # We probe for a few plausible accessors to stay forward-compatible.
        if hasattr(pyatr, "load_rules"):
            return list(pyatr.load_rules())  # type: ignore[no-any-return]
        if hasattr(pyatr, "rules"):
            return list(pyatr.rules())  # type: ignore[no-any-return]
        if hasattr(pyatr, "Ruleset"):
            ruleset = pyatr.Ruleset()  # type: ignore[attr-defined]
            return list(getattr(ruleset, "rules", []))
    except NameError:
        # pyatr was not importable; optional dependency missing.
        logger.debug("pyatr not installed; ATRGuardrail will load zero rules.")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("ATR rule load failed: %s", exc)
    return []


def _compile_rule(raw: dict[str, Any]) -> _CompiledRule | None:
    """Compile a single ATR rule dict to a regex matcher.

    Rules without a usable pattern are skipped. Only ``pattern`` /
    ``regex`` / ``signature`` style fields are honoured here; richer ATR
    matchers (semantic, multi-stage) remain server-side in PanGuard.
    """
    rule_id = str(raw.get("id") or raw.get("rule_id") or "").strip()
    if not rule_id:
        return None
    pattern = raw.get("pattern") or raw.get("regex") or raw.get("signature")
    if not isinstance(pattern, str) or not pattern:
        return None
    try:
        compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    except re.error as exc:
        logger.debug("Skipping ATR rule %s (invalid regex): %s", rule_id, exc)
        return None
    severity = str(raw.get("severity") or "medium").lower()
    if severity not in _SEVERITY_ORDER:
        severity = "medium"
    category = str(raw.get("category") or "uncategorised")
    return _CompiledRule(rule_id=rule_id, severity=severity, category=category, pattern=compiled)


class ATRGuardrail(AgentCapability):
    """Composable capability that screens tool outputs and outgoing LLM
    messages against ATR rules.

    The capability subscribes to two ``ConversableAgent`` hookable methods:

    - ``safeguard_tool_outputs`` — invoked after every tool/function call;
      we scan the serialised tool response for matches.
    - ``safeguard_llm_inputs`` — invoked before sending messages to the LLM;
      we scan the most recent message content for matches.

    Matches are recorded on ``self.matches`` and forwarded to the optional
    ``on_match`` callback. In ``action="block"`` mode a match on
    ``safeguard_llm_inputs`` returns ``None`` so the core hook chain halts
    the send; tool outputs are not blocked (they have already executed) but
    the matched content is redacted in ``block`` mode.
    """

    def __init__(
        self,
        action: str = _DEFAULT_ACTION,
        min_severity: str = "low",
        rule_loader: Callable[[], Iterable[dict[str, Any]]] | None = None,
        on_match: Callable[[ATRMatch], None] | None = None,
    ) -> None:
        """Args:
        action: One of ``"allow"`` (record only), ``"warn"`` (log and record),
            or ``"block"`` (record, redact tool outputs, drop LLM inputs).
        min_severity: Lowest severity to act on. One of ``info``, ``low``,
            ``medium``, ``high``, ``critical``.
        rule_loader: Zero-argument callable returning an iterable of ATR
            rule dicts. Defaults to loading from the optional ``pyatr``
            package. Override in tests or to pin a specific ruleset.
        on_match: Optional callback fired once per match.
        """
        if action not in _VALID_ACTIONS:
            raise ValueError(f"action must be one of {_VALID_ACTIONS}, got {action!r}")
        if min_severity not in _SEVERITY_ORDER:
            raise ValueError(f"min_severity must be one of {tuple(_SEVERITY_ORDER)}, got {min_severity!r}")

        self.action = action
        self.min_severity = min_severity
        self._severity_floor = _SEVERITY_ORDER[min_severity]
        self._on_match = on_match
        self._loader = rule_loader or _default_rule_loader
        self.matches: list[ATRMatch] = []
        self._rules: list[_CompiledRule] = self._load_compiled_rules()

    # ------------------------------------------------------------------ rules

    def _load_compiled_rules(self) -> list[_CompiledRule]:
        compiled: list[_CompiledRule] = []
        try:
            raw_rules = list(self._loader())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ATR rule_loader raised %s; continuing with zero rules.", exc)
            return compiled
        for raw in raw_rules:
            if not isinstance(raw, dict):
                continue
            rule = _compile_rule(raw)
            if rule is not None and _SEVERITY_ORDER[rule.severity] >= self._severity_floor:
                compiled.append(rule)
        logger.info("ATRGuardrail loaded %d rules at or above severity %s.", len(compiled), self.min_severity)
        return compiled

    # ------------------------------------------------------------- capability

    def add_to_agent(self, agent: ConversableAgent) -> None:
        """Register the guardrail's hooks on the given agent."""
        agent.register_hook(hookable_method="safeguard_tool_outputs", hook=self._on_tool_output)
        agent.register_hook(hookable_method="safeguard_llm_inputs", hook=self._on_llm_input)

    # ----------------------------------------------------------------- hooks

    def _on_tool_output(self, response: dict[str, Any]) -> dict[str, Any]:
        """Scan a tool response. Returns the original dict, a redacted copy
        in ``block`` mode, or the unchanged dict in ``allow``/``warn`` mode.
        """
        if not self._rules:
            return response
        text = self._stringify(response.get("content"))
        match = self._scan(text, hook="tool_output")
        if match is None:
            return response
        if self.action == "block":
            redacted = dict(response)
            redacted["content"] = f"[ATR:{match.rule_id} redacted match of severity {match.severity}]"
            return redacted
        return response

    def _on_llm_input(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        """Scan the most recent LLM input message. Returns ``None`` in
        ``block`` mode on match (which halts the send), otherwise the
        original list.
        """
        if not self._rules or not messages:
            return messages
        last = messages[-1]
        text = self._stringify(last.get("content"))
        match = self._scan(text, hook="llm_input")
        if match is None:
            return messages
        if self.action == "block":
            return None
        return messages

    # --------------------------------------------------------------- scanning

    def _scan(self, text: str, hook: str) -> ATRMatch | None:
        if not text:
            return None
        for rule in self._rules:
            m = rule.pattern.search(text)
            if m is None:
                continue
            snippet = text[max(0, m.start() - 24) : min(len(text), m.end() + 24)]
            match = ATRMatch(
                rule_id=rule.rule_id,
                severity=rule.severity,
                category=rule.category,
                hook=hook,
                snippet=snippet,
                action=self.action,
            )
            self.matches.append(match)
            if self.action == "warn":
                logger.warning(
                    "ATR rule %s (severity=%s, category=%s) matched on %s",
                    match.rule_id,
                    match.severity,
                    match.category,
                    hook,
                )
            if self._on_match is not None:
                try:
                    self._on_match(match)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("ATRGuardrail on_match callback raised: %s", exc)
            return match
        return None

    @staticmethod
    def _stringify(content: Any) -> str:
        """Reduce a message ``content`` field to a single string for scanning."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            return "\n".join(p for p in parts if p)
        try:
            return json.dumps(content, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(content)
