# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""ATR (Agent Threat Rules) guardrail capability.

Hooks into an agent's ``safeguard_tool_outputs`` and ``safeguard_llm_inputs``
hookable methods to scan tool results and outgoing LLM messages against the
ATR open detection ruleset. Matching is delegated to the ``pyatr`` engine (the
reference ATR evaluator), so the full ATR rule semantics -- multi-field
conditions, any/all logic, severity, categories -- are honoured rather than
re-implemented. ATR is MIT-licensed
(https://github.com/Agent-Threat-Rule/agent-threat-rules); ``pyatr`` is an
optional dependency and this capability degrades gracefully to a no-op when it
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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ....import_utils import optional_import_block
from ...assistant_agent import ConversableAgent
from .agent_capability import AgentCapability

with optional_import_block():
    from pyatr.engine import ATREngine
    from pyatr.types import AgentEvent

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

# Maps each hookable method to the ATR (event_type, field) the content is
# evaluated as, so rules keyed on a specific surface still fire.
_HOOK_EVENT: dict[str, tuple[str, str]] = {
    "tool_output": ("tool_response", "tool_response"),
    "llm_input": ("llm_input", "user_input"),
}


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


class ATRGuardrail(AgentCapability):
    """Composable capability that screens tool outputs and outgoing LLM
    messages against ATR rules, evaluated by the ``pyatr`` engine.

    Subscribes to two ``ConversableAgent`` hookable methods:

    - ``safeguard_tool_outputs`` -- invoked after every tool/function call; the
      serialised tool response is evaluated as an ATR ``tool_response`` event.
    - ``safeguard_llm_inputs`` -- invoked before sending messages to the LLM;
      the most recent message content is evaluated as an ATR ``llm_input``
      event.

    Matches are recorded on ``self.matches`` and forwarded to the optional
    ``on_match`` callback. In ``action="block"`` mode a match on
    ``safeguard_llm_inputs`` returns ``None`` so the core hook chain halts the
    send; tool outputs are not blocked (they have already executed) but the
    matched content is redacted.
    """

    def __init__(
        self,
        action: str = _DEFAULT_ACTION,
        min_severity: str = "low",
        *,
        rules_dir: str | Path | None = None,
        engine: "ATREngine | None" = None,
        on_match: Callable[[ATRMatch], None] | None = None,
    ) -> None:
        """Args:
        action: One of ``"allow"`` (record only), ``"warn"`` (log and record),
            or ``"block"`` (record, redact tool outputs, drop LLM inputs).
        min_severity: Lowest severity to act on. One of ``info``, ``low``,
            ``medium``, ``high``, ``critical``.
        rules_dir: Optional directory of ATR rule YAML files. When omitted, the
            rules bundled with ``pyatr`` (>= 0.2.6) are used.
        engine: Optional pre-built :class:`pyatr.engine.ATREngine`, primarily
            for tests or pinning a specific ruleset; overrides ``rules_dir``.
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
        self._engine = engine if engine is not None else _build_engine(rules_dir)
        self.matches: list[ATRMatch] = []

    # ------------------------------------------------------------- capability

    def add_to_agent(self, agent: ConversableAgent) -> None:
        """Register the guardrail's hooks on the given agent."""
        agent.register_hook(hookable_method="safeguard_tool_outputs", hook=self._on_tool_output)
        agent.register_hook(hookable_method="safeguard_llm_inputs", hook=self._on_llm_input)

    # ----------------------------------------------------------------- hooks

    def _on_tool_output(self, response: dict[str, Any]) -> dict[str, Any]:
        """Scan a tool response. Returns the original dict, or a redacted copy
        in ``block`` mode (the tool has already executed, so it is not blocked)."""
        if self._engine is None:
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
        """Scan the most recent LLM input message. Returns ``None`` in ``block``
        mode on match (halting the send), otherwise the original list."""
        if self._engine is None or not messages:
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
        if not text or self._engine is None:
            return None
        event_type, field = _HOOK_EVENT[hook]
        try:
            results = self._engine.evaluate(
                AgentEvent(content=text, event_type=event_type, fields={field: text})
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("ATR evaluation failed on %s: %s", hook, exc)
            return None
        # pyatr returns matches sorted by severity (critical first); act on the
        # highest-severity match at or above the configured floor.
        for m in results:
            severity = (m.severity or "medium").lower()
            if _SEVERITY_ORDER.get(severity, _SEVERITY_ORDER["medium"]) < self._severity_floor:
                continue
            tags = getattr(m, "tags", None) or {}
            match = ATRMatch(
                rule_id=m.rule_id,
                severity=severity,
                category=tags.get("category", "uncategorised"),
                hook=hook,
                snippet=" ".join(text.split())[:120],
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


def _build_engine(rules_dir: str | Path | None) -> "ATREngine | None":
    """Build the default ATR engine, or ``None`` when ``pyatr`` is absent."""
    try:
        engine = ATREngine()
    except NameError:
        logger.debug("pyatr is not installed; ATRGuardrail will be a no-op.")
        return None
    if rules_dir is not None:
        loaded = engine.load_rules_from_directory(rules_dir)
    elif hasattr(engine, "load_default_rules"):
        loaded = engine.load_default_rules()
    else:  # pyatr < 0.2.6 has no bundled rules
        logger.warning(
            "pyatr>=0.2.6 is required for the bundled rule set; found an older "
            "version. Pass rules_dir or upgrade pyatr; ATRGuardrail is a no-op."
        )
        loaded = 0
    if loaded == 0:
        logger.warning(
            "ATRGuardrail loaded 0 ATR rules and will be a no-op. Install "
            "pyatr>=0.2.6 (which bundles the rule set) or pass rules_dir."
        )
    return engine
