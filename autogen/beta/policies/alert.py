# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""AlertPolicy — delivers observer alerts to the LLM via assembly.

Replaces the old Signal delivery system (SignalPolicy + SignalInjectionMiddleware).
AlertPolicy reads ObserverAlert events from the event stream and injects them
into the LLM prompt. FATAL alerts emit a HaltEvent on the stream.

Key behaviors preserved from the old system:
- Deduplication: each alert is injected once, then marked as delivered
- Cross-turn accumulation: reads alerts from stream history across LLM calls
- FATAL handling: emits HaltEvent for observers to halt execution
"""

from autogen.beta.context import ConversationContext as Context
from autogen.beta.events import BaseEvent
from autogen.beta.events.alert import HaltEvent, ObserverAlert, Severity


class AlertPolicy:
    """Injects observer alerts into the LLM prompt.

    Reads ObserverAlert events from the event list, formats them as
    prompt text, and tracks which alerts have been delivered to avoid
    duplicates across LLM calls.

    Deduplication keys on ``(source, severity, message)`` — the semantic
    content of the alert, not the Python object identity — so dedup
    survives history replay, compaction (which rewrites history into
    fresh event objects), and serialization round-trips. Observers that
    need to re-alert with identical wording can vary the ``data`` dict
    or the ``message`` string.

    For FATAL alerts, emits a HaltEvent on the stream and adds a halt
    note to the prompt.

    Ordering: this is an injection policy. Place it after other injection
    policies (working memory, episodic memory) and before reduction policies
    (sliding window, token budget).
    """

    name = "alert"

    def __init__(self) -> None:
        self._delivered_keys: set[tuple[str, str, str]] = set()

    async def apply(
        self,
        prompts: list[str],
        events: list[BaseEvent],
        context: Context,
    ) -> tuple[list[str], list[BaseEvent]]:
        # Collect undelivered alerts from the event stream
        new_alerts: list[ObserverAlert] = []
        for event in events:
            if isinstance(event, ObserverAlert):
                key = (event.source, str(event.severity), event.message)
                if key not in self._delivered_keys:
                    new_alerts.append(event)
                    self._delivered_keys.add(key)

        if not new_alerts:
            return prompts, events

        # Split fatal vs non-fatal
        fatal = [a for a in new_alerts if a.severity == Severity.FATAL]
        non_fatal = [a for a in new_alerts if a.severity != Severity.FATAL]

        # Inject non-fatal alerts as prompt text
        if non_fatal:
            alert_text = self._format_alerts(non_fatal)
            prompts = prompts + [alert_text]

        # FATAL: emit HaltEvent on the stream
        if fatal:
            await context.send(
                HaltEvent(
                    reason=f"FATAL: {fatal[0].message}",
                    source=fatal[0].source,
                    alerts=fatal,
                )
            )
            # Add halt note to prompt so LLM sees it if the call isn't short-circuited
            prompts = prompts + [f"[FATAL ALERT] ({fatal[0].source}): {fatal[0].message}. Execution halting."]

        return prompts, events

    @staticmethod
    def _format_alerts(alerts: list[ObserverAlert]) -> str:
        lines = ["[OBSERVER MONITORING ALERTS]"]
        for a in alerts:
            level = a.severity.upper() if isinstance(a.severity, str) else str(a.severity)
            lines.append(f"- [{level}] ({a.source}): {a.message}")
        return "\n".join(lines)
