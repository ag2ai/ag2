# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal
from uuid import UUID

from termcolor import colored

from ....events.base_event import BaseEvent, wrap_event

# Type for termcolor colors
TermColor = Literal[
    "black",
    "grey",
    "red",
    "green",
    "yellow",
    "blue",
    "magenta",
    "cyan",
    "light_grey",
    "dark_grey",
    "light_red",
    "light_green",
    "light_yellow",
    "light_blue",
    "light_magenta",
    "light_cyan",
    "white",
]


@wrap_event
class SafeguardEvent(BaseEvent):
    """Event for safeguard actions"""

    event_type: str  # e.g., "load", "check", "violation", "action"
    message: str
    source_agent: str | None = None
    target_agent: str | None = None
    guardrail_type: str | None = None
    action: str | None = None
    content_preview: str | None = None

    def __init__(
        self,
        *,
        uuid: UUID | None = None,
        event_type: str,
        message: str,
        source_agent: str | None = None,
        target_agent: str | None = None,
        guardrail_type: str | None = None,
        action: str | None = None,
        content_preview: str | None = None,
    ):
        super().__init__(
            uuid=uuid,
            event_type=event_type,
            message=message,
            source_agent=source_agent,
            target_agent=target_agent,
            guardrail_type=guardrail_type,
            action=action,
            content_preview=content_preview,
        )

    def print(self, f: Callable[..., Any] | None = None) -> None:
        f = f or print

        # Choose color based on event type
        color: TermColor = "green"
        if self.event_type == "load":
            color = "green"
        elif self.event_type == "check":
            color = "cyan"
        elif self.event_type == "violation":
            color = "red"
        elif self.event_type == "action":
            color = "yellow"

        # Choose emoji based on event type
        emoji = ""
        if self.event_type == "load":
            emoji = "✅"
        elif self.event_type == "check":
            emoji = "🔍"
        elif self.event_type == "violation":
            emoji = "🛡️"
        elif self.event_type == "action":
            if self.action == "block":
                emoji = "🚨"
            elif self.action == "mask":
                emoji = "🎭"
            elif self.action == "warning":
                emoji = "⚠️"
            else:
                emoji = "⚙️"

        # Create header based on event type (skip for load events)
        if self.event_type == "check":
            header = f"***** Safeguard Check: {self.message} *****"
            f(colored(header, color), flush=True)
        elif self.event_type == "violation":
            header = "***** Safeguard Violation: DETECTED *****"
            f(colored(header, color), flush=True)
        elif self.event_type == "action":
            header = f"***** Safeguard Enforcement Action: {self.action.upper() if self.action else 'APPLIED'} *****"
            f(colored(header, color), flush=True)

        # Format the output
        output_parts = [f"{emoji} {self.message}" if emoji else self.message]

        if self.source_agent and self.target_agent:
            output_parts.append(f"  • From: {self.source_agent}")
            output_parts.append(f"  • To: {self.target_agent}")
        elif self.source_agent:
            output_parts.append(f"  • Agent: {self.source_agent}")

        if self.guardrail_type:
            output_parts.append(f"  • Guardrail: {self.guardrail_type}")

        if self.action:
            output_parts.append(f"  • Action: {self.action}")

        if self.content_preview:
            # Replace actual newlines with \n for display
            content_display = self.content_preview.replace("\n", "\\n").replace("\r", "\\r")
            output_parts.append(f"  • Content: {content_display}")

        f(colored("\n".join(output_parts), color), flush=True)

        # Print footer with matching length (skip for load events)
        if self.event_type in ["check", "violation", "action"]:
            footer = "*" * len(header)
            f(colored(footer, color), flush=True)
