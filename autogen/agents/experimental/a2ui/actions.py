# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class A2UIAction:
    """Defines an action that can be triggered by A2UI buttons.

    Actions are registered on an ``A2UIAgent`` and define how button clicks
    are handled. Each action has a name (matching the button's ``event.name``),
    an optional tool mapping, and a description for prompt injection.

    Args:
        name: Action identifier. Must match the ``event.name`` in the button's
            action definition.
        tool_name: Name of a registered tool/function on the agent to call
            when this action is triggered. If None, the action is passed to
            the LLM as a prompt using the description.
        description: Human-readable description of what this action does.
            Injected into the system prompt so the LLM knows what actions
            are available. Also used as context when routing LLM-handled actions.
        example_context: Example context dict showing what values the button
            should include. Injected into the system prompt so the LLM knows
            what to put in the action's ``context`` field. For tool-mapped
            actions, this should match the tool's parameter names.

    Example::

        A2UIAction(
            name="schedule_2pm",
            tool_name="schedule_posts",
            description="Schedule all posts for 2:00 PM",
            example_context={"time": "2:00 PM"},
        )

        A2UIAction(
            name="rewrite_previews",
            description="Regenerate all previews with a different creative angle",
        )
    """

    name: str
    tool_name: str | None = None
    description: str = ""
    example_context: dict[str, Any] | None = None
