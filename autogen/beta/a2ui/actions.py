# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Literal

from ._types import JsonValue


@dataclass(slots=True, frozen=True)
class A2UIEventAction:
    """A **server** ``event`` action — the declaration behind a clickable button.

    Produced internally by the :func:`a2ui_action` decorator (it is not meant to
    be constructed directly): each ``@a2ui_action`` tool carries one, with
    ``tool_name`` set to its own name. It is used to (1) describe the button in
    the system prompt and (2) route an incoming click back to the tool.

    Args:
        name: Action identifier; matches the ``event.name`` in the button's
            action definition (and the tool's name).
        tool_name: Name of the registered tool to call when this action fires.
        description: Human-readable description, injected into the system prompt
            so the LLM knows the action exists.
        example_context: Example ``event.context`` dict shown to the LLM,
            matching the tool's parameter names.
    """

    name: str
    tool_name: str | None = None
    description: str = ""
    example_context: dict[str, JsonValue] | None = None

    # Discriminator as a fixed instance field (``init=False`` so callers can't
    # set it), kept for forward-compatibility with additional action kinds.
    action_type: Literal["event"] = field(default="event", init=False)
