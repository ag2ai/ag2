# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Literal

from ._types import JsonValue


@dataclass(slots=True, frozen=True)
class A2UIEventAction:
    """A **server** ``event`` action — dispatches a named event to the server.

    The ``name`` matches the button's ``event.name``. The click is either routed
    to a registered tool (``tool_name``) or handled by the LLM via ``description``.

    Args:
        name: Action identifier; matches the ``event.name`` in the button's
            action definition.
        tool_name: Name of a registered tool to call when this action fires.
            If ``None``, the click is passed to the LLM as a prompt using
            ``description``.
        description: Human-readable description, injected into the system prompt
            so the LLM knows the action exists.
        example_context: Example ``event.context`` dict shown to the LLM. For
            tool-mapped actions this should match the tool's parameter names.

    Example::

        # Server action routed to a tool
        A2UIEventAction(
            name="schedule_2pm",
            tool_name="schedule_posts",
            description="Schedule all posts for 2:00 PM",
            example_context={"time": "2:00 PM"},
        )

        # Server action handled by the LLM
        A2UIEventAction(
            name="rewrite_previews",
            description="Regenerate all previews with a different creative angle",
        )
    """

    name: str
    tool_name: str | None = None
    description: str = ""
    example_context: dict[str, JsonValue] | None = None

    # Discriminator as a fixed instance field (``init=False`` so callers can't
    # set it). Unlike a ``ClassVar``, a ``Literal`` instance field lets type
    # checkers narrow the union on ``action.action_type == "event"``, while the
    # value stays pinned by the type so it can never disagree with the class.
    action_type: Literal["event"] = field(default="event", init=False)


@dataclass(slots=True, frozen=True)
class A2UIFunctionCallAction:
    """A **client** ``functionCall`` action — runs a client-side function.

    Executes on the client without a server round-trip (e.g. ``openUrl``). The
    ``name`` is the client function name and ``example_args`` shows the expected
    arguments. There is no server tool body, so these are declared directly
    rather than via :func:`a2ui_action`.

    Args:
        name: The client-side function name (e.g. ``"openUrl"``).
        description: Human-readable description, injected into the system prompt.
        example_args: Example ``functionCall.args`` dict shown to the LLM.

    Example::

        A2UIFunctionCallAction(
            name="openUrl",
            description="Open a URL in the user's browser",
            example_args={"url": "https://example.com"},
        )
    """

    name: str
    description: str = ""
    example_args: dict[str, JsonValue] | None = None

    action_type: Literal["functionCall"] = field(default="functionCall", init=False)


# An A2UI action registered on an ``A2UIAgent``. A tagged union so the two modes
# carry only their own fields (an event can't hold ``example_args``, a
# functionCall can't hold ``tool_name``); branch with ``isinstance`` or the
# ``action_type`` discriminator.
A2UIAction = A2UIEventAction | A2UIFunctionCallAction
