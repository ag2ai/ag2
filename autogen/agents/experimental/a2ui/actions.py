# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class A2UIAction:
    """Defines an action that can be triggered by A2UI buttons.

    Actions are registered on an ``A2UIAgent`` and define how button clicks
    are handled. There are two action types matching the A2UI spec:

    - **event** (server action): Dispatches a named event to the server.
      The ``name`` field matches the button's ``event.name``. Optionally
      routed to a tool via ``tool_name``, or handled by the LLM.
    - **functionCall** (client action): Executes a client-side function
      (e.g., ``openUrl``) without server communication. The ``name`` field
      is the function name and ``example_args`` shows the expected arguments.

    Args:
        name: Action identifier. For ``"event"`` actions, this matches the
            ``event.name`` in the button's action definition. For
            ``"functionCall"`` actions, this is the client-side function name.
        action_type: The action type — ``"event"`` for server actions or
            ``"functionCall"`` for client-side actions. Default is ``"event"``.
        tool_name: Name of a registered tool/function on the agent to call
            when this action is triggered. Only applicable to ``"event"``
            actions. If None, the action is passed to the LLM as a prompt
            using the description.
        description: Human-readable description of what this action does.
            Injected into the system prompt so the LLM knows what actions
            are available.
        example_context: Example context dict for ``"event"`` actions, showing
            what values the button should include in ``event.context``. For
            tool-mapped actions, this should match the tool's parameter names.
        example_args: Example args dict for ``"functionCall"`` actions, showing
            what values the button should include in ``functionCall.args``.

    Example::

        # Server action routed to a tool
        A2UIAction(
            name="schedule_2pm",
            tool_name="schedule_posts",
            description="Schedule all posts for 2:00 PM",
            example_context={"time": "2:00 PM"},
        )

        # Server action handled by the LLM
        A2UIAction(
            name="rewrite_previews",
            description="Regenerate all previews with a different creative angle",
        )

        # Client-side action
        A2UIAction(
            name="openUrl",
            action_type="functionCall",
            description="Open a URL in the user's browser",
            example_args={"url": "https://example.com"},
        )
    """

    name: str
    action_type: Literal["event", "functionCall"] = "event"
    tool_name: str | None = None
    description: str = ""
    example_context: dict[str, Any] | None = None
    example_args: dict[str, Any] | None = None
