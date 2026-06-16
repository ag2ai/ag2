# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Public A2UI surface.

Only the broadly-reusable, user-facing types live here: the agent, the action
declarations (and the ``@a2ui_action`` decorator), client capabilities, and the
stream events. Advanced/internal pieces — the parser, schema manager, validation
middleware, inbound-wire parse types, and the prompt-synthesis helpers — are
imported directly from their submodules (e.g. ``autogen.beta.a2ui.incoming``,
``autogen.beta.a2ui.parser``) when needed.
"""

from autogen.beta.exceptions import missing_optional_dependency

try:
    from .action_tool import a2ui_action
    from .actions import A2UIEventAction, A2UIFunctionCallAction
    from .agent import A2UIAgent
    from .capabilities import A2UIClientCapabilities
    from .events import A2UIMessageEvent, A2UIValidationFailedEvent
except ImportError as e:
    A2UIAgent = missing_optional_dependency("A2UIAgent", "a2ui", e)  # type: ignore[misc]
    a2ui_action = missing_optional_dependency("a2ui_action", "a2ui", e)  # type: ignore[misc]
    A2UIEventAction = missing_optional_dependency("A2UIEventAction", "a2ui", e)  # type: ignore[misc]
    A2UIFunctionCallAction = missing_optional_dependency("A2UIFunctionCallAction", "a2ui", e)  # type: ignore[misc]
    A2UIClientCapabilities = missing_optional_dependency("A2UIClientCapabilities", "a2ui", e)  # type: ignore[misc]
    A2UIMessageEvent = missing_optional_dependency("A2UIMessageEvent", "a2ui", e)  # type: ignore[misc]
    A2UIValidationFailedEvent = missing_optional_dependency("A2UIValidationFailedEvent", "a2ui", e)  # type: ignore[misc]

__all__ = (
    "A2UIAgent",
    "A2UIClientCapabilities",
    "A2UIEventAction",
    "A2UIFunctionCallAction",
    "A2UIMessageEvent",
    "A2UIValidationFailedEvent",
    "a2ui_action",
)
