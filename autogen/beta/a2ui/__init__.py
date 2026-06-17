# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Public A2UI surface.

A2UI is a transport over a plain ``autogen.beta.Agent`` (mirroring A2A / AG-UI):
the agent stays a normal ``Agent`` and A2UI behaviour is applied by a transport
wrapper — :class:`~autogen.beta.a2ui.rest.A2UIServer` (REST/SSE) or
:class:`~autogen.beta.a2ui.a2a.A2UIAgentExecutor` (A2A) — configured with flat
A2UI kwargs.

Only the broadly-reusable, user-facing surface lives here: the ``@a2ui_action``
decorator (clickable buttons, passed to ``Agent(tools=[...])``), client
capabilities, and the stream events. Advanced/internal pieces — the parser,
schema manager, validation middleware, inbound-wire parse types, and
prompt-synthesis helpers — are imported directly from their submodules (e.g.
``autogen.beta.a2ui.incoming``) when needed.
"""

from autogen.beta.exceptions import missing_optional_dependency

try:
    from .action_tool import a2ui_action
    from .capabilities import A2UIClientCapabilities
    from .events import A2UIClientEvent, A2UIMessageEvent, A2UIValidationFailedEvent
except ImportError as e:
    a2ui_action = missing_optional_dependency("a2ui_action", "a2ui", e)  # type: ignore[misc]
    A2UIClientCapabilities = missing_optional_dependency("A2UIClientCapabilities", "a2ui", e)  # type: ignore[misc]
    A2UIClientEvent = missing_optional_dependency("A2UIClientEvent", "a2ui", e)  # type: ignore[misc]
    A2UIMessageEvent = missing_optional_dependency("A2UIMessageEvent", "a2ui", e)  # type: ignore[misc]
    A2UIValidationFailedEvent = missing_optional_dependency("A2UIValidationFailedEvent", "a2ui", e)  # type: ignore[misc]

__all__ = (
    "A2UIClientCapabilities",
    "A2UIClientEvent",
    "A2UIMessageEvent",
    "A2UIValidationFailedEvent",
    "a2ui_action",
)
