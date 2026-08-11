# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Public A2UI surface: the :class:`A2UIServer` ASGI app, the ``@a2ui_action``
decorator, client capabilities, and stream events. Pick a wire transport from
the ``transports`` submodule (or use the ``a2a`` submodule for A2A).
"""

from ag2.exceptions import missing_additional_dependency, missing_optional_dependency

try:
    from .actions import A2UIAction, a2ui_action
    from .capabilities import A2UIClientCapabilities
    from .events import A2UIClientEvent, A2UIMessageEvent, A2UIValidationFailedEvent
except ImportError as e:
    a2ui_action = missing_optional_dependency("a2ui_action", "a2ui", e)  # type: ignore[misc]
    A2UIAction = missing_optional_dependency("A2UIAction", "a2ui", e)  # type: ignore[misc]
    A2UIClientCapabilities = missing_optional_dependency("A2UIClientCapabilities", "a2ui", e)  # type: ignore[misc]
    A2UIClientEvent = missing_optional_dependency("A2UIClientEvent", "a2ui", e)  # type: ignore[misc]
    A2UIMessageEvent = missing_optional_dependency("A2UIMessageEvent", "a2ui", e)  # type: ignore[misc]
    A2UIValidationFailedEvent = missing_optional_dependency("A2UIValidationFailedEvent", "a2ui", e)  # type: ignore[misc]

try:
    # Serving needs Starlette (and a transport); a missing install surfaces here.
    from .server import A2UIServer
except ImportError as e:  # pragma: no cover - exercised only without starlette
    A2UIServer = missing_additional_dependency("A2UIServer", "starlette>=0.40,<1", e)  # type: ignore[misc]

__all__ = (
    "A2UIAction",
    "A2UIClientCapabilities",
    "A2UIClientEvent",
    "A2UIMessageEvent",
    "A2UIServer",
    "A2UIValidationFailedEvent",
    "a2ui_action",
)
