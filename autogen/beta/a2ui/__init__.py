# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Public A2UI surface: the ``@a2ui_action`` decorator, client capabilities,
and stream events. Serve an agent over A2UI via the transport wrappers in the
``rest`` and ``a2a`` submodules.
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
