# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""AG-UI adapter for A2UI: serve a plain :class:`~autogen.beta.Agent` over AG-UI
via :class:`A2UIAGUIServer` so CopilotKit's ``@copilotkit/a2ui-renderer`` renders
the agent's A2UI output. Requires ``ag2[ag-ui]`` (and Starlette to serve).
"""

from autogen.beta.exceptions import missing_optional_dependency

try:
    from .server import A2UIAGUIServer
except ImportError as e:  # pragma: no cover - exercised only without ag-ui-protocol
    A2UIAGUIServer = missing_optional_dependency("A2UIAGUIServer", "ag-ui", e)  # type: ignore[misc]

__all__ = ("A2UIAGUIServer",)
