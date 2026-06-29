# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.exceptions import missing_optional_dependency

try:
    from ..capabilities import (
        A2UI_CLIENT_CAPABILITIES_METADATA_KEY,
        A2UIClientCapabilities,
        parse_client_capabilities,
    )
    from .extension import get_a2ui_agent_extension, try_activate_a2ui_extension
    from .parts import create_a2ui_parts, get_a2ui_data, is_a2ui_part
except ImportError as e:
    get_a2ui_agent_extension = missing_optional_dependency(  # type: ignore[misc]
        "get_a2ui_agent_extension", "a2a", e
    )
    try_activate_a2ui_extension = missing_optional_dependency(  # type: ignore[misc]
        "try_activate_a2ui_extension", "a2a", e
    )
    create_a2ui_parts = missing_optional_dependency("create_a2ui_parts", "a2a", e)  # type: ignore[misc]
    get_a2ui_data = missing_optional_dependency("get_a2ui_data", "a2a", e)  # type: ignore[misc]
    is_a2ui_part = missing_optional_dependency("is_a2ui_part", "a2a", e)  # type: ignore[misc]
    A2UIClientCapabilities = missing_optional_dependency(  # type: ignore[misc]
        "A2UIClientCapabilities", "a2a", e
    )
    parse_client_capabilities = missing_optional_dependency(  # type: ignore[misc]
        "parse_client_capabilities", "a2a", e
    )
    A2UI_CLIENT_CAPABILITIES_METADATA_KEY = "a2uiClientCapabilities"

try:
    from .executor import A2UIAgentExecutor
except ImportError as e:
    A2UIAgentExecutor = missing_optional_dependency("A2UIAgentExecutor", "a2a", e)  # type: ignore[misc]

__all__ = (
    "A2UI_CLIENT_CAPABILITIES_METADATA_KEY",
    "A2UIAgentExecutor",
    "A2UIClientCapabilities",
    "create_a2ui_parts",
    "get_a2ui_agent_extension",
    "get_a2ui_data",
    "is_a2ui_part",
    "parse_client_capabilities",
    "try_activate_a2ui_extension",
)
