# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.exceptions import missing_additional_dependency

try:
    from ..capabilities import (
        A2UI_CLIENT_CAPABILITIES_METADATA_KEY,
        A2UIClientCapabilities,
        parse_client_capabilities,
    )
    from .extension import get_a2ui_agent_extension, try_activate_a2ui_extension
    from .parts import create_a2ui_parts, get_a2ui_data, is_a2ui_part
except ImportError as e:
    get_a2ui_agent_extension = missing_additional_dependency(  # type: ignore[misc]
        "get_a2ui_agent_extension", "a2a-sdk", e
    )
    try_activate_a2ui_extension = missing_additional_dependency(  # type: ignore[misc]
        "try_activate_a2ui_extension", "a2a-sdk", e
    )
    create_a2ui_parts = missing_additional_dependency("create_a2ui_parts", "a2a-sdk", e)  # type: ignore[misc]
    get_a2ui_data = missing_additional_dependency("get_a2ui_data", "a2a-sdk", e)  # type: ignore[misc]
    is_a2ui_part = missing_additional_dependency("is_a2ui_part", "a2a-sdk", e)  # type: ignore[misc]
    A2UIClientCapabilities = missing_additional_dependency(  # type: ignore[misc]
        "A2UIClientCapabilities", "a2a-sdk", e
    )
    parse_client_capabilities = missing_additional_dependency(  # type: ignore[misc]
        "parse_client_capabilities", "a2a-sdk", e
    )
    A2UI_CLIENT_CAPABILITIES_METADATA_KEY = "a2uiClientCapabilities"

try:
    from .executor import A2UIAgentExecutor
except ImportError as e:
    A2UIAgentExecutor = missing_additional_dependency("A2UIAgentExecutor", "a2a-sdk", e)  # type: ignore[misc]

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
