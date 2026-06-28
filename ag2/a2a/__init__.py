# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.exceptions import missing_additional_dependency, missing_optional_dependency

try:
    from .card import build_card
    from .config import A2AConfig
except ImportError as e:
    build_card = missing_optional_dependency("build_card", "a2a", e)  # type: ignore[misc]
    A2AConfig = missing_optional_dependency("A2AConfig", "a2a", e)  # type: ignore[misc]

try:
    from .server import A2AServer
except ImportError as e:
    A2AServer = missing_additional_dependency("A2AServer", "a2a-sdk[http-server]", e)  # type: ignore[misc]

__all__ = (
    "A2AConfig",
    "A2AServer",
    "build_card",
)
