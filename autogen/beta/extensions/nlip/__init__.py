# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.exceptions import missing_optional_dependency

try:
    from .config import NlipConfig
except ImportError as e:
    NlipConfig = missing_optional_dependency("NlipConfig", "nlip", e)  # type: ignore[misc]

try:
    from .server import NlipServer
except ImportError as e:
    NlipServer = missing_optional_dependency("NlipServer", "nlip", e)  # type: ignore[misc]

__all__ = (
    "NlipConfig",
    "NlipServer",
)
