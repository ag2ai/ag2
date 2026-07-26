# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.exceptions import missing_additional_dependency

try:
    from .config import NlipConfig
except ImportError as e:
    NlipConfig = missing_additional_dependency("NlipConfig", 'nlip-sdk>=0.1.0,<1" "nlip-server>=0.1.3,<1', e)  # type: ignore[misc]

try:
    from .server import NlipServer
except ImportError as e:
    NlipServer = missing_additional_dependency("NlipServer", 'nlip-sdk>=0.1.0,<1" "nlip-server>=0.1.3,<1', e)  # type: ignore[misc]

__all__ = (
    "NlipConfig",
    "NlipServer",
)
