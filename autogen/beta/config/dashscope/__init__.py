# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.import_utils import optional_import_block

with optional_import_block():
    from .config import DashScopeConfig
    from .dashscope_client import DashScopeClient

__all__ = (
    "DashScopeClient",
    "DashScopeConfig",
)
