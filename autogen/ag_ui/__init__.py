# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
try:
    import ag_ui as ag_ui
except ImportError as e:
    raise ImportError("ag-ui-protocol is not installed. Please install it with:\npip install ag2[ag-ui]") from e

from .adapter import AGUIStream

__all__ = ("AGUIStream",)
