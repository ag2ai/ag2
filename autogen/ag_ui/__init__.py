# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

try:
    from ag_ui.core import RunAgentInput
except ImportError as e:
    raise ImportError("ag-ui-protocol is not installed. Please install it with:\npip install ag2[ag-ui]") from e


from .events import AGUIEvent
from .stream import AGUIStream

__all__ = (
    "AGUIEvent",
    "AGUIStream",
    "RunAgentInput",
)
