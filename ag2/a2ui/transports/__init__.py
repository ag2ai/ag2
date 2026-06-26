# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A2UI wire transports for :class:`~ag2.a2ui.A2UIServer`.

One :class:`A2UITransport` == one deployment's wire encoding. :class:`RestTransport`
(SSE/NDJSON) needs Starlette; :class:`AgUiTransport` (CopilotKit's AG-UI renderer)
additionally needs ``ag2[ag-ui]``. A missing install surfaces as a clear hint on
first use. The :class:`A2UITransport` protocol itself is always importable.
"""

from ag2.exceptions import missing_additional_dependency, missing_optional_dependency

from .base import A2UITransport

try:
    from .rest import RestTransport
except ImportError as e:  # pragma: no cover - exercised only without starlette
    RestTransport = missing_additional_dependency("RestTransport", "starlette>=0.40,<1", e)  # type: ignore[misc]

try:
    from .ag_ui import AgUiTransport
except ImportError as e:  # pragma: no cover - exercised only without ag-ui-protocol/starlette
    AgUiTransport = missing_optional_dependency("AgUiTransport", "ag-ui", e)  # type: ignore[misc]

__all__ = (
    "A2UITransport",
    "AgUiTransport",
    "RestTransport",
)
