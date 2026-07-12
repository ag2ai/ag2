# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.exceptions import missing_optional_dependency

try:
    from . import actions
    from .resources import external_url, raw_html, remote_dom
except ImportError as e:  # pragma: no cover - exercised only when ag2[mcp-ui] is absent
    external_url = missing_optional_dependency("external_url", "mcp-ui", e)  # type: ignore[misc]
    raw_html = missing_optional_dependency("raw_html", "mcp-ui", e)  # type: ignore[misc]
    remote_dom = missing_optional_dependency("remote_dom", "mcp-ui", e)  # type: ignore[misc]
    actions = missing_optional_dependency("actions", "mcp-ui", e)  # type: ignore[misc]

__all__ = (
    "actions",
    "external_url",
    "raw_html",
    "remote_dom",
)
