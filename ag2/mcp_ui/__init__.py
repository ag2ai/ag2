# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.exceptions import missing_optional_dependency

try:
    from .actions import intent, link, notify, post_message, prompt, tool_call
    from .resources import external_url, raw_html, remote_dom
except ImportError as e:  # pragma: no cover - exercised only when ag2[mcp-ui] is absent
    external_url = missing_optional_dependency("external_url", "mcp-ui", e)  # type: ignore[misc]
    raw_html = missing_optional_dependency("raw_html", "mcp-ui", e)  # type: ignore[misc]
    remote_dom = missing_optional_dependency("remote_dom", "mcp-ui", e)  # type: ignore[misc]
    tool_call = missing_optional_dependency("tool_call", "mcp-ui", e)  # type: ignore[misc]
    prompt = missing_optional_dependency("prompt", "mcp-ui", e)  # type: ignore[misc]
    link = missing_optional_dependency("link", "mcp-ui", e)  # type: ignore[misc]
    intent = missing_optional_dependency("intent", "mcp-ui", e)  # type: ignore[misc]
    notify = missing_optional_dependency("notify", "mcp-ui", e)  # type: ignore[misc]
    post_message = missing_optional_dependency("post_message", "mcp-ui", e)  # type: ignore[misc]

__all__ = (
    "external_url",
    "intent",
    "link",
    "notify",
    "post_message",
    "prompt",
    "raw_html",
    "remote_dom",
    "tool_call",
)
