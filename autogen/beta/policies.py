# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Qualified-key constants for ``ConversationContext.dependencies`` injection.

Agent-level plumbing keys stamped by the Agent execution loop so that
middleware and tools can introspect turn state without tight coupling to
internal agent classes.
"""

__all__ = ("AVAILABLE_TOOLS_DEP",)

AVAILABLE_TOOLS_DEP = "ag2.agent.available_tools"
"""Names of all tools (function + non-function) registered for the current turn.

Stamped into ``context.dependencies`` by :class:`autogen.beta.agent.Agent`
immediately after building ``all_schemas``. Value is a :class:`list[str]`.
"""
