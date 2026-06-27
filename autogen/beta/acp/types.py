# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Shared ACP type aliases used across the integration.

The ``acp`` SDK does not export named aliases for its discriminated unions, so we
mirror them here once (``SessionUpdate``, ``ContentBlock``, ``ToolCallContent``)
and reuse them everywhere instead of repeating the member lists.
"""

from typing import TypeAlias

from acp import schema

# The ``session/update`` payloads (the ``SessionNotification.update`` union).
SessionUpdate: TypeAlias = (
    schema.UserMessageChunk
    | schema.AgentMessageChunk
    | schema.AgentThoughtChunk
    | schema.ToolCallStart
    | schema.ToolCallProgress
    | schema.AgentPlanUpdate
    | schema.AvailableCommandsUpdate
    | schema.CurrentModeUpdate
    | schema.ConfigOptionUpdate
    | schema.SessionInfoUpdate
    | schema.UsageUpdate
)

# A single content block carried by message/thought chunks and tool-call content.
ContentBlock: TypeAlias = (
    schema.TextContentBlock
    | schema.ImageContentBlock
    | schema.AudioContentBlock
    | schema.ResourceContentBlock
    | schema.EmbeddedResourceContentBlock
)

# The items of a tool call's ``content`` list.
ToolCallContent: TypeAlias = (
    schema.ContentToolCallContent | schema.FileEditToolCallContent | schema.TerminalToolCallContent
)
