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
    | schema.AgentPlanContentUpdate
    | schema.AgentPlanRemovedUpdate
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

# One property of an elicitation form's requested schema, mirroring the union
# ``ElicitationSchema.properties`` admits.
ElicitationProperty: TypeAlias = (
    schema.ElicitationStringPropertySchema
    | schema.ElicitationNumberPropertySchema
    | schema.ElicitationIntegerPropertySchema
    | schema.ElicitationBooleanPropertySchema
    | schema.ElicitationMultiSelectPropertySchema
    | schema.ElicitationOtherPropertySchema
)

# What an answered property carries: the root of ``ElicitationContentValue``, which
# the SDK exports only as a ``RootModel`` wrapper, and the set the property schemas
# above declare their ``default`` from. ``None`` is deliberately not a member —
# unanswered is spelled ``ElicitationValue | None`` at the sites that allow it.
ElicitationValue: TypeAlias = str | int | float | bool | list[str]

# Every MCP server shape a Client may declare in ``session/new``, mirroring
# ``NewSessionRequest.mcp_servers``. The ``acp`` SDK parses the request; naming
# the result here is what keeps a recorded server readable as its own model
# rather than as ``Any`` at every point it is passed along.
McpServer: TypeAlias = schema.HttpMcpServer | schema.SseMcpServer | schema.AcpMcpServer | schema.McpServerStdio
