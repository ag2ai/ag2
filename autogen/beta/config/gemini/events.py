# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from uuid import uuid4

from google.genai import types

from autogen.beta.events import BuiltinToolCallEvent, BuiltinToolResultEvent
from autogen.beta.events.base import Field
from autogen.beta.events.tool_events import ToolResult
from autogen.beta.tools.builtin.code_execution import CODE_EXECUTION_TOOL_NAME


class GeminiServerToolCallEvent(BuiltinToolCallEvent):
    part: types.Part | None = Field(default=None, repr=False)
    grounding_metadata: types.GroundingMetadata | None = Field(default=None, repr=False)

    @classmethod
    def from_executable_code(cls, part: types.Part) -> "GeminiServerToolCallEvent | None":
        if part.executable_code is None:
            return None
        language = part.executable_code.language
        return cls(
            name=CODE_EXECUTION_TOOL_NAME,
            arguments=json.dumps({
                "code": part.executable_code.code or "",
                "language": language.name if language.name else str(language) or "",
            }),
            part=part,
        )

    @classmethod
    def from_grounding(cls, gm: types.GroundingMetadata, *, name: str) -> "GeminiServerToolCallEvent":
        return cls(
            id=str(uuid4()),
            name=name,
            arguments=json.dumps({"queries": list(gm.web_search_queries or [])}),
            grounding_metadata=gm,
        )


class GeminiServerToolResultEvent(BuiltinToolResultEvent):
    part: types.Part | None = Field(default=None, repr=False)
    grounding_metadata: types.GroundingMetadata | None = Field(default=None, repr=False)

    @classmethod
    def from_code_execution_result(cls, part: types.Part, *, parent_id: str) -> "GeminiServerToolResultEvent | None":
        if part.code_execution_result is None:
            return None
        return cls(
            parent_id=parent_id,
            name=CODE_EXECUTION_TOOL_NAME,
            result=ToolResult(),
            part=part,
        )

    @classmethod
    def from_grounding(cls, gm: types.GroundingMetadata, *, parent_id: str, name: str) -> "GeminiServerToolResultEvent":
        return cls(
            parent_id=parent_id,
            name=name,
            result=ToolResult(),
            grounding_metadata=gm,
        )
