# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any
from uuid import uuid4

from google.genai import types

from ag2.events import (
    BinaryType,
    BuiltinToolCallEvent,
    BuiltinToolResultEvent,
    Field,
    Input,
    TextInput,
    ToolCallEvent,
    ToolResult,
    UrlInput,
)
from ag2.tools.builtin.code_execution import CODE_EXECUTION_TOOL_NAME


class GeminiToolCallEvent(ToolCallEvent):
    """Function tool call from Gemini, carries thinking metadata."""

    thought_signature: bytes | None = Field(default=None, repr=False)


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
            # web_search / web_fetch use web_search_queries; file_search uses retrieval_queries.
            arguments=json.dumps({"queries": list(gm.web_search_queries or []) + list(gm.retrieval_queries or [])}),
            grounding_metadata=gm,
        )


class GeminiServerToolResultEvent(BuiltinToolResultEvent):
    part: types.Part | None = Field(default=None, repr=False)
    grounding_metadata: types.GroundingMetadata | None = Field(default=None, repr=False)

    @classmethod
    def from_code_execution_result(cls, part: types.Part, *, parent_id: str) -> "GeminiServerToolResultEvent | None":
        result = part.code_execution_result
        if result is None:
            return None
        parts: list[Input] = [TextInput(result.output)] if result.output else []
        metadata = {"outcome": result.outcome.name if result.outcome is not None else None}
        return cls(
            parent_id=parent_id,
            name=CODE_EXECUTION_TOOL_NAME,
            result=ToolResult(parts=parts, metadata=metadata),
            part=part,
        )

    @classmethod
    def from_grounding(cls, gm: types.GroundingMetadata, *, parent_id: str, name: str) -> "GeminiServerToolResultEvent":
        chunks = gm.grounding_chunks or []
        parts: list[Input] = []
        for chunk in chunks:
            if chunk.web and chunk.web.uri:
                parts.append(
                    UrlInput(
                        chunk.web.uri,
                        kind=BinaryType.BINARY,
                        metadata={"title": chunk.web.title, "domain": chunk.web.domain},
                    )
                )
            elif chunk.maps and chunk.maps.uri:
                parts.append(
                    UrlInput(
                        chunk.maps.uri,
                        kind=BinaryType.BINARY,
                        metadata={"title": chunk.maps.title, "place_id": chunk.maps.place_id},
                    )
                )
            elif chunk.retrieved_context:
                rc = chunk.retrieved_context
                if rc.text:
                    # file_search returns the retrieved chunk as text (no uri).
                    parts.append(
                        TextInput(
                            rc.text,
                            metadata={
                                "title": rc.title,
                                "uri": rc.uri,
                                "document_name": rc.document_name,
                                "file_search_store": rc.file_search_store,
                                "page_number": rc.page_number,
                            },
                        )
                    )
                elif rc.uri:
                    parts.append(
                        UrlInput(
                            rc.uri,
                            kind=BinaryType.BINARY,
                            metadata={
                                "title": rc.title,
                                "file_search_store": rc.file_search_store,
                            },
                        )
                    )
        queries = list(gm.web_search_queries or []) + list(gm.retrieval_queries or [])
        metadata: dict[str, Any] = {"queries": queries} if chunks else {}
        return cls(
            parent_id=parent_id,
            name=name,
            result=ToolResult(parts=parts, metadata=metadata),
            grounding_metadata=gm,
        )
