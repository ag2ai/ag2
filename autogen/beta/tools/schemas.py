# Copyright (c) 2023 - 2026, AG2ai, Inc. AG2ai open-source projects maintainers and contributors.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from pydantic import BaseModel, Field


class FunctionParameters(BaseModel):
    type: Literal["object"] = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class FunctionDefinition(BaseModel):
    name: str = Field(description="Name of the function to call.")
    description: str = Field(default="", description="Description of what the function does.")
    parameters: FunctionParameters = Field(
        default_factory=FunctionParameters,
        description="JSON Schema for the function parameters.",
    )


class FunctionTool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition = Field(description="The function definition.")

    def to_api(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


__all__ = [
    "FunctionDefinition",
    "FunctionParameters",
    "FunctionTool",
]
