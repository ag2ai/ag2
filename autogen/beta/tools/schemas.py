# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field

FunctionParameters: TypeAlias = dict[str, Any]


class FunctionDefinition(BaseModel):
    name: str = Field(description="Name of the function to call.")
    description: str = Field(default="", description="Description of what the function does.")
    parameters: FunctionParameters = Field(
        default_factory=dict,
        description="JSON Schema for the function parameters.",
    )
    strict: bool | None = True

    def model_post_init(self, context: Any) -> None:
        self.parameters.pop("title", None)
        if self.strict is not None:
            self.parameters = {"additionalProperties": not self.strict} | self.parameters


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
