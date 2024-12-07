# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/lion-agi/lion-core are under the Apache-2.0 License
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import BaseModel, JsonValue, field_validator

from .field_model import FieldModel
from .prompts import (
    context_examples,
    context_field_description,
    guidance_examples,
    guidance_field_description,
    instruction_examples,
    instruction_field_description,
)


def validate_instruction(cls, value) -> JsonValue | None:
    """Validates that the instruction is not empty or None and is in the correct format.

    Args:
        cls: The validator class method.
        value (JsonValue | None): The instruction value to validate.

    Returns:
        JsonValue | None: The validated instruction or None if invalid.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return value


# Field Models
INSTRUCTION_FIELD = FieldModel(
    name="instruction",
    annotation=JsonValue | None,
    default=None,
    title="Primary Instruction",
    description=instruction_field_description,
    examples=instruction_examples,
    validator=validate_instruction,
    validator_kwargs={"mode": "before"},
)

GUIDANCE_FIELD = FieldModel(
    name="guidance",
    annotation=JsonValue | None,
    default=None,
    title="Execution Guidance",
    description=guidance_field_description,
    examples=guidance_examples,
)

CONTEXT_FIELD = FieldModel(
    name="context",
    annotation=JsonValue | None,
    default=None,
    title="Task Context",
    description=context_field_description,
    examples=context_examples,
)


class Instruct(BaseModel):
    """Model for defining instruction parameters and execution requirements.

    Attributes:
        instruction (JsonValue | None): The primary instruction.
        guidance (JsonValue | None): Execution guidance.
        context (JsonValue | None): Task context.
        reason (bool): Whether to include reasoning.
        actions (bool): Whether specific actions are required.
    """

    instruction: JsonValue | None = INSTRUCTION_FIELD.field_info
    guidance: JsonValue | None = GUIDANCE_FIELD.field_info
    context: JsonValue | None = CONTEXT_FIELD.field_info

    @field_validator("instruction", **INSTRUCTION_FIELD.validator_kwargs)
    def _validate_instruction(cls, v):
        """Field validator for the 'instruction' field.

        Args:
            v: The value to validate.

        Returns:
            JsonValue | None: The validated instruction value.
        """
        return INSTRUCTION_FIELD.validator(cls, v)


class InstructResponse(BaseModel):
    instruct: Instruct
    response: Any = None


__all__ = ["Instruct", "InstructResponse"]
