# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from typing import Any

from pydantic import BaseModel, TypeAdapter
from pydantic._internal._typing_extra import try_eval_type as evaluate_forwardref
from pydantic.json_schema import JsonSchemaValue

__all__ = ("JsonSchemaValue", "evaluate_forwardref", "model_dump", "model_dump_json", "type2schema")


def type2schema(t: Any) -> JsonSchemaValue:
    """Convert a type to a JSON schema

    Args:
        t (Type): The type to convert

    Returns:
        JsonSchemaValue: The JSON schema
    """
    return TypeAdapter(t).json_schema()


def model_dump(model: BaseModel) -> dict[str, Any]:
    """Convert a pydantic model to a dict

    Args:
        model (BaseModel): The model to convert

    Returns:
        Dict[str, Any]: The dict representation of the model

    """
    return model.model_dump()


def model_dump_json(model: BaseModel) -> str:
    """Convert a pydantic model to a JSON string

    Args:
        model (BaseModel): The model to convert

    Returns:
        str: The JSON string representation of the model
    """
    return model.model_dump_json()
