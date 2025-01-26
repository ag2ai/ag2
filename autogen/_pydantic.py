# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from typing import Any, Tuple, Union, get_args

from pydantic import BaseModel
from pydantic.version import VERSION as PYDANTIC_VERSION
from typing_extensions import get_origin

__all__ = ("JsonSchemaValue", "evaluate_forwardref", "model_dump", "model_dump_json", "type2schema")

PYDANTIC_V1 = PYDANTIC_VERSION.startswith("1.")

if not PYDANTIC_V1:
    from pydantic import TypeAdapter
    from pydantic._internal._typing_extra import eval_type_lenient as evaluate_forwardref
    from pydantic.json_schema import JsonSchemaValue

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


# Remove this once we drop support for pydantic 1.x
else:  # pragma: no cover
    from pydantic import schema_of
    from pydantic.typing import evaluate_forwardref as evaluate_forwardref  # type: ignore[no-redef]

    JsonSchemaValue = dict[str, Any]  # type: ignore[misc]

    def type2schema(t: Any) -> JsonSchemaValue:
        """Convert a type to a JSON schema

        Args:
            t (Type): The type to convert

        Returns:
            JsonSchemaValue: The JSON schema
        """
        if t is None:
            return {"type": "null"}
        elif get_origin(t) is Union:
            return {"anyOf": [type2schema(tt) for tt in get_args(t)]}
        # we need to support both syntaxes for Tuple
        elif get_origin(t) in [Tuple, tuple]:
            prefix_items = [type2schema(tt) for tt in get_args(t)]
            return {
                "maxItems": len(prefix_items),
                "minItems": len(prefix_items),
                "prefixItems": prefix_items,
                "type": "array",
            }
        else:
            d = schema_of(t)
            if "title" in d:
                d.pop("title")
            if "description" in d:
                d.pop("description")

            return d

    def model_dump(model: BaseModel) -> dict[str, Any]:
        """Convert a pydantic model to a dict

        Args:
            model (BaseModel): The model to convert

        Returns:
            Dict[str, Any]: The dict representation of the model

        """
        return model.dict()

    def model_dump_json(model: BaseModel) -> str:
        """Convert a pydantic model to a JSON string

        Args:
            model (BaseModel): The model to convert

        Returns:
            str: The JSON string representation of the model
        """
        return model.json()
