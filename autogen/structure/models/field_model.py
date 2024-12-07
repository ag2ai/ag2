# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/lion-agi/lion-core are under the Apache-2.0 License
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined


class FieldModel(BaseModel):
    """Model for defining and managing field definitions.

    Provides a structured way to define fields with:
    - Type annotations and validation
    - Default values and factories
    - Documentation and metadata
    - Serialization options

    Example:
        ```python
        field = FieldModel(
            name="age",
            annotation=int,
            default=0,
            description="User age in years",
            validator=lambda v: v if v >= 0 else 0
        )
        ```

    Attributes:
        default: Default field value
        default_factory: Function to generate default value
        title: Field title for documentation
        description: Field description
        examples: Example values
        validators: Validation functions
        exclude: Exclude from serialization
        deprecated: Mark as deprecated
        frozen: Mark as immutable
        alias: Alternative field name
        alias_priority: Priority for alias resolution
        name: Field name (required)
        annotation: Type annotation
        validator: Validation function
        validator_kwargs: Validator parameters

    Notes:
        - All attributes except 'name' can be UNDEFINED
        - validator_kwargs are passed to field_validator decorator
        - Cannot have both default and default_factory
    """

    model_config = ConfigDict(
        extra="allow",
        validate_default=False,
        arbitrary_types_allowed=True,
        use_enum_values=True,
    )

    # Field configuration attributes
    default: Any = PydanticUndefined  # Default value
    default_factory: Callable = PydanticUndefined  # Factory function for default value
    title: str = PydanticUndefined  # Field title
    description: str = PydanticUndefined  # Field description
    examples: list = PydanticUndefined  # Example values
    validators: list = PydanticUndefined  # Validation functions
    exclude: bool = PydanticUndefined  # Exclude from serialization
    deprecated: bool = PydanticUndefined  # Mark as deprecated
    frozen: bool = PydanticUndefined  # Mark as immutable
    alias: str = PydanticUndefined  # Alternative field name
    alias_priority: int = PydanticUndefined  # Priority for alias resolution

    # Core field attributes
    name: str = Field(..., exclude=True)  # Field name (required)
    annotation: type | Any = Field(PydanticUndefined, exclude=True)  # Type annotation
    validator: Callable | Any = Field(PydanticUndefined, exclude=True)  # Validation function
    validator_kwargs: dict | Any = Field(default_factory=dict, exclude=True)  # Validator parameters

    @property
    def field_info(self) -> FieldInfo:
        """Generate Pydantic FieldInfo object from field configuration.

        Returns:
            FieldInfo object with all configured attributes

        Notes:
            - Uses clean dict to exclude UNDEFINED values
            - Sets annotation to Any if not specified
            - Preserves all metadata in field_info
        """
        annotation = self.annotation if self.annotation is not PydanticUndefined else Any
        field_obj: FieldInfo = Field(**self.model_dump(exclude_unset=True))  # type: ignore
        field_obj.annotation = annotation
        return field_obj

    @property
    def field_validator(self) -> dict[str, Callable] | None:
        """Generate field validator configuration.

        Returns:
            Dictionary mapping validator name to function,
            or None if no validator defined

        Notes:
            - Validator name is f"{field_name}_validator"
            - Uses validator_kwargs if provided
            - Returns None if validator is UNDEFINED
        """
        if self.validator is PydanticUndefined:
            return None
        kwargs = self.validator_kwargs or {}
        return {f"{self.name}_validator": field_validator(self.name, **kwargs)(self.validator)}


__all__ = ["FieldModel"]
