# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/lion-agi/lion-core are under the Apache-2.0 License
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, create_model, field_validator, model_validator
from pydantic.fields import FieldInfo

from .field_model import FieldModel


class NewModelParams(BaseModel):
    """Configuration class for dynamically creating new Pydantic models."""

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        use_enum_values=True,
    )

    name: str | None = None
    parameter_fields: dict[str, FieldInfo] = Field(default_factory=dict)
    base_type: type[BaseModel] = Field(default=BaseModel)
    field_models: list[FieldModel] = Field(default_factory=list)
    exclude_fields: list = Field(default_factory=list)
    field_descriptions: dict = Field(default_factory=dict)
    inherit_base: bool = Field(default=True)
    config_dict: dict | None = Field(default=None)
    doc: str | None = Field(default=None)
    frozen: bool = False
    _validators: dict[str, Callable] | None = PrivateAttr(default=None)
    _use_keys: set[str] = PrivateAttr(default_factory=set)

    @property
    def use_fields(self):
        """Get field definitions to use in new model."""
        params = {k: v for k, v in self.parameter_fields.items() if k in self._use_keys}
        params.update({f.name: f.field_info for f in self.field_models if f.name in self._use_keys})
        return {k: (v.annotation, v) for k, v in params.items()}

    @field_validator("parameter_fields", mode="before")
    def validate_parameters(cls, value):
        """Validate parameter field definitions."""
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("Fields must be a dictionary.")
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError("Field names must be strings.")
            if not isinstance(v, FieldInfo):
                raise ValueError("Field values must be FieldInfo objects.")
        return value.copy()

    @field_validator("base_type", mode="before")
    def validate_base(cls, value) -> type[BaseModel]:
        """Validate base model type."""
        if value is None:
            return BaseModel
        if isinstance(value, type) and issubclass(value, BaseModel):
            return value
        if isinstance(value, BaseModel):
            return value.__class__
        raise ValueError("Base must be a BaseModel subclass or instance.")

    @field_validator("exclude_fields", mode="before")
    def validate_fields(cls, value) -> list[str]:
        """Validate excluded fields list."""
        if value is None:
            return []
        if isinstance(value, dict):
            value = list(value.keys())
        if isinstance(value, set | tuple):
            value = list(value)
        if isinstance(value, list):
            if not all(isinstance(i, str) for i in value):
                raise ValueError("Field names must be strings.")
            return value.copy()
        raise ValueError("Fields must be a list, set, or dictionary.")

    @field_validator("field_descriptions", mode="before")
    def validate_field_descriptions(cls, value) -> dict[str, str]:
        """Validate field descriptions dictionary."""
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("Field descriptions must be a dictionary.")
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError("Field names must be strings.")
            if not isinstance(v, str):
                raise ValueError("Field descriptions must be strings.")
        return value

    @field_validator("name", mode="before")
    def validate_name(cls, value) -> str:
        """Validate model name."""
        if value is None:
            return "StepModel"
        if not isinstance(value, str):
            raise ValueError("Name must be a string.")
        return value

    @field_validator("field_models", mode="before")
    def _validate_field_models(cls, value):
        """Validate field model definitions."""
        if value is None:
            return []
        value = [value] if not isinstance(value, list) else value
        if not all(isinstance(i, FieldModel) for i in value):
            raise ValueError("Field models must be FieldModel objects.")
        return value

    @model_validator(mode="after")
    def validate_param_model(self):
        """Validate complete model configuration."""
        if self.base_type is not None:
            self.parameter_fields.update(self.base_type.model_fields)

        self.parameter_fields.update({f.name: f.field_info for f in self.field_models})

        use_keys = list(self.parameter_fields.keys())
        use_keys.extend(list(self._use_keys))

        if self.exclude_fields:
            use_keys = [i for i in use_keys if i not in self.exclude_fields]

        self._use_keys = set(use_keys)

        validators = {}

        for i in self.field_models:
            if i.field_validator is not None:
                validators.update(i.field_validator)
        self._validators = validators

        if self.field_descriptions:
            for i in self.field_models:
                if i.name in self.field_descriptions:
                    i.description = self.field_descriptions[i.name]

        if not isinstance(self.name, str):
            if hasattr(self.base_type, "class_name"):
                if callable(self.base_type.class_name):
                    self.name = self.base_type.class_name()
                else:
                    self.name = self.base_type.class_name
            elif inspect.isclass(self.base_type):
                self.name = self.base_type.__name__

        return self

    def create_new_model(self) -> type[BaseModel]:
        """Create new Pydantic model with specified configuration."""
        a: type[BaseModel] = create_model(
            self.name,
            __config__=self.config_dict,
            __doc__=self.doc,
            __base__=self.base_type if self.inherit_base else None,
            __validators__=self._validators,
            **self.use_fields,
        )
        if self.frozen:
            a.model_config["frozen"] = True
        return a


__all__ = ["NewModelParams"]
