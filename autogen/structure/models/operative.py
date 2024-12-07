# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/lion-agi/lion-core are under the Apache-2.0 License
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field, PrivateAttr, model_validator
from pydantic.fields import FieldInfo

from autogen.structure.utils import to_json, validate_keys

from .field_model import FieldModel
from .new_model_params import NewModelParams


class Operative(BaseModel):
    """Class representing an operative that handles request and response models for operations."""

    name: str | None = None

    request_params: NewModelParams | None = Field(default=None)
    request_type: type[BaseModel] | None = Field(default=None)

    response_params: NewModelParams | None = Field(default=None)
    response_type: type[BaseModel] | None = Field(default=None)
    response_model: BaseModel | None = Field(default=None)
    response_str_dict: dict | str | None = Field(default=None)

    auto_retry_parse: bool = True
    max_retries: int = 3
    _should_retry: bool = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate(self):
        """Validates the operative instance after initialization."""
        if self.request_type is None:
            self.request_type = self.request_params.create_new_model()
        if self.name is None:
            self.name = self.request_params.name or self.request_type.__name__
        return self

    def raise_validate_pydantic(self, text: str) -> None:
        """Validates and updates the response model using strict matching.

        Args:
            text (str): The text to validate and parse into the response model.

        Raises:
            Exception: If the validation fails.
        """
        d_ = to_json(text, fuzzy_parse=True)
        if isinstance(d_, list | tuple) and len(d_) == 1:
            d_ = d_[0]
        try:
            d_ = validate_keys(d_, self.request_type.model_fields, handle_unmatched="raise")
            self.response_model = self.request_type.model_validate(d_)
            self._should_retry = False
        except Exception:
            self.response_str_dict = d_
            self._should_retry = True

    def force_validate_pydantic(self, text: str):
        """Forcibly validates and updates the response model, allowing unmatched fields.

        Args:
            text (str): The text to validate and parse into the response model.
        """
        d_ = text
        try:
            d_ = to_json(text, fuzzy_parse=True)
            if isinstance(d_, list | tuple) and len(d_) == 1:
                d_ = d_[0]
            d_ = validate_keys(d_, self.request_type.model_fields, handle_unmatched="force")
            self.response_model = self.request_type.model_validate(d_)
            self._should_retry = False
        except Exception:
            self.response_str_dict = d_
            self.response_model = None
            self._should_retry = True

    def update_response_model(self, text: str | None = None, data: dict | None = None) -> BaseModel | dict | str | None:
        """Updates the response model based on the provided text or data.

        Args:
            text (str, optional): The text to parse and validate.
            data (dict, optional): The data to update the response model with.

        Returns:
            BaseModel | dict | str | None: The updated response model or raw data.

        Raises:
            ValueError: If neither text nor data is provided.
        """
        if text is None and data is None:
            raise ValueError("Either text or data must be provided.")

        if text:
            self.response_str_dict = text
            try:
                self.raise_validate_pydantic(text)
            except Exception:
                self.force_validate_pydantic(text)

        if data and self.response_type:
            d_ = self.response_model.model_dump()
            d_.update(data)
            self.response_model = self.response_type.model_validate(d_)

        if not self.response_model and isinstance(self.response_str_dict, list):
            try:
                self.response_model = [self.request_type.model_validate(d_) for d_ in self.response_str_dict]
            except Exception:
                pass

        return self.response_model or self.response_str_dict

    def create_response_type(
        self,
        response_params: NewModelParams | None = None,
        field_models: list[FieldModel] | None = None,
        parameter_fields: dict[str, FieldInfo] | None = None,
        exclude_fields: list[str] | None = None,
        field_descriptions: dict[str, str] | None = None,
        inherit_base: bool = True,
        config_dict: dict | None = None,
        doc: str | None = None,
        frozen: bool = False,
        validators: dict | None = None,
    ) -> None:
        """Creates a new response type based on the provided parameters.

        Args:
            response_params (NewModelParams, optional): Parameters for the new response model.
            field_models (list[FieldModel], optional): List of field models.
            parameter_fields (dict[str, FieldInfo], optional): Dictionary of parameter fields.
            exclude_fields (list, optional): List of fields to exclude.
            field_descriptions (dict, optional): Dictionary of field descriptions.
            inherit_base (bool, optional): Whether to inherit the base model.
            config_dict (dict | None, optional): Configuration dictionary.
            doc (str | None, optional): Documentation string.
            frozen (bool, optional): Whether the model is frozen.
            validators (dict, optional): Dictionary of validators.
        """
        self.response_params = response_params or NewModelParams(
            parameter_fields=parameter_fields,
            field_models=field_models,
            exclude_fields=exclude_fields,
            field_descriptions=field_descriptions,
            inherit_base=inherit_base,
            config_dict=config_dict,
            doc=doc,
            frozen=frozen,
            base_type=self.request_params.base_type,
        )
        if validators and isinstance(validators, dict):
            self.response_params._validators.update(validators)

        self.response_type = self.response_params.create_new_model()
