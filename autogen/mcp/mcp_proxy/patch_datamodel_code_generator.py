# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Union

from datamodel_code_generator.imports import (
    IMPORT_LITERAL,
    IMPORT_LITERAL_BACKPORT,
    Imports,
)
from datamodel_code_generator.model import pydantic as pydantic_model
from datamodel_code_generator.model import pydantic_v2 as pydantic_model_v2
from datamodel_code_generator.model.base import (
    DataModel,
)
from datamodel_code_generator.reference import Reference

# from datamodel_code_generator.parser import base
from fastapi_code_generator.parser import OpenAPIParser

logger = logging.getLogger(__name__)

# Save the original method before patching
original_apply_discriminator_type = OpenAPIParser._Parser__apply_discriminator_type


def patch_apply_discriminator_type() -> None:  # noqa: C901
    def __apply_discriminator_type_patched(  # noqa: C901
        self: OpenAPIParser,
        models: list[DataModel],
        imports: Imports,
    ) -> None:
        for model in models:
            for field in model.fields:
                discriminator = field.extras.get("discriminator")
                if not discriminator or not isinstance(discriminator, dict):
                    continue
                property_name = discriminator.get("propertyName")
                if not property_name:  # pragma: no cover
                    continue
                mapping = discriminator.get("mapping", {})
                for data_type in field.data_type.data_types:
                    if not data_type.reference:  # pragma: no cover
                        continue
                    discriminator_model = data_type.reference.source

                    if not isinstance(  # pragma: no cover
                        discriminator_model,
                        (pydantic_model.BaseModel, pydantic_model_v2.BaseModel),
                    ):
                        continue  # pragma: no cover

                    type_names: list[str] = []

                    def check_paths(
                        model: Union[
                            pydantic_model.BaseModel,
                            pydantic_model_v2.BaseModel,
                            Reference,
                        ],
                        mapping: dict[str, str],
                        type_names: list[str] = type_names,
                    ) -> None:
                        """Helper function to validate paths for a given model."""
                        for name, path in mapping.items():
                            if (model.path.split("#/")[-1] != path.split("#/")[-1]) and (
                                path.startswith("#/") or model.path[:-1] != path.split("/")[-1]
                            ):
                                t_path = path[str(path).find("/") + 1 :]
                                t_disc = model.path[: str(model.path).find("#")].lstrip(  # noqa: B005
                                    "../"
                                )
                                t_disc_2 = "/".join(t_disc.split("/")[1:])
                                if t_path != t_disc and t_path != t_disc_2:
                                    continue
                            type_names.append(name)

                    # Check the main discriminator model path
                    if mapping:
                        check_paths(discriminator_model, mapping)

                        # Check the base_classes if they exist
                        for base_class in discriminator_model.base_classes:
                            if base_class.reference and base_class.reference.path:
                                check_paths(base_class.reference, mapping)
                    else:
                        type_names = [discriminator_model.path.split("/")[-1]]
                    if not type_names:  # pragma: no cover
                        raise RuntimeError(f"Discriminator type is not found. {data_type.reference.path}")
                    has_one_literal = False
                    for discriminator_field in discriminator_model.fields:
                        if (discriminator_field.original_name or discriminator_field.name) != property_name:
                            continue
                        literals = discriminator_field.data_type.literals
                        if len(literals) == 1 and literals[0] == type_names[0] if type_names else None:
                            has_one_literal = True
                            continue
                        for field_data_type in discriminator_field.data_type.all_data_types:
                            if field_data_type.reference:  # pragma: no cover
                                field_data_type.remove_reference()
                        discriminator_field.data_type = self.data_type(literals=type_names)
                        discriminator_field.data_type.parent = discriminator_field
                        discriminator_field.required = True
                        imports.append(discriminator_field.imports)
                        has_one_literal = True
                    if not has_one_literal:
                        discriminator_model.fields.append(
                            self.data_model_field_type(
                                name=property_name,
                                data_type=self.data_type(literals=type_names),
                                required=True,
                            )
                        )
                    literal = IMPORT_LITERAL if self.target_python_version.has_literal_type else IMPORT_LITERAL_BACKPORT
                    has_imported_literal = any(
                        literal == import_  # type: ignore [comparison-overlap]
                        for import_ in imports
                    )
                    if has_imported_literal:  # pragma: no cover
                        imports.append(literal)

    # Patch the method using the exact mangled name
    OpenAPIParser._Parser__apply_discriminator_type = __apply_discriminator_type_patched

    logger.info("Patched Parser.__apply_discriminator_type,")
