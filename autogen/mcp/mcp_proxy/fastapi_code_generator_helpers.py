# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cached_property

from ...import_utils import optional_import_block

with optional_import_block() as result:
    from fastapi_code_generator.parser import (
        Argument,
        OpenAPIParser,
        ParameterObject,
        ReferenceObject,
    )

SUCCESFUL_IMPORT = result.is_successful

__all__ = ["SUCCESFUL_IMPORT", "patch_get_parameter_type"]


@contextmanager
def patch_get_parameter_type() -> Iterator[None]:
    class ArgumentWithDescription(Argument):  # type: ignore[misc]
        description: str | None = None

        @cached_property
        def argument(self) -> str:
            if self.description:
                description = self.description.replace('"""', '"""')
                type_hint = f'Annotated[{self.type_hint}, """{description}"""]'
            else:
                type_hint = self.type_hint

            if self.default is None and self.required:
                return f"{self.name}: {type_hint}"

            return f"{self.name}: {type_hint} = {self.default}"

    original_get_parameter_type = OpenAPIParser.get_parameter_type

    def get_parameter_type(
        self: OpenAPIParser,
        parameters: ReferenceObject | ParameterObject,
        snake_case: bool,
        path: list[str],
    ) -> Argument | None:
        # get the original argument
        argument = original_get_parameter_type(self, parameters, snake_case, path)

        # add description to the argument
        parameters = self.resolve_object(parameters, ParameterObject)
        argument_with_description = ArgumentWithDescription(description=parameters.description, **argument.model_dump())
        return argument_with_description

    OpenAPIParser.get_parameter_type = get_parameter_type

    try:
        yield
    finally:
        OpenAPIParser.get_parameter_type = original_get_parameter_type
