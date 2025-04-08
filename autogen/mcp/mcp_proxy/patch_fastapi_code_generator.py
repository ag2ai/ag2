# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import json
import logging
import re
from functools import cached_property, wraps
from typing import Any

import stringcase
from fastapi_code_generator import __main__ as fastapi_code_generator_main
from fastapi_code_generator.parser import OpenAPIParser, Operation

logger = logging.getLogger(__name__)


def patch_parse_schema() -> None:
    org_parse_schema = OpenAPIParser.parse_schema

    @wraps(org_parse_schema)
    def my_parse_schema(*args: Any, **kwargs: Any) -> Any:
        data_type = org_parse_schema(*args, **kwargs)
        if data_type.reference and data_type.reference.duplicate_name:
            data_type.reference.name = data_type.reference.duplicate_name
        return data_type

    OpenAPIParser.parse_schema = my_parse_schema
    logger.info("Patched OpenAPIParser.parse_schema")


def patch_function_name_parsing() -> None:
    def function_name(self: Operation) -> str:
        if self.operationId:
            name: str = self.operationId.replace("/", "_")
        else:
            path = re.sub(r"/{|/", "_", self.snake_case_path).replace("}", "")
            name = f"{self.type}{path}"
        return stringcase.snakecase(name)  # type: ignore[no-any-return]

    Operation.function_name = cached_property(function_name)
    Operation.function_name.__set_name__(Operation, "function_name")

    logger.info("Patched Operation.function_name")


def patch_generate_code() -> None:
    # Save reference to the original generate_code function
    org_generate_code = fastapi_code_generator_main.generate_code

    @wraps(org_generate_code)
    def patched_generate_code(*args: Any, **kwargs: Any) -> Any:
        try:
            input_text: str = kwargs["input_text"]

            json_spec = json.loads(input_text)

            schemas_with_dots = sorted(
                [name for name in json_spec.get("components", {}).get("schemas", {}) if "." in name],
                key=len,
                reverse=True,  # Sort by length in descending order
            )

            for schema_name in schemas_with_dots:
                new_schema_name = schema_name.replace(".", "_")
                input_text = input_text.replace(schema_name, new_schema_name)

            kwargs["input_text"] = input_text

        except Exception as e:
            logger.info(
                f"Patched fastapi_code_generator.__main__.generate_code raised: {e}, passing untouched arguments to original generate_code"
            )

        return org_generate_code(*args, **kwargs)

    fastapi_code_generator_main.generate_code = patched_generate_code

    logger.info("Patched fastapi_code_generator.__main__.generate_code")
