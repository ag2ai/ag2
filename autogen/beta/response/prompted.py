# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import overload

from fast_depends import Provider
from typing_extensions import TypeVar as TypeVar313

from autogen.beta.annotations import Context
from autogen.beta.types import ClassInfo

from .proto import ResponseProto
from .schema import ResponseSchema

T = TypeVar313("T", default=str)

_DEFAULT_PROMPT_TEMPLATE = (
    "You must respond with valid JSON that conforms to the following JSON schema:\n"
    "```json\n"
    "{schema}\n"
    "```\n"
    "Do not include any text, markdown formatting, or explanation outside the JSON object."
)


class PromptedSchema(ResponseProto[T]):
    """Response schema that uses prompt-based instructions instead of native API structured output.

    Use this for models or providers that do not support built-in structured output
    (e.g., ``response_format``). The JSON schema is injected into the system prompt,
    instructing the model to return valid JSON matching the schema. Validation is
    still performed using the inner schema's ``validate`` method.

    Examples:
        Using with a type directly::

            agent = Agent(..., response_schema=PromptedSchema(MyModel))

        Wrapping an existing ResponseProto::

            schema = ResponseSchema(int | str)
            agent = Agent(..., response_schema=PromptedSchema(schema))
    """

    @overload
    def __init__(
        self,
        inner: ResponseProto[T],
        /,
        *,
        prompt_template: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        inner: type[T],
        /,
        *,
        prompt_template: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        inner: ClassInfo,
        /,
        *,
        prompt_template: str | None = None,
    ) -> None: ...

    def __init__(
        self,
        inner: "ResponseProto[T] | type[T] | ClassInfo",
        /,
        *,
        prompt_template: str | None = None,
    ) -> None:
        self._inner = ResponseSchema[T].ensure_schema(inner)

        self.name = self._inner.name
        self.description = self._inner.description

        self._json_schema = self._inner.json_schema
        self._prompt_template = prompt_template or _DEFAULT_PROMPT_TEMPLATE
        if self._json_schema:
            schema_str = json.dumps(self._json_schema, indent=2)
            self.system_prompt = self._prompt_template.format(schema=schema_str)
        else:
            self.system_prompt = None

        # Set public property to None to avoid native JSON schema validation
        self.json_schema = None

    async def validate(
        self,
        response: str,
        context: "Context",
        provider: "Provider | None" = None,
    ) -> T:
        return await self._inner.validate(response, context, provider)
