# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Any, Union, overload

from fast_depends import Provider
from pydantic import TypeAdapter
from typing_extensions import TypeVar as TypeVar313

from autogen.beta.annotations import Context
from autogen.beta.types import ClassInfo

from .proto import ResponseProto

T = TypeVar313("T", default=str)


class ResponseSchema(ResponseProto[T]):
    @overload
    def __init__(
        self,
        types: type[T],
        /,
        name: str | None = None,
        description: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        types: ClassInfo,
        /,
        name: str | None = None,
        description: str | None = None,
    ) -> None: ...

    def __init__(
        self,
        types: ClassInfo,
        /,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        if hasattr(types, "__len__"):
            self._adapter: TypeAdapter[T] = TypeAdapter(Union[tuple(types)])  # noqa: UP007
        else:
            self._adapter = TypeAdapter(types)

        schema = self._adapter.json_schema()

        name = name or schema.pop("title", None) or "ResponseSchema"
        if not name:
            raise ValueError("You should provide `name` explicitly")

        self.name = name
        self.description: str | None = description or schema.pop("description", None)
        if not description and (docstring := getattr(types, "__doc__", None)) and "PEP" not in docstring:
            self.description = docstring

        self.json_schema = schema

    @classmethod
    def ensure_schema(
        cls,
        obj: "ResponseProto[T] | type[T] | ClassInfo | None",
    ) -> "ResponseProto[T] | None":
        if obj is None:
            return None
        if isinstance(obj, ResponseProto):
            return obj
        return ResponseSchema[T](obj)

    @classmethod
    def from_schema(
        cls,
        schema: dict[str, Any],
        /,
        name: str,
        description: str | None = None,
    ) -> "RawSchema":
        return RawSchema(schema, name=name, description=description)

    async def validate(
        self,
        response: str,
        context: "Context",
        provider: "Provider | None" = None,
    ) -> T:
        return self._adapter.validate_json(response)


class RawSchema(ResponseProto[str]):
    def __init__(
        self,
        schema: dict[str, Any],
        /,
        name: str,
        description: str | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.json_schema = schema

    async def validate(
        self,
        response: str,
        context: "Context",
        provider: "Provider | None" = None,
    ) -> str:
        warnings.warn(
            "RawSchema can't validate model response. "
            "It always return string as is. "
            "Please, validate content manually.",
            category=RuntimeWarning,
            stacklevel=2,
        )
        return response
