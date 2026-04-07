# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from dataclasses import is_dataclass
from types import UnionType
from typing import Annotated, Any, Union, get_origin, overload

from fast_depends import Provider
from pydantic import BaseModel, Field, TypeAdapter
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
        embed: bool = True,
    ) -> None: ...

    @overload
    def __init__(
        self,
        types: ClassInfo,
        /,
        name: str | None = None,
        description: str | None = None,
        embed: bool = True,
    ) -> None: ...

    def __init__(
        self,
        types: ClassInfo,
        /,
        name: str | None = None,
        description: str | None = None,
        embed: bool = True,
    ) -> None:
        self._adapter, self._embedded_type = make_adapter(types, embed=embed)
        schema = self._adapter.json_schema() if self._adapter else None

        if not name:
            name = schema_title if (schema_title := (schema or {}).pop("title", None)) else "ResponseSchema"
        self.name = name

        if not description:
            if schema_description := (schema or {}).pop("description", None):
                self.description = schema_description
            elif (docstring := getattr(types, "__doc__", None)) and "PEP" not in docstring:
                self.description = docstring
            else:
                self.description = None
        else:
            self.description = description

        self.json_schema = schema
        self.system_prompt = None

    @overload
    @classmethod
    def ensure_schema(
        cls,
        obj: "ResponseProto[T] | type[T] | ClassInfo",
    ) -> "ResponseSchema[T]": ...

    @overload
    @classmethod
    def ensure_schema(
        cls,
        obj: "None",
    ) -> "None": ...

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
        if self._adapter is None:
            return response
        result = self._adapter.validate_json(response)
        if self._embedded_type:
            return result.data
        return result


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
        self.system_prompt = None

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


@overload
def make_adapter(types: type[T], *, embed: bool = True) -> tuple[TypeAdapter[T] | None, bool]: ...


@overload
def make_adapter(types: ClassInfo, *, embed: bool = True) -> tuple[TypeAdapter[T] | None, bool]: ...


def make_adapter(types: ClassInfo, *, embed: bool = True) -> tuple[TypeAdapter[T] | None, bool]:
    origin = get_origin(types)
    embedded_type = True

    if types is str:
        return None, True

    if _is_safe_subclass(types, (list, tuple)):
        # Process `T1, T2]` and `(T1, T2)`
        _final_type = Union[tuple(types)]  # noqa: UP007

    elif origin and origin in (Union, UnionType):
        # Process `typing.Union[T1, T2]` and `T1 | T2`
        _final_type = types

    elif any((
        is_dataclass(types),
        origin and issubclass(origin, dict),
        _is_safe_subclass(types, BaseModel),
        _is_safe_subclass(types, dict),
    )):
        embedded_type = False
        _final_type = types

    else:
        # Process primitive types
        _final_type = types

    if not embed:
        embedded_type = False

    if embedded_type:

        class _EmbeddedSchema(BaseModel):
            data: Annotated[_final_type, Field(description='Response with a one-field JSON `"{"data":...}"`')]
            model_config = {"title": "ResponseSchema"}

        _final_type = _EmbeddedSchema

    return TypeAdapter[T](_final_type), embedded_type


def _is_safe_subclass(cls: type, base: type | tuple[type, ...]) -> bool:
    try:
        return issubclass(cls, base)
    except TypeError:
        return issubclass(type(cls), base)
