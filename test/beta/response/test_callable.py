# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Annotated, Any

import pytest
from dirty_equals import IsPartialDict
from pydantic import Field

from autogen.beta import Context, Depends, Variable
from autogen.beta.response import response_schema
from autogen.beta.stream import MemoryStream


class TestNameDescription:
    def test_name_from_function(self) -> None:
        @response_schema
        def my_parser(content: str) -> int:
            return int(content)

        assert my_parser.name == "my_parser"

    def test_explicit_name(self) -> None:
        @response_schema(name="custom")
        def parse(content: str) -> int:
            return int(content)

        assert parse.name == "custom"

    def test_description_from_docstring(self) -> None:
        @response_schema
        def parse(content: str) -> int:
            """Parse an integer."""
            return int(content)

        assert parse.description == "Parse an integer."

    def test_explicit_description(self) -> None:
        @response_schema(description="Custom description")
        def parse(content: str) -> int:
            return int(content)

        assert parse.description == "Custom description"

    def test_no_docstring_gives_empty_string(self) -> None:
        @response_schema
        async def parse(content: str) -> int:
            return int(content)

        assert parse.description is None


class TestSchemaGeneration:
    def test_single_str_param_no_schema(self) -> None:
        @response_schema
        def parse(content: str) -> int:
            return int(content)

        assert parse.json_schema is None

    def test_single_int_param_generates_schema(self) -> None:
        @response_schema
        def double(value: int) -> int:
            return value * 2

        assert double.json_schema == IsPartialDict({
            "type": "object",
            "properties": IsPartialDict({
                "data": IsPartialDict({
                    "description": 'Response with a one-field JSON `"{"data":...}"`',
                    "title": "Data",
                    "type": "integer",
                }),
            }),
            "required": ["data"],
        })

    def test_single_dataclass_param_generates_schema(self) -> None:
        @dataclass
        class Point:
            x: int
            y: int

        @response_schema
        def process(point: Point) -> str:
            return f"{point.x},{point.y}"

        assert process.json_schema == IsPartialDict({
            "type": "object",
            "properties": IsPartialDict({
                "x": IsPartialDict({"type": "integer"}),
                "y": IsPartialDict({"type": "integer"}),
            }),
        })

    def test_single_union_param_generates_schema(self) -> None:
        @response_schema
        def process(value: int | str) -> str:
            return str(value)

        assert process.json_schema == IsPartialDict({
            "type": "object",
            "properties": IsPartialDict({
                "data": IsPartialDict({
                    "description": 'Response with a one-field JSON `"{"data":...}"`',
                    "title": "Data",
                    "anyOf": [
                        {"type": "integer"},
                        {"type": "string"},
                    ],
                }),
            }),
            "required": ["data"],
        })

    def test_multi_params_generates_object_schema(self) -> None:
        @response_schema
        def combine(name: str, age: int) -> str:
            return f"{name} is {age}"

        assert combine.json_schema == IsPartialDict({
            "type": "object",
            "properties": IsPartialDict({
                "name": IsPartialDict({"type": "string"}),
                "age": IsPartialDict({"type": "integer"}),
            }),
            "required": ["name", "age"],
        })

    def test_multi_params_with_field_descriptions(self) -> None:
        @response_schema
        def process(
            name: Annotated[str, Field(description="User name")],
            score: float = Field(1.0, description="Test score"),
        ) -> str:
            return f"{name}: {score}"

        assert process.json_schema == IsPartialDict({
            "properties": IsPartialDict({
                "name": IsPartialDict({"description": "User name"}),
                "score": IsPartialDict({"description": "Test score", "default": 1.0}),
            }),
        })

    def test_multi_params_with_defaults(self) -> None:
        @response_schema
        def process(
            name: str,
            retries: int = 3,
        ) -> str:
            return f"{name} x{retries}"

        assert process.json_schema == IsPartialDict({
            "required": ["name"],
            "properties": IsPartialDict({
                "retries": IsPartialDict({"default": 3}),
            }),
        })

    def test_custom_fields_has_no_effect(self) -> None:
        @response_schema
        def combine(
            name: str,
            age: int,
            # has no effect
            ctx: Context,
            var: Annotated[int, Variable()],
        ) -> str:
            return f"{name} is {age}"

        assert combine.json_schema == IsPartialDict({
            "type": "object",
            "properties": IsPartialDict({
                "name": IsPartialDict({"type": "string"}),
                "age": IsPartialDict({"type": "integer"}),
            }),
            "required": ["name", "age"],
        })

    def test_respect_dependencies(self) -> None:
        def get_age(age: int) -> int:
            return age

        @response_schema
        def combine(
            name: str,
            age: Annotated[int, Depends(get_age)],
        ) -> str:
            return f"{name} is {age}"

        assert combine.json_schema == IsPartialDict({
            "type": "object",
            "properties": IsPartialDict({
                "name": IsPartialDict({"type": "string"}),
                "age": IsPartialDict({"type": "integer"}),
            }),
            "required": ["name", "age"],
        })

    def test_custom_schema_overrides_generated(self) -> None:
        custom = {"type": "object", "properties": {"x": {"type": "number"}}}

        @response_schema(schema=custom)
        def parse(content: str) -> float:
            return float(content)

        assert parse.json_schema == custom


class TestValidation:
    @pytest.mark.asyncio()
    async def test_sync_str_param(self) -> None:
        @response_schema
        def parse(content: str) -> int:
            return int(content)

        result = await parse.validate("42", context=None)  # type: ignore[arg-type]

        assert result == 42

    @pytest.mark.asyncio()
    async def test_async_str_param(self) -> None:
        @response_schema
        async def parse(content: str) -> int:
            return int(content)

        result = await parse.validate("42", context=None)  # type: ignore[arg-type]

        assert result == 42

    @pytest.mark.asyncio()
    async def test_single_int_param_deserializes(self) -> None:
        @response_schema
        def double(value: int) -> int:
            return value * 2

        result = await double.validate('{"data": 21}', context=None)  # type: ignore[arg-type]

        assert result == 42

    @pytest.mark.asyncio()
    async def test_single_dataclass_param_deserializes(self) -> None:
        @dataclass
        class Point:
            x: int
            y: int

        @response_schema
        def stringify(point: Point) -> str:
            return f"{point.x},{point.y}"

        result = await stringify.validate(
            '{"x": 3, "y": 4}',
            context=None,  # type: ignore[arg-type]
        )

        assert result == "3,4"

    @pytest.mark.asyncio()
    async def test_single_union_param_deserializes(self) -> None:
        @response_schema
        def to_str(value: int | str) -> str:
            return str(value)

        assert await to_str.validate('{"data": 42}', context=None) == "42"  # type: ignore[arg-type]
        assert await to_str.validate('{"data": "hello"}', context=None) == "hello"  # type: ignore[arg-type]

    @pytest.mark.asyncio()
    async def test_multi_params_with_defaults_uses_default(self) -> None:
        @response_schema
        def process(name: str, retries: int = 3) -> str:
            return f"{name} x{retries}"

        result = await process.validate(
            '{"name": "test"}',
            context=None,  # type: ignore[arg-type]
        )

        assert result == "test x3"

    @pytest.mark.asyncio()
    async def test_multi_params_with_defaults_overridden(self) -> None:
        @response_schema
        def process(name: str, retries: int = 3) -> str:
            return f"{name} x{retries}"

        result = await process.validate(
            '{"name": "test", "retries": 5}',
            context=None,  # type: ignore[arg-type]
        )

        assert result == "test x5"

    @pytest.mark.asyncio()
    async def test_multi_params_deserializes_json(self) -> None:
        @response_schema
        def greet(name: str, age: int) -> str:
            return f"{name} is {age}"

        result = await greet.validate(
            '{"name": "Alice", "age": 30}',
            context=None,  # type: ignore[arg-type]
        )

        assert result == "Alice is 30"

    @pytest.mark.asyncio()
    async def test_async_multi_params(self) -> None:
        @response_schema
        async def greet(name: str, age: int) -> str:
            return f"{name} is {age}"

        result = await greet.validate(
            '{"name": "Bob", "age": 25}',
            context=None,  # type: ignore[arg-type]
        )

        assert result == "Bob is 25"


class TestDependencyInjection:
    @staticmethod
    def _make_context(
        *,
        variables: dict[str, Any] | None = None,
        dependencies: dict[Any, Any] | None = None,
    ) -> "Context":
        return Context(
            stream=MemoryStream(),
            variables=variables or {},
            dependencies=dependencies or {},
        )

    @pytest.mark.asyncio()
    async def test_context_injected(self) -> None:
        @response_schema
        def parse(content: str, ctx: Context) -> dict[str, Any]:
            return {"value": content, "has_vars": "key" in ctx.variables}

        context = self._make_context(variables={"key": "val"})

        result = await parse.validate("hello", context=context)

        assert result == {"value": "hello", "has_vars": True}

    @pytest.mark.asyncio()
    async def test_variable_injected(self) -> None:
        @response_schema
        def parse(content: str, lang: Annotated[str, Variable()]) -> str:
            return f"{content} ({lang})"

        context = self._make_context(variables={"lang": "en"})

        result = await parse.validate("hello", context=context)

        assert result == "hello (en)"

    @pytest.mark.asyncio()
    async def test_variable_with_custom_name(self) -> None:
        @response_schema
        def parse(content: str, lang: Annotated[str, Variable("language")]) -> str:
            return f"{content} ({lang})"

        context = self._make_context(variables={"language": "fr"})

        result = await parse.validate("hello", context=context)

        assert result == "hello (fr)"

    @pytest.mark.asyncio()
    async def test_variable_with_default(self) -> None:
        @response_schema
        def parse(content: str, lang: Annotated[str, Variable(default="en")]) -> str:
            return f"{content} ({lang})"

        context = self._make_context()

        result = await parse.validate("hello", context=context)

        assert result == "hello (en)"

    @pytest.mark.asyncio()
    async def test_depends_injected(self) -> None:
        def get_suffix() -> str:
            return "!!!"

        @response_schema
        def parse(content: str, suffix: Annotated[str, Depends(get_suffix)]) -> str:
            return content + suffix

        context = self._make_context()

        result = await parse.validate("hello", context=context)

        assert result == "hello!!!"

    @pytest.mark.asyncio()
    async def test_all_di_together(self) -> None:
        def get_separator() -> str:
            return " | "

        @response_schema
        def parse(
            content: str,
            ctx: Context,
            lang: Annotated[str, Variable()],
            sep: Annotated[str, Depends(get_separator)],
        ) -> str:
            prefix = "stream" if ctx.stream is not None else "no-stream"
            return sep.join([prefix, content, lang])

        context = self._make_context(variables={"lang": "en"})

        result = await parse.validate("hello", context=context)

        assert result == "stream | hello | en"

    @pytest.mark.asyncio()
    async def test_di_excluded_from_multi_param_schema(self) -> None:
        @response_schema
        def parse(
            name: str,
            age: int,
            ctx: Context,
            lang: Annotated[str, Variable(default="en")],
        ) -> str:
            return f"{name}({age}) [{lang}]"

        # DI params excluded from schema (multi-arg message body is a flat object, not ``data``-wrapped)
        assert parse.json_schema == IsPartialDict({
            "type": "object",
            "properties": IsPartialDict({
                "name": IsPartialDict({"type": "string"}),
                "age": IsPartialDict({"type": "integer"}),
            }),
            "required": ["name", "age"],
        })

        # But resolved at runtime
        context = self._make_context(variables={"lang": "fr"})

        result = await parse.validate(
            '{"name": "Alice", "age": 30}',
            context=context,
        )

        assert result == "Alice(30) [fr]"
