# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

import inspect
import unittest.mock
from enum import Enum
from typing import Annotated, Any, Callable, Literal, Optional, Set, Tuple, Union

import pytest
from pydantic import BaseModel, Field

from autogen.tools.dependency_injection import Field as AG2Field
from autogen.tools.function_utils import (
    get_default_values,
    get_function_schema,
    get_load_param_if_needed_function,
    get_missing_annotations,
    get_param_annotations,
    get_parameter_json_schema,
    get_parameters,
    get_required_params,
    get_typed_annotation,
    get_typed_return_annotation,
    get_typed_signature,
    load_basemodels_if_needed,
    serialize_to_str,
)


def f(a: Annotated[str, "Parameter a"], b: int = 2, c: Annotated[float, "Parameter c"] = 0.1, *, d):  # type: ignore[no-untyped-def]
    pass


def g(  # type: ignore[empty-body]
    a: Annotated[str, AG2Field(description="Parameter a")],
    b: int = 2,
    c: Annotated[float, AG2Field(description="Parameter c")] = 0.1,
    *,
    d: dict[str, tuple[Optional[int], list[float]]],
) -> str:
    pass


async def a_g(  # type: ignore[empty-body]
    a: Annotated[str, AG2Field(description="Parameter a")],
    b: int = 2,
    c: Annotated[float, AG2Field(description="Parameter c")] = 0.1,
    *,
    d: dict[str, tuple[Optional[int], list[float]]],
) -> str:
    pass


def test_get_typed_annotation() -> None:
    globalns = getattr(f, "__globals__", {})
    assert get_typed_annotation(str, globalns) == str
    assert get_typed_annotation("float", globalns) == float


def test_get_typed_signature() -> None:
    assert get_typed_signature(f).parameters == inspect.signature(f).parameters
    assert get_typed_signature(g).parameters == inspect.signature(g).parameters


def test_get_typed_return_annotation() -> None:
    assert get_typed_return_annotation(f) is None
    assert get_typed_return_annotation(g) == str


def test_get_parameter_json_schema() -> None:
    assert get_parameter_json_schema("c", str, {}) == {"type": "string", "description": "c"}
    assert get_parameter_json_schema("c", str, {"c": "ccc"}) == {"type": "string", "description": "c", "default": "ccc"}

    assert get_parameter_json_schema("a", Annotated[str, AG2Field(description="parameter a")], {}) == {
        "type": "string",
        "description": "parameter a",
    }
    assert get_parameter_json_schema("a", Annotated[str, AG2Field(description="parameter a")], {"a": "3.14"}) == {
        "type": "string",
        "description": "parameter a",
        "default": "3.14",
    }
    assert get_parameter_json_schema(
        "d", Annotated[Optional[str], AG2Field(description="parameter d")], {"d": None}
    ) == {
        "anyOf": [{"type": "string"}, {"type": "null"}],
        "default": None,
        "description": "parameter d",
    }

    class B(BaseModel):
        b: float
        c: str

    expected: dict[str, Any] = {
        "description": "b",
        "properties": {"b": {"title": "B", "type": "number"}, "c": {"title": "C", "type": "string"}},
        "required": ["b", "c"],
        "title": "B",
        "type": "object",
    }
    assert get_parameter_json_schema("b", B, {}) == expected

    expected["default"] = B(b=1.2, c="3.4")
    assert get_parameter_json_schema("b", B, {"b": B(b=1.2, c="3.4")}) == expected


def test_get_required_params() -> None:
    assert get_required_params(inspect.signature(f)) == ["a", "d"]
    assert get_required_params(inspect.signature(g)) == ["a", "d"]


def test_get_default_values() -> None:
    assert get_default_values(inspect.signature(f)) == {"b": 2, "c": 0.1}
    assert get_default_values(inspect.signature(g)) == {"b": 2, "c": 0.1}


def test_get_param_annotations() -> None:
    def f(a: Annotated[str, "Parameter a"], b=1, c: Annotated[float, "Parameter c"] = 1.0):  # type: ignore[no-untyped-def]
        pass

    expected = {"a": Annotated[str, "Parameter a"], "c": Annotated[float, "Parameter c"]}

    typed_signature = get_typed_signature(f)
    param_annotations = get_param_annotations(typed_signature)

    assert param_annotations == expected, param_annotations  # type: ignore[comparison-overlap]


def test_get_missing_annotations() -> None:
    def _f1(a: str, b=2):  # type: ignore[no-untyped-def]
        pass

    missing, unannotated_with_default = get_missing_annotations(get_typed_signature(_f1), ["a"])
    assert missing == set()
    assert unannotated_with_default == {"b"}

    def _f2(a: str, b) -> str:  # type: ignore[empty-body,no-untyped-def]
        """Ok"""

    missing, unannotated_with_default = get_missing_annotations(get_typed_signature(_f2), ["a", "b"])
    assert missing == {"b"}
    assert unannotated_with_default == set()

    def _f3() -> None:
        pass

    missing, unannotated_with_default = get_missing_annotations(get_typed_signature(_f3), [])
    assert missing == set()
    assert unannotated_with_default == set()


def test_get_parameters() -> None:
    def f(  # type: ignore[no-untyped-def]
        a: Annotated[str, AG2Field(description="Parameter a")],
        b=1,  # type: ignore[no-untyped-def]
        c: Annotated[float, AG2Field(description="Parameter c")] = 1.0,
    ):
        pass

    typed_signature = get_typed_signature(f)
    param_annotations = get_param_annotations(typed_signature)
    required = get_required_params(typed_signature)
    default_values = get_default_values(typed_signature)

    expected = {
        "type": "object",
        "properties": {
            "a": {"type": "string", "description": "Parameter a"},
            "c": {"type": "number", "description": "Parameter c", "default": 1.0},
        },
        "required": ["a"],
    }

    actual = (get_parameters(required, param_annotations, default_values)).model_dump()

    assert actual == expected, actual


def test_get_function_schema_no_return_type() -> None:
    def f(a: Annotated[str, AG2Field(description="Parameter a")], b: int, c: float = 0.1):  # type: ignore[no-untyped-def]
        pass

    expected = (
        "The return type of the function 'f' is not annotated. Although annotating it is "
        + "optional, the function should return either a string, a subclass of 'pydantic.BaseModel'."
    )

    with unittest.mock.patch("autogen.tools.function_utils.logger.warning") as mock_logger_warning:
        get_function_schema(f, description="function g")

        mock_logger_warning.assert_called_once_with(expected)


def test_get_function_schema_unannotated_with_default() -> None:
    with unittest.mock.patch("autogen.tools.function_utils.logger.warning") as mock_logger_warning:

        def f(  # type: ignore[no-untyped-def]
            a: Annotated[str, AG2Field(description="Parameter a")],
            b=2,
            c: Annotated[float, AG2Field(description="Parameter c")] = 0.1,
            d="whatever",
            e=None,
        ) -> str:
            return "ok"

        get_function_schema(f, description="function f")

        mock_logger_warning.assert_called_once_with(
            "The following parameters of the function 'f' with default values are not annotated: 'b', 'd', 'e'."
        )


def test_get_function_schema_missing() -> None:
    def f(a: Annotated[str, "Parameter a"], b, c: Annotated[float, "Parameter c"] = 0.1) -> float:  # type: ignore[no-untyped-def, empty-body]
        pass

    expected = (
        "All parameters of the function 'f' without default values must be annotated. "
        + "The annotations are missing for the following parameters: 'b'"
    )

    with pytest.raises(TypeError) as e:
        get_function_schema(f, description="function f")

    assert str(e.value) == expected, e.value


def test_get_function_schema() -> None:
    expected = {
        "type": "function",
        "function": {
            "description": "function g",
            "name": "fancy name for g",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "string", "description": "Parameter a"},
                    "b": {"type": "integer", "description": "b", "default": 2},
                    "c": {"type": "number", "description": "Parameter c", "default": 0.1},
                    "d": {
                        "additionalProperties": {
                            "maxItems": 2,
                            "minItems": 2,
                            "prefixItems": [
                                {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                                {"items": {"type": "number"}, "type": "array"},
                            ],
                            "type": "array",
                        },
                        "type": "object",
                        "description": "d",
                    },
                },
                "required": ["a", "d"],
            },
        },
    }

    actual = get_function_schema(g, description="function g", name="fancy name for g")
    assert actual == expected, actual

    actual = get_function_schema(a_g, description="function g", name="fancy name for g")
    assert actual == expected, actual


CurrencySymbol = Literal["USD", "EUR"]


class Currency(BaseModel):
    currency: Annotated[CurrencySymbol, Field(..., description="Currency code")]
    amount: Annotated[float, Field(100.0, description="Amount of money in the currency")]


def test_get_function_schema_pydantic() -> None:
    from autogen.tools.dependency_injection import _string_metadata_to_description_field

    def currency_calculator(  # type: ignore[empty-body]
        base: Annotated[Currency, "Base currency: amount and currency symbol"],
        quote_currency: Annotated[CurrencySymbol, "Quote currency symbol (default: 'EUR')"] = "EUR",
    ) -> Currency:
        pass

    currency_calculator = _string_metadata_to_description_field(currency_calculator)

    expected = {
        "type": "function",
        "function": {
            "description": "Currency exchange calculator.",
            "name": "currency_calculator",
            "parameters": {
                "type": "object",
                "properties": {
                    "base": {
                        "properties": {
                            "currency": {
                                "description": "Currency code",
                                "enum": ["USD", "EUR"],
                                "title": "Currency",
                                "type": "string",
                            },
                            "amount": {
                                "default": 100.0,
                                "description": "Amount of money in the currency",
                                "title": "Amount",
                                "type": "number",
                            },
                        },
                        "required": ["currency"],
                        "title": "Currency",
                        "type": "object",
                        "description": "Base currency: amount and currency symbol",
                    },
                    "quote_currency": {
                        "enum": ["USD", "EUR"],
                        "type": "string",
                        "default": "EUR",
                        "description": "Quote currency symbol (default: 'EUR')",
                    },
                },
                "required": ["base"],
            },
        },
    }

    actual = get_function_schema(
        currency_calculator, description="Currency exchange calculator.", name="currency_calculator"
    )

    assert actual == expected, actual


# test_load_param_if_needed_function tests


class NonBaseModelClass:
    def __init__(self, value: int):
        self.value = value


def test_get_load_param_if_needed_function() -> None:
    assert get_load_param_if_needed_function(CurrencySymbol) is None
    assert get_load_param_if_needed_function(Currency)({"currency": "USD", "amount": 123.45}, Currency) == Currency(  # type: ignore[misc]
        currency="USD", amount=123.45
    )

    f = get_load_param_if_needed_function(Annotated[Currency, "amount and a symbol of a currency"])
    actual = f({"currency": "USD", "amount": 123.45}, Currency)  # type: ignore[misc]
    expected = Currency(currency="USD", amount=123.45)
    assert actual == expected, actual


def test_get_load_param_if_needed_function_base_model() -> None:
    """Tests direct BaseModel subclasses."""
    loader = get_load_param_if_needed_function(Currency)
    assert loader is not None, "Should return a loader function for BaseModel"
    assert callable(loader), "Returned value should be callable"
    instance = loader({"currency": "USD", "amount": 123.45}, Currency)
    assert isinstance(instance, Currency)
    assert instance == Currency(currency="USD", amount=123.45)


def test_get_load_param_if_needed_function_annotated_base_model() -> None:
    """Tests Annotated BaseModel subclasses."""
    loader = get_load_param_if_needed_function(Annotated[Currency, "Description"])
    assert loader is not None, "Should return a loader for Annotated[BaseModel]"
    assert callable(loader), "Returned value should be callable"
    instance = loader({"currency": "EUR", "amount": 99.0}, Currency)
    assert isinstance(instance, Currency)
    assert instance == Currency(currency="EUR", amount=99.0)


def test_get_load_param_if_needed_function_basic_types() -> None:
    """Tests basic built-in types."""
    assert get_load_param_if_needed_function(str) is None
    assert get_load_param_if_needed_function(int) is None
    assert get_load_param_if_needed_function(float) is None
    assert get_load_param_if_needed_function(bool) is None
    assert get_load_param_if_needed_function(bytes) is None
    assert get_load_param_if_needed_function(type(None)) is None


def test_get_load_param_if_needed_function_plain_classes() -> None:
    """Tests non-BaseModel classes."""
    assert get_load_param_if_needed_function(NonBaseModelClass) is None
    assert get_load_param_if_needed_function(Enum) is None  # The Enum base class itself
    assert get_load_param_if_needed_function(CurrencySymbol) is None  # Specific Enum


def test_get_load_param_if_needed_function_generic_aliases_fixed() -> None:
    """Tests the generic types that previously caused errors."""
    assert get_load_param_if_needed_function(list[str]) is None
    assert get_load_param_if_needed_function(dict[str, int]) is None
    assert get_load_param_if_needed_function(list[Currency]) is None  # List *containing* BaseModel
    assert get_load_param_if_needed_function(dict[str, Currency]) is None  # Dict *containing* BaseModel


def test_get_load_param_if_needed_function_other_typing_constructs() -> None:
    """Tests other constructs from the typing module."""
    assert get_load_param_if_needed_function(Any) is None
    assert get_load_param_if_needed_function(Union[str, int]) is None
    assert get_load_param_if_needed_function(Optional[str]) is None  # Equivalent to Union[str, None]
    assert get_load_param_if_needed_function(Optional[Currency]) is None  # Optional BaseModel
    assert get_load_param_if_needed_function(Callable[[], str]) is None
    assert get_load_param_if_needed_function(Tuple[int, str]) is None
    assert get_load_param_if_needed_function(Set[float]) is None
    assert get_load_param_if_needed_function(list) is None  # The raw list type
    assert get_load_param_if_needed_function(dict) is None  # The raw dict type


def test_get_load_param_if_needed_function_annotated_non_base_model() -> None:
    """Tests Annotated non-BaseModel types."""
    assert get_load_param_if_needed_function(Annotated[str, "A string"]) is None
    assert get_load_param_if_needed_function(Annotated[list[int], "A list of ints"]) is None


def test_get_load_param_if_needed_function_nested_annotated() -> None:
    """Tests nested Annotated types wrapping a BaseModel."""
    loader = get_load_param_if_needed_function(Annotated[Annotated[Currency, "Inner"], "Outer"])
    assert loader is not None, "Should find BaseModel inside nested Annotated"
    assert callable(loader), "Returned value should be callable"
    instance = loader({"currency": "USD", "amount": 1.0}, Currency)
    assert isinstance(instance, Currency)
    assert instance == Currency(currency="USD", amount=1.0)


def test_load_basemodels_if_needed_sync() -> None:
    @load_basemodels_if_needed
    def f(
        base: Annotated[Currency, "Base currency"],
        quote_currency: Annotated[CurrencySymbol, "Quote currency"] = "EUR",
    ) -> tuple[Currency, CurrencySymbol]:
        return base, quote_currency

    assert not inspect.iscoroutinefunction(f)

    actual = f(base={"currency": "USD", "amount": 123.45}, quote_currency="EUR")
    assert isinstance(actual[0], Currency)
    assert actual[0].amount == 123.45
    assert actual[0].currency == "USD"
    assert actual[1] == "EUR"


@pytest.mark.asyncio
async def test_load_basemodels_if_needed_async() -> None:
    @load_basemodels_if_needed
    async def f(
        base: Annotated[Currency, "Base currency"],
        quote_currency: Annotated[CurrencySymbol, "Quote currency"] = "EUR",
    ) -> tuple[Currency, CurrencySymbol]:
        return base, quote_currency

    assert inspect.iscoroutinefunction(f)

    actual = await f(base={"currency": "USD", "amount": 123.45}, quote_currency="EUR")
    assert isinstance(actual[0], Currency)
    assert actual[0].amount == 123.45
    assert actual[0].currency == "USD"
    assert actual[1] == "EUR"


def test_serialize_to_str_with_nonascii() -> None:
    assert serialize_to_str("中文") == "中文"


def test_serialize_to_json() -> None:
    assert serialize_to_str("abc") == "abc"
    assert serialize_to_str(123) == "123"
    assert serialize_to_str([123, 456]) == "[123, 456]"
    assert serialize_to_str({"a": 1, "b": 2.3}) == '{"a": 1, "b": 2.3}'.replace('"', "'")

    class A(BaseModel):
        a: int
        b: float
        c: str

    assert serialize_to_str(A(a=1, b=2.3, c="abc")) == '{"a":1,"b":2.3,"c":"abc"}'


def test_serilize_to_str_list_pydantic() -> None:
    class A(BaseModel):
        a: int
        b: float
        c: str

    assert (
        serialize_to_str([A(a=1, b=2.3, c="abc"), A(a=4, b=5.6, c="def")])
        == "[{'a': 1, 'b': 2.3, 'c': 'abc'}, {'a': 4, 'b': 5.6, 'c': 'def'}]"
    )
