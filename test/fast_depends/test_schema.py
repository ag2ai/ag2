# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/https://github.com/Lancetnik/FastDepends are under the MIT License.
# SPDX-License-Identifier: MIT

from typing import Optional

from dirty_equals import IsDict, IsPartialDict
from pydantic import BaseModel, Field

from autogen.fast_depends import Depends
from autogen.fast_depends._compat import PYDANTIC_V2
from autogen.fast_depends.core import build_call_model
from autogen.fast_depends.schema import get_schema

REF_KEY = "$defs" if PYDANTIC_V2 else "definitions"


def test_base() -> None:
    def handler() -> None:
        pass

    schema = get_schema(build_call_model(handler))
    assert schema == {"title": "handler", "type": "null"}, schema


class TestNoType:
    def test_no_type(self) -> None:
        def handler(a) -> None:
            pass

        schema = get_schema(build_call_model(handler))
        assert schema == {
            "properties": {"a": {"title": "A"}},
            "required": ["a"],
            "title": "handler",
            "type": "object",
        }, schema

    def test_no_type_embeded(self) -> None:
        def handler(a) -> None:
            pass

        schema = get_schema(build_call_model(handler), embed=True)
        assert schema == {"title": "A"}, schema

    def test_no_type_with_default(self) -> None:
        def handler(a=None) -> None:
            pass

        schema = get_schema(build_call_model(handler))
        assert schema == {
            "properties": {"a": IsPartialDict({"title": "A"})},
            "title": "handler",
            "type": "object",
        }, schema

    def test_no_type_with_default_and_embed(self) -> None:
        def handler(a=None) -> None:
            pass

        schema = get_schema(build_call_model(handler), embed=True)
        assert schema == IsPartialDict({"title": "A"}), schema


class TestOneArg:
    def test_one_arg(self) -> None:
        def handler(a: int) -> None:
            pass

        schema = get_schema(build_call_model(handler))
        assert schema == {
            "properties": {"a": {"title": "A", "type": "integer"}},
            "required": ["a"],
            "title": "handler",
            "type": "object",
        }, schema

    def test_one_arg_with_embed(self) -> None:
        def handler(a: int) -> None:
            pass

        schema = get_schema(build_call_model(handler), embed=True)
        assert schema == {"title": "A", "type": "integer"}, schema

    def test_one_arg_with_optional(self) -> None:
        def handler(a: Optional[int]) -> None:
            pass

        schema = get_schema(build_call_model(handler))
        assert schema == {
            "properties": {
                "a": IsDict({"anyOf": [{"type": "integer"}, {"type": "null"}], "title": "A"})
                | IsDict({"type": "integer", "title": "A"})
            },
            "required": ["a"],
            "title": "handler",
            "type": "object",
        }, schema

    def test_one_arg_with_default(self) -> None:
        def handler(a: int = 0) -> None:
            pass

        schema = get_schema(build_call_model(handler))
        assert schema == {
            "properties": {"a": {"default": 0, "title": "A", "type": "integer"}},
            "title": "handler",
            "type": "object",
        }, schema

    def test_one_arg_with_default_and_embed(self) -> None:
        def handler(a: int = 0) -> None:
            pass

        schema = get_schema(build_call_model(handler), embed=True)
        assert schema == {"default": 0, "title": "A", "type": "integer"}, schema


class TestOneArgWithModel:
    def test_base(self) -> None:
        class Model(BaseModel):
            a: int

        def handler(a: Model) -> None:
            pass

        schema = get_schema(build_call_model(handler))
        assert schema == {
            REF_KEY: {
                "Model": {
                    "properties": {"a": {"title": "A", "type": "integer"}},
                    "required": ["a"],
                    "title": "Model",
                    "type": "object",
                }
            },
            "properties": {"a": {"$ref": f"#/{REF_KEY}/Model"}},
            "required": ["a"],
            "title": "handler",
            "type": "object",
        }, schema

    def test_resolved_model(self) -> None:
        class Model(BaseModel):
            a: int

        def handler(a: Model) -> None:
            pass

        schema = get_schema(build_call_model(handler), resolve_refs=True)
        assert schema == {
            "properties": {
                "a": {
                    "properties": {"a": {"title": "A", "type": "integer"}},
                    "required": ["a"],
                    "title": "Model",
                    "type": "object",
                }
            },
            "required": ["a"],
            "title": "handler",
            "type": "object",
        }, schema

    def test_optional_model(self) -> None:
        class Model(BaseModel):
            a: int

        def handler(a: Optional[Model] = None) -> None:
            pass

        schema = get_schema(build_call_model(handler), resolve_refs=True)
        assert schema == {
            "properties": {
                "a": IsDict({
                    "anyOf": [
                        {
                            "properties": {"a": {"title": "A", "type": "integer"}},
                            "required": ["a"],
                            "title": "Model",
                            "type": "object",
                        },
                        {"type": "null"},
                    ],
                    "default": None,
                })
                | IsDict({
                    "properties": {"a": {"title": "A", "type": "integer"}},
                    "required": ["a"],
                    "title": "Model",
                    "type": "object",
                })
            },
            "title": "handler",
            "type": "object",
        }, schema

    def test_optional_embeded_model(self) -> None:
        class Model(BaseModel):
            a: int

        def handler(a: Optional[Model]) -> None:
            pass

        schema = get_schema(build_call_model(handler), resolve_refs=True, embed=True)
        assert schema == IsDict({
            "anyOf": [
                {
                    "properties": {"a": {"title": "A", "type": "integer"}},
                    "required": ["a"],
                    "title": "Model",
                    "type": "object",
                },
                {"type": "null"},
            ]
        }) | IsDict({
            "properties": {"a": {"title": "A", "type": "integer"}},
            "required": ["a"],
            "title": "Model",
            "type": "object",
        }), schema

    def test_nested_resolved_model(self) -> None:
        class Model2(BaseModel):
            a: int

        class Model(BaseModel):
            a: Model2

        def handler(a: Model) -> None:
            pass

        schema = get_schema(build_call_model(handler), resolve_refs=True)
        assert schema == {
            "properties": {
                "a": {
                    "properties": {
                        "a": {
                            "properties": {"a": {"title": "A", "type": "integer"}},
                            "required": ["a"],
                            "title": "Model2",
                            "type": "object",
                        }
                    },
                    "required": ["a"],
                    "title": "Model",
                    "type": "object",
                }
            },
            "required": ["a"],
            "title": "handler",
            "type": "object",
        }, schema

    def test_embeded_model(self) -> None:
        class Model(BaseModel):
            a: int

        def handler(a: Model) -> None:
            pass

        schema = get_schema(build_call_model(handler), resolve_refs=True, embed=True)
        assert schema == {
            "properties": {"a": {"title": "A", "type": "integer"}},
            "required": ["a"],
            "title": "Model",
            "type": "object",
        }, schema

    def test_embeded_resolved_model(self) -> None:
        class Model2(BaseModel):
            a: int

        class Model(BaseModel):
            a: Model2

        def handler(a: Model) -> None:
            pass

        schema = get_schema(build_call_model(handler), resolve_refs=True, embed=True)
        assert schema == {
            "properties": {
                "a": {
                    "properties": {"a": {"title": "A", "type": "integer"}},
                    "required": ["a"],
                    "title": "Model2",
                    "type": "object",
                }
            },
            "required": ["a"],
            "title": "Model",
            "type": "object",
        }, schema


class TestMultiArgs:
    def test_base(self) -> None:
        def handler(a, b) -> None:
            pass

        schema = get_schema(build_call_model(handler))
        assert schema == {
            "properties": {"a": {"title": "A"}, "b": {"title": "B"}},
            "required": ["a", "b"],
            "title": "handler",
            "type": "object",
        }, schema

    def test_types_and_default(self) -> None:
        def handler(a: str, b: int = 0) -> None:
            pass

        schema = get_schema(build_call_model(handler))
        assert schema == {
            "properties": {
                "a": {"title": "A", "type": "string"},
                "b": {"default": 0, "title": "B", "type": "integer"},
            },
            "required": ["a"],
            "title": "handler",
            "type": "object",
        }, schema

    def test_ignores_embed(self) -> None:
        def handler(a: str, b: int = 0) -> None:
            pass

        schema = get_schema(build_call_model(handler), embed=True)
        assert schema == {
            "properties": {
                "a": {"title": "A", "type": "string"},
                "b": {"default": 0, "title": "B", "type": "integer"},
            },
            "required": ["a"],
            "title": "handler",
            "type": "object",
        }, schema

    def test_model(self) -> None:
        class Model(BaseModel):
            a: int = Field(0, description="description")

        def handler(a: str, b: Model) -> None:
            pass

        schema = get_schema(build_call_model(handler))
        assert schema == {
            REF_KEY: {
                "Model": {
                    "properties": {
                        "a": {
                            "default": 0,
                            "description": "description",
                            "title": "A",
                            "type": "integer",
                        }
                    },
                    "title": "Model",
                    "type": "object",
                }
            },
            "properties": {
                "a": {"title": "A", "type": "string"},
                "b": {"$ref": f"#/{REF_KEY}/Model"},
            },
            "required": ["a", "b"],
            "title": "handler",
            "type": "object",
        }, schema

    def test_resolved_model(self) -> None:
        class Model(BaseModel):
            a: int = Field(0, description="description")

        def handler(a: str, b: Model) -> None:
            pass

        schema = get_schema(build_call_model(handler), resolve_refs=True)
        assert schema == {
            "properties": {
                "a": {"title": "A", "type": "string"},
                "b": {
                    "properties": {
                        "a": {
                            "default": 0,
                            "description": "description",
                            "title": "A",
                            "type": "integer",
                        }
                    },
                    "title": "Model",
                    "type": "object",
                },
            },
            "required": ["a", "b"],
            "title": "handler",
            "type": "object",
        }, schema


def test_depends() -> None:
    class Model(BaseModel):
        a: int = Field(0, description="description")

    class CustomClass:
        pass

    def dep4(d: Model) -> None: ...

    def dep2(c: float = Field(..., title="best")) -> None: ...

    def dep(a: float = 1, d: CustomClass = Depends(dep2)) -> None: ...

    def handler(
        b: str = Field(""),
        a=Depends(dep),
    ) -> None: ...

    schema = get_schema(
        build_call_model(handler, extra_dependencies=(Depends(dep4),)),
        resolve_refs=True,
        embed=True,
    )

    assert schema == {
        "properties": {
            "a": {"default": 1, "title": "A", "type": "number"},
            "b": {"default": "", "title": "B", "type": "string"},
            "c": {"title": "best", "type": "number"},
            "d": {
                "properties": {
                    "a": {
                        "default": 0,
                        "description": "description",
                        "title": "A",
                        "type": "integer",
                    }
                },
                "title": "Model",
                "type": "object",
            },
        },
        "required": ["c", "d"],
        "title": "handler",
        "type": "object",
    }, schema
