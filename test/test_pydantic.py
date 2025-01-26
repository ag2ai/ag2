# Copyright (c) 2023 - 2025, AG2ai, Inc, AG2AI OSS project maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from typing import Annotated, Optional, Union

from pydantic import BaseModel

from autogen._pydantic import model_dump, model_dump_json, type2schema


def test_type2schema() -> None:
    assert type2schema(str) == {"type": "string"}
    assert type2schema(int) == {"type": "integer"}
    assert type2schema(float) == {"type": "number"}
    assert type2schema(bool) == {"type": "boolean"}
    assert type2schema(None) == {"type": "null"}
    assert type2schema(Optional[int]) == {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    assert type2schema(list[int]) == {"items": {"type": "integer"}, "type": "array"}
    assert type2schema(tuple[int, float, str]) == {
        "maxItems": 3,
        "minItems": 3,
        "prefixItems": [{"type": "integer"}, {"type": "number"}, {"type": "string"}],
        "type": "array",
    }
    assert type2schema(dict[str, int]) == {"additionalProperties": {"type": "integer"}, "type": "object"}
    assert type2schema(Annotated[str, "some text"]) == {"type": "string"}
    assert type2schema(Union[int, float]) == {"anyOf": [{"type": "integer"}, {"type": "number"}]}


def test_model_dump() -> None:
    class A(BaseModel):
        a: str
        b: int = 2

    assert model_dump(A(a="aaa")) == {"a": "aaa", "b": 2}


def test_model_dump_json() -> None:
    class A(BaseModel):
        a: str
        b: int = 2

    assert model_dump_json(A(a="aaa")).replace(" ", "") == '{"a":"aaa","b":2}'
