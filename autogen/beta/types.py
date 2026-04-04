# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import types
from typing import Literal, TypeAlias, TypeVar

ClassInfo: TypeAlias = type | types.UnionType | tuple["ClassInfo", ...]


class Omit:
    def __bool__(self) -> Literal[False]:
        return False


omit = Omit()

_T = TypeVar("_T")

Omittable = _T | Omit
