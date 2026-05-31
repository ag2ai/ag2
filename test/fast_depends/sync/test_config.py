# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/https://github.com/Lancetnik/FastDepends are under the MIT License.
# SPDX-License-Identifier: MIT

import pytest
from pydantic import ValidationError

from autogen.fast_depends import Depends, inject
from autogen.fast_depends._compat import PYDANTIC_V2


def dep(a: str):
    return a


@inject(pydantic_config={"str_max_length" if PYDANTIC_V2 else "max_anystr_length": 1})
def limited_str(a=Depends(dep)): ...


@inject()
def regular(a=Depends(dep)):
    return a


def test_config():
    regular("123")

    with pytest.raises(ValidationError):
        limited_str("123")
