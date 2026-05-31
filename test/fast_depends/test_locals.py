# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/https://github.com/Lancetnik/FastDepends are under the MIT License.
# SPDX-License-Identifier: MIT

from pydantic import BaseModel

from autogen.fast_depends import inject


def wrap(func):
    return inject(func)


def test_locals():
    class M(BaseModel):
        a: str

    @wrap
    def m(a: M) -> M:
        return a

    m(a={"a": "Hi!"})
