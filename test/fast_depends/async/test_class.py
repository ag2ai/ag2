# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/https://github.com/Lancetnik/FastDepends are under the MIT License.
# SPDX-License-Identifier: MIT

import pytest

from autogen.fast_depends import Depends, inject


def _get_var():
    return 1


class Class:
    @inject
    def __init__(self, a=Depends(_get_var)) -> None:
        self.a = a

    @inject
    async def calc(self, a=Depends(_get_var)) -> int:
        return a + self.a


@pytest.mark.anyio
async def test_class():
    assert await Class().calc() == 2
