# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from autogen.interoperability import Interoperable


def test_interoperable() -> None:
    assert Interoperable is not None
