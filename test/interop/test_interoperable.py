# Copyright (c) 2023 - 2025, AG2ai, Inc, AG2AI OSS project maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.interop import Interoperable


def test_interoperable() -> None:
    assert Interoperable is not None
