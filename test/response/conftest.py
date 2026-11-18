# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Context, MemoryStream


@pytest.fixture
def context() -> Context:
    return Context(stream=MemoryStream())
