# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from ag2 import Context


@pytest.fixture
def context() -> Context:
    return Context(stream=MagicMock())


@pytest.fixture
def make_context() -> Callable[..., Context]:
    def _make(**variables: object) -> Context:
        return Context(stream=MagicMock(), variables=variables)

    return _make
