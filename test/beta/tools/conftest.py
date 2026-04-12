# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from autogen.beta.context import ConversationContext


@pytest.fixture
def context() -> ConversationContext:
    return ConversationContext(stream=MagicMock())


@pytest.fixture
def make_context() -> Callable[..., ConversationContext]:

    def _make(**variables: object) -> ConversationContext:
        return ConversationContext(stream=MagicMock(), variables=variables)

    return _make
