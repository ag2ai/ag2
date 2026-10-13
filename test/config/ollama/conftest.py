# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from test.config.ollama._helpers import FakeOllama


@pytest.fixture
def ollama() -> FakeOllama:
    return FakeOllama()
