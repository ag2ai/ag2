# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def gemini_config() -> MagicMock:
    config = MagicMock()
    config.api_key = "test-key"
    return config
