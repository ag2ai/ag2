# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def zai_config() -> MagicMock:
    config = MagicMock()
    config.api_key = "test-key"
    config.base_url = None
    config.timeout = None
    config.max_retries = 3
    config.http_client = None
    config.custom_headers = None
    config.disable_token_cache = True
    config.source_channel = None
    return config
