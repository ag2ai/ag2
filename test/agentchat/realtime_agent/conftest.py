# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import autogen

from ...conftest import MOCK_OPEN_AI_API_KEY  # noqa: E402
from ..test_assistant_agent import KEY_LOC, OAI_CONFIG_LIST
from .realtime_test_utils import Credentials


@pytest.fixture
def credentials() -> Credentials:
    """Fixture to load the LLM config."""
    config_list = autogen.config_list_from_json(
        OAI_CONFIG_LIST,
        filter_dict={
            "tags": ["gpt-4o-realtime"],
        },
        file_location=KEY_LOC,
    )
    assert config_list, "No config list found"

    return Credentials(
        llm_config={
            "config_list": config_list,
            "temperature": 0.6,
        }
    )


@pytest.fixture
def mock_credentials() -> Credentials:
    llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": MOCK_OPEN_AI_API_KEY,
            },
        ],
        "temperature": 0.8,
    }

    return Credentials(llm_config=llm_config)
