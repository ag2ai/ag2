# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from autogen.oai.deepseek import DeepSeekLLMConfigEntry


def test_deepseek_llm_config_entry() -> None:
    deepseek_llm_config = DeepSeekLLMConfigEntry(
        api_key="fake_api_key",
        model="deepseek-chat",
    )

    expected = {
        "api_type": "deepseek",
        "api_key": "fake_api_key",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
        "max_tokens": 10000,
        "temperature": 0.5,
        "top_p": 0.2,
        "tags": [],
    }
    actual = deepseek_llm_config.model_dump()
    assert actual == expected, actual

    with pytest.raises(ValidationError) as e:
        deepseek_llm_config = DeepSeekLLMConfigEntry(
            model="deepseek-chat",
            temperature=1,
            top_p=0.8,
        )
    assert "Value error, temperature and top_p cannot be set at the same time" in str(e.value)
