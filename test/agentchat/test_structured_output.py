# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

import os
import sys

import pytest
from pydantic import BaseModel, ValidationError
from test_assistant_agent import KEY_LOC, OAI_CONFIG_LIST

import autogen

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from conftest import reason, skip_openai  # noqa: E402


@pytest.mark.skipif(skip_openai, reason=reason)
def test_structured_output():
    config_list = autogen.config_list_from_json(
        OAI_CONFIG_LIST,
        file_location=KEY_LOC,
        filter_dict={
            "model": ["gpt-4o", "gpt-4o-mini"],
        },
    )

    class ResponseModel(BaseModel):
        question: str
        short_answer: str
        reasoning: str
        difficulty: float

    llm_config = {"config_list": config_list, "cache_seed": 41}

    user_proxy = autogen.UserProxyAgent(
        name="User_proxy",
        system_message="A human admin.",
        human_input_mode="NEVER",
    )

    assistant = autogen.AssistantAgent(
        name="Assistant",
        llm_config=llm_config,
        response_format=ResponseModel,
    )

    chat_result = user_proxy.initiate_chat(
        assistant,
        message="What is the air-speed velocity of an unladen swallow?",
        max_turns=1,
        summary_method="last_msg",
    )

    try:
        ResponseModel.model_validate_json(chat_result.chat_history[-1]["content"])
    except ValidationError as e:
        raise AssertionError(f"Agent did not return a structured report. Exception: {e}")
