# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# !/usr/bin/env python3 -m pytest

from unittest.mock import AsyncMock

import pytest

import autogen

from ..conftest import Credentials, credentials_all_llms, suppress_gemini_resource_exhausted


async def _test_async_get_human_input(credentials: Credentials) -> None:
    config_list = credentials.config_list

    # create an AssistantAgent instance named "assistant"
    assistant = autogen.AssistantAgent(
        name="assistant",
        max_consecutive_auto_reply=2,
        llm_config={"config_list": config_list, "temperature": 0},
    )

    user_proxy = autogen.UserProxyAgent(name="user", human_input_mode="ALWAYS", code_execution_config=False)

    user_proxy.a_get_human_input = AsyncMock(return_value="This is a test")

    user_proxy.register_reply([autogen.Agent, None], autogen.ConversableAgent.a_check_termination_and_human_reply)

    await user_proxy.a_initiate_chat(assistant, clear_history=True, message="Hello.")
    # Test without message
    res = await user_proxy.a_initiate_chat(assistant, clear_history=True, summary_method="reflection_with_llm")
    # Assert that custom a_get_human_input was called at least once
    user_proxy.a_get_human_input.assert_called()
    print("Result summary:", res.summary)
    print("Human input:", res.human_input)


@pytest.mark.parametrize("credentials_from_test_param", credentials_all_llms, indirect=True)
@suppress_gemini_resource_exhausted
@pytest.mark.asyncio
async def test_async_get_human_input(
    credentials_from_test_param: Credentials,
) -> None:
    await _test_async_get_human_input(credentials_from_test_param)


async def _test_async_max_turn(credentials: Credentials):
    config_list = credentials.config_list

    # create an AssistantAgent instance named "assistant"
    assistant = autogen.AssistantAgent(
        name="assistant",
        max_consecutive_auto_reply=10,
        llm_config={
            "seed": 41,
            "config_list": config_list,
        },
    )

    user_proxy = autogen.UserProxyAgent(name="user", human_input_mode="ALWAYS", code_execution_config=False)

    user_proxy.a_get_human_input = AsyncMock(return_value="Not funny. Try again.")

    res = await user_proxy.a_initiate_chat(
        assistant, clear_history=True, max_turns=3, message="Hello, make a non-offensive joke about AI."
    )
    print("Result summary:", res.summary)
    print("Human input:", res.human_input)
    print("chat history:", res.chat_history)
    assert len(res.chat_history) == 6, (
        f"Chat history should have 6 messages because max_turns is set to 3 (and user keep request try again) but has {len(res.chat_history)}."
    )


@pytest.mark.parametrize("credentials_from_test_param", credentials_all_llms, indirect=True)
@suppress_gemini_resource_exhausted
@pytest.mark.asyncio
async def test_async_max_turn(
    credentials_from_test_param: Credentials,
) -> None:
    await _test_async_max_turn(credentials_from_test_param)
