# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import os
import tempfile
import time

import pytest

import autogen
from autogen.agentchat import AssistantAgent, UserProxyAgent
from autogen.cache import Cache
from autogen.import_utils import run_for_optional_imports

from ..conftest import Credentials, suppress_gemini_resource_exhausted


@run_for_optional_imports("openai", "openai")
@run_for_optional_imports(["openai"], "openai")
def test_legacy_disk_cache(credentials_gpt_4o_mini: Credentials):
    random_cache_seed = int.from_bytes(os.urandom(2), "big")
    start_time = time.time()
    cold_cache_messages = run_conversation(
        credentials_gpt_4o_mini,
        cache_seed=random_cache_seed,
    )
    end_time = time.time()
    duration_with_cold_cache = end_time - start_time

    start_time = time.time()
    warm_cache_messages = run_conversation(
        credentials_gpt_4o_mini,
        cache_seed=random_cache_seed,
    )
    end_time = time.time()
    duration_with_warm_cache = end_time - start_time
    assert cold_cache_messages == warm_cache_messages
    assert duration_with_warm_cache < duration_with_cold_cache


def _test_redis_cache(credentials: Credentials):
    random_cache_seed = int.from_bytes(os.urandom(2), "big")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    start_time = time.time()
    with Cache.redis(random_cache_seed, redis_url) as cache_client:
        cold_cache_messages = run_conversation(credentials, cache_seed=None, cache=cache_client)
        end_time = time.time()
        duration_with_cold_cache = end_time - start_time

        start_time = time.time()
        warm_cache_messages = run_conversation(credentials, cache_seed=None, cache=cache_client)
        end_time = time.time()
        duration_with_warm_cache = end_time - start_time
        assert cold_cache_messages == warm_cache_messages
        assert duration_with_warm_cache < duration_with_cold_cache

    random_cache_seed = int.from_bytes(os.urandom(2), "big")
    with Cache.redis(random_cache_seed, redis_url) as cache_client:
        cold_cache_messages = run_groupchat_conversation(credentials, cache=cache_client)
        end_time = time.time()
        duration_with_cold_cache = end_time - start_time

        start_time = time.time()
        warm_cache_messages = run_groupchat_conversation(credentials, cache=cache_client)
        end_time = time.time()
        duration_with_warm_cache = end_time - start_time
        assert cold_cache_messages == warm_cache_messages
        assert duration_with_warm_cache < duration_with_cold_cache


@pytest.mark.skip(reason="Currently not working")
@run_for_optional_imports("openai", "openai")
@pytest.mark.redis
@run_for_optional_imports(["openai", "redis"], "redis")
def test_redis_cache(credentials_gpt_4o_mini: Credentials):
    _test_redis_cache(credentials_gpt_4o_mini)


@pytest.mark.skip(reason="Currently not working")
@pytest.mark.gemini
@suppress_gemini_resource_exhausted
@pytest.mark.redis
@run_for_optional_imports(["openai", "redis"], "redis")
def test_redis_cache_gemini(credentials_gemini_flash: Credentials):
    _test_redis_cache(credentials_gemini_flash)


@pytest.mark.skip(reason="Currently not working")
@pytest.mark.anthropic
@pytest.mark.redis
@run_for_optional_imports(["openai", "redis"], "redis")
def test_redis_cache_anthropic(credentials_anthropic_claude_sonnet: Credentials):
    _test_redis_cache(credentials_anthropic_claude_sonnet)


@run_for_optional_imports("openai", "openai")
@run_for_optional_imports(["openai"], "openai")
def test_disk_cache(credentials_gpt_4o_mini: Credentials):
    random_cache_seed = int.from_bytes(os.urandom(2), "big")
    start_time = time.time()
    with Cache.disk(random_cache_seed) as cache_client:
        cold_cache_messages = run_conversation(credentials_gpt_4o_mini, cache_seed=None, cache=cache_client)
        end_time = time.time()
        duration_with_cold_cache = end_time - start_time

        start_time = time.time()
        warm_cache_messages = run_conversation(credentials_gpt_4o_mini, cache_seed=None, cache=cache_client)
        end_time = time.time()
        duration_with_warm_cache = end_time - start_time
        assert cold_cache_messages == warm_cache_messages
        assert duration_with_warm_cache < duration_with_cold_cache

    random_cache_seed = int.from_bytes(os.urandom(2), "big")
    with Cache.disk(random_cache_seed) as cache_client:
        cold_cache_messages = run_groupchat_conversation(credentials_gpt_4o_mini, cache=cache_client)
        end_time = time.time()
        duration_with_cold_cache = end_time - start_time

        start_time = time.time()
        warm_cache_messages = run_groupchat_conversation(credentials_gpt_4o_mini, cache=cache_client)
        end_time = time.time()
        duration_with_warm_cache = end_time - start_time
        assert cold_cache_messages == warm_cache_messages
        assert duration_with_warm_cache < duration_with_cold_cache


def run_conversation(
    credentials: Credentials, cache_seed, human_input_mode="NEVER", max_consecutive_auto_reply=5, cache=None
):
    config_list = credentials.config_list
    llm_config = {
        "cache_seed": cache_seed,
        "config_list": config_list,
        "max_tokens": 1024,
    }
    with tempfile.TemporaryDirectory() as work_dir:
        assistant = AssistantAgent(
            "coding_agent",
            llm_config=llm_config,
        )
        user = UserProxyAgent(
            "user",
            human_input_mode=human_input_mode,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            code_execution_config={
                "work_dir": work_dir,
                "use_docker": "python:3",
                "timeout": 60,
            },
            llm_config=llm_config,
            system_message="""Is code provided but not enclosed in ``` blocks?
        If so, remind that code blocks need to be enclosed in ``` blocks.
        Reply TERMINATE to end the conversation if the task is finished. Don't say appreciation.
        If "Thank you" or "You\'re welcome" are said in the conversation, then say TERMINATE and that is your last message.""",
        )

        user.initiate_chat(assistant, message="TERMINATE", cache=cache)
        # should terminate without sending any message
        assert assistant.last_message()["content"] == assistant.last_message(user)["content"] == "TERMINATE"
        coding_task = "Print hello world to a file called hello.txt"

        # track how long this takes
        user.initiate_chat(assistant, message=coding_task, cache=cache)
        return user.chat_messages[assistant]


def run_groupchat_conversation(credentials: Credentials, cache, human_input_mode="NEVER", max_consecutive_auto_reply=5):
    config_list = credentials.config_list
    llm_config = {
        "cache_seed": None,
        "config_list": config_list,
        "max_tokens": 1024,
    }

    with tempfile.TemporaryDirectory() as work_dir:
        assistant = AssistantAgent(
            "coding_agent",
            llm_config=llm_config,
        )

        planner = AssistantAgent(
            "planner",
            llm_config=llm_config,
        )

        user = UserProxyAgent(
            "user",
            human_input_mode=human_input_mode,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            code_execution_config={
                "work_dir": work_dir,
                "use_docker": "python:3",
                "timeout": 60,
            },
            system_message="""Is code provided but not enclosed in ``` blocks?
        If so, remind that code blocks need to be enclosed in ``` blocks.
        Reply TERMINATE to end the conversation if the task is finished. Don't say appreciation.
        If "Thank you" or "You\'re welcome" are said in the conversation, then say TERMINATE and that is your last message.""",
        )

        group_chat = autogen.GroupChat(
            agents=[planner, assistant, user],
            messages=[],
            max_round=4,
            speaker_selection_method="round_robin",
        )
        manager = autogen.GroupChatManager(groupchat=group_chat, llm_config=llm_config)

        coding_task = "Print hello world to a file called hello.txt"

        user.initiate_chat(manager, message=coding_task, cache=cache)
        return user.chat_messages[list(user.chat_messages.keys())[-0]]
