# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
import os
import re
import sys

import pytest

from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from autogen.agentchat.contrib.captainagent import CaptainAgent
from autogen.agentchat.contrib.tool_retriever import ToolBuilder

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from conftest import MOCK_OPEN_AI_API_KEY, reason, skip_openai  # noqa: E402

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from test_assistant_agent import KEY_LOC, OAI_CONFIG_LIST  # noqa: E402

try:
    import chromadb
    import huggingface_hub
except ImportError:
    skip = True
else:
    skip = False


@pytest.mark.skipif(
    skip_openai,
    reason=reason,
)
def test_captain_agent_from_scratch():
    config_list = config_list_from_json(OAI_CONFIG_LIST, file_location=KEY_LOC)
    llm_config = {
        "temperature": 0,
        "config_list": config_list,
    }
    nested_config = {
        "autobuild_init_config": {
            "config_file_or_env": os.path.join(KEY_LOC, OAI_CONFIG_LIST),
            "builder_model": "gpt-4o",
            "agent_model": "gpt-4o",
        },
        "autobuild_build_config": {
            "default_llm_config": {"temperature": 1, "top_p": 0.95, "max_tokens": 1500, "seed": 52},
            "code_execution_config": {"timeout": 300, "work_dir": "groupchat", "last_n_messages": 1},
            "coding": True,
        },
        "group_chat_config": {"max_round": 10},
        "group_chat_llm_config": llm_config.copy(),
    }
    captain_agent = CaptainAgent(
        name="captain_agent",
        llm_config=llm_config,
        code_execution_config={"use_docker": False, "work_dir": "groupchat"},
        nested_config=nested_config,
        agent_config_save_path=None,
    )
    captain_user_proxy = UserProxyAgent(name="user_proxy", human_input_mode="NEVER")
    task = (
        "Find a paper on arxiv by programming, and analyze its application in some domain. "
        "For example, find a recent paper about gpt-4 on arxiv "
        "and find its potential applications in software."
    )

    result = captain_user_proxy.initiate_chat(captain_agent, message=task, max_turns=4)
    print(result)


@pytest.mark.skipif(
    skip_openai or skip,
    reason=reason,
)
def test_captain_agent_with_library():

    config_list = config_list_from_json(OAI_CONFIG_LIST, file_location=KEY_LOC)
    llm_config = {
        "temperature": 0,
        "config_list": config_list,
    }
    nested_config = {
        "autobuild_init_config": {
            "config_file_or_env": os.path.join(KEY_LOC, OAI_CONFIG_LIST),
            "builder_model": "gpt-4o",
            "agent_model": "gpt-4o",
        },
        "autobuild_build_config": {
            "default_llm_config": {"temperature": 1, "top_p": 0.95, "max_tokens": 1500, "seed": 52},
            "code_execution_config": {"timeout": 300, "work_dir": "groupchat", "last_n_messages": 1},
            "coding": True,
        },
        "autobuild_tool_config": {
            "retriever": "all-mpnet-base-v2",
        },
        "group_chat_config": {"max_round": 10},
        "group_chat_llm_config": llm_config.copy(),
    }
    captain_agent = CaptainAgent(
        name="captain_agent",
        llm_config=llm_config,
        code_execution_config={"use_docker": False, "work_dir": "groupchat"},
        nested_config=nested_config,
        agent_lib="example_test_captainagent.json",
        tool_lib="default",
        agent_config_save_path=None,
    )
    captain_user_proxy = UserProxyAgent(name="captain_user_proxy", human_input_mode="NEVER")
    task = (
        "Find a paper on arxiv by programming, and analyze its application in some domain. "
        "For example, find a recent paper about gpt-4 on arxiv "
        "and find its potential applications in software."
    )

    result = captain_user_proxy.initiate_chat(captain_agent, message=task, max_turns=4)
    print(result)


if __name__ == "__main__":
    test_captain_agent_from_scratch()
    test_captain_agent_with_library()
