# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import os

from autogen import UserProxyAgent
from autogen.agentchat.contrib.captainagent.captainagent import CaptainAgent
from autogen.import_utils import optional_import_block, run_for_optional_imports

from ...conftest import KEY_LOC, OAI_CONFIG_LIST, Credentials

with optional_import_block() as result:
    import chromadb  # noqa: F401
    import huggingface_hub  # noqa: F401


@run_for_optional_imports("openai", "openai")
def test_captain_agent_from_scratch(credentials_all: Credentials):
    config_list = credentials_all.config_list
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


@run_for_optional_imports("openai", "openai")
@run_for_optional_imports(["chromadb", "huggingface_hub"], "autobuild")
def test_captain_agent_with_library(credentials_all: Credentials):
    config_list = credentials_all.config_list
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
