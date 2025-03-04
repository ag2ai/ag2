# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# !/usr/bin/env python3 -m pytest

import json
import os
import tempfile

import pytest

from autogen.agentchat.contrib.captainagent.agent_builder import AgentBuilder
from autogen.agentchat.contrib.text_analyzer_agent import TextAnalyzerAgent
from autogen.import_utils import optional_import_block, run_for_optional_imports

from ...conftest import KEY_LOC, OAI_CONFIG_LIST

with optional_import_block() as result:
    import chromadb  # noqa: F401
    import huggingface_hub  # noqa: F401


here = os.path.abspath(os.path.dirname(__file__))


def _config_check(config):
    # check config loading
    assert config.get("coding", None) is not None
    assert config.get("default_llm_config", None) is not None
    assert config.get("code_execution_config", None) is not None

    for agent_config in config["agent_configs"]:
        assert agent_config.get("name", None) is not None
        assert agent_config.get("model", None) is not None
        assert agent_config.get("description", None) is not None
        assert agent_config.get("system_message", None) is not None


@pytest.fixture
def builder() -> AgentBuilder:
    return AgentBuilder(
        config_file_or_env=OAI_CONFIG_LIST,
        config_file_location=KEY_LOC,
        builder_model_tags=["gpt-4o"],
        agent_model_tags=["gpt-4o"],
    )


@run_for_optional_imports("openai", "openai")
@run_for_optional_imports(["openai"], "openai")
def test_build(builder: AgentBuilder):
    building_task = (
        "Find a paper on arxiv by programming, and analyze its application in some domain. "
        "For example, find a recent paper about gpt-4 on arxiv "
        "and find its potential applications in software."
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        _, agent_config = builder.build(
            building_task=building_task,
            default_llm_config={"temperature": 0},
            code_execution_config={
                "last_n_messages": 2,
                "work_dir": f"{temp_dir}/test_agent_scripts",
                "timeout": 60,
                "use_docker": "python:3",
            },
        )
        _config_check(agent_config)

    # check number of agents
    assert len(agent_config["agent_configs"]) <= builder.max_agents


@run_for_optional_imports("openai", "openai")
@run_for_optional_imports(["chromadb", "huggingface_hub"], "autobuild")
@run_for_optional_imports(["openai"], "openai")
def test_build_from_library(builder: AgentBuilder):
    building_task = (
        "Find a paper on arxiv by programming, and analyze its application in some domain. "
        "For example, find a recent paper about gpt-4 on arxiv "
        "and find its potential applications in software."
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        _, agent_config = builder.build_from_library(
            building_task=building_task,
            library_path_or_json=f"{here}/example_agent_builder_library.json",
            default_llm_config={"temperature": 0},
            code_execution_config={
                "last_n_messages": 2,
                "work_dir": f"{temp_dir}/test_agent_scripts",
                "timeout": 60,
                "use_docker": "python:3",
            },
        )
        _config_check(agent_config)

    # check number of agents
    assert len(agent_config["agent_configs"]) <= builder.max_agents

    builder.clear_all_agents()

    # test embedding similarity selection
    with tempfile.TemporaryDirectory() as temp_dir:
        _, agent_config = builder.build_from_library(
            building_task=building_task,
            library_path_or_json=f"{here}/example_agent_builder_library.json",
            default_llm_config={"temperature": 0},
            embedding_model="all-mpnet-base-v2",
            code_execution_config={
                "last_n_messages": 2,
                "work_dir": f"{temp_dir}/test_agent_scripts",
                "timeout": 60,
                "use_docker": "python:3",
            },
        )
        _config_check(agent_config)

    # check number of agents
    assert len(agent_config["agent_configs"]) <= builder.max_agents


@pytest.mark.openai
def test_build_with_agent_configs(builder: AgentBuilder):
    conf = {
        "building_task": "Generate one TextAnalyzerAgent to analyze text",
        "agent_configs": [
            {
                "name": "TextAnalyzerAgent",
                "model": ["gpt-4o"],
                "description": "A helpful assistant to analyze text. Ask them to analyze any text, and they will provide you with the analysis.",
                "system_message": "",
                "agent_path": "autogen/agentchat/contrib/text_analyzer_agent/TextAnalyzerAgent",
            }
        ],
        "coding": True,
        "default_llm_config": {"temperature": 0},
        "code_execution_config": {"work_dir": ".", "use_docker": False, "timeout": 60, "last_n_messages": 2},
    }

    agents, _ = builder.build(**conf)

    is_agent_found = False
    for agent in agents:
        if isinstance(agent, TextAnalyzerAgent):
            is_agent_found = True
            break

    assert is_agent_found, "TextAnalyzerAgent not found in agents"


@pytest.mark.openai
def test_save(builder: AgentBuilder):
    building_task = (
        "Find a paper on arxiv by programming, and analyze its application in some domain. "
        "For example, find a recent paper about gpt-4 on arxiv "
        "and find its potential applications in software."
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        builder.build(
            building_task=building_task,
            default_llm_config={"temperature": 0},
            code_execution_config={
                "last_n_messages": 2,
                "work_dir": f"{temp_dir}/test_agent_scripts",
                "timeout": 60,
                "use_docker": "python:3",
            },
        )

        # creat
        saved_files = builder.save(f"{temp_dir}/example_save_agent_builder_config.json")

        # check config file path
        assert os.path.isfile(saved_files)

        saved_configs = json.load(open(saved_files))  # noqa: SIM115

        _config_check(saved_configs)


@run_for_optional_imports("openai", "openai")
def test_load(builder: AgentBuilder):
    with tempfile.TemporaryDirectory() as temp_dir:
        config_save_path = f"{here}/example_test_agent_builder_config.json"
        json.load(open(config_save_path))  # noqa: SIM115

        _, loaded_agent_configs = builder.load(
            config_save_path,
            code_execution_config={
                "last_n_messages": 2,
                "work_dir": f"{temp_dir}/test_agent_scripts",
                "timeout": 60,
                "use_docker": "python:3",
            },
        )
        print(loaded_agent_configs)

        _config_check(loaded_agent_configs)


@run_for_optional_imports("openai", "openai")
def test_clear_agent(builder: AgentBuilder):
    with tempfile.TemporaryDirectory() as temp_dir:
        config_save_path = f"{here}/example_test_agent_builder_config.json"
        builder.load(
            config_save_path,
            code_execution_config={
                "last_n_messages": 2,
                "work_dir": f"{temp_dir}/test_agent_scripts",
                "timeout": 60,
                "use_docker": "python:3",
            },
        )
        builder.clear_all_agents()

        # check if the agent cleared
        assert len(builder.agent_procs_assign) == 0


if __name__ == "__main__":
    test_build()
    test_build_from_library()
    test_save()
    test_load()
    test_clear_agent()
