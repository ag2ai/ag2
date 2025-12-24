# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from autogen.agentchat.conversable_agent import ConversableAgent


def agent_config_parser(agent: ConversableAgent) -> dict[str, Any]:
    agent_config = []
    if agent.response_format is not None:
        agent_config.append({
            "response_format": agent.pop("response_format"),
        })
    return agent_config
