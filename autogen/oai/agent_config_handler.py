# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from autogen.llm_config import AgentConfig


def agent_config_parser(agent_config: "AgentConfig") -> dict[str, Any]:
    _agent_config: dict[str, Any] = {}
    if hasattr(agent_config, "response_format") and agent_config.response_format is not None:
        _agent_config["response_format"] = agent_config.response_format
    return _agent_config
