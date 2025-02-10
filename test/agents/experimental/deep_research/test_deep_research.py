# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.agents.experimental.deep_research.deep_research_tool import DeepResearchTool

from ....conftest import Credentials


class TestDeepResearchTool:
    def test___init__(self, mock_credentials: Credentials) -> None:
        tool = DeepResearchTool(
            llm_config=mock_credentials.llm_config,
        )

        assert isinstance(tool, DeepResearchTool)
        assert tool.name == "delegate_research_task"
