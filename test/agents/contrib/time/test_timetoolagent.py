# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.agents.contrib import TimeToolAgent

# from autogen.import_utils import skip_on_missing_imports
from ....conftest import Credentials


# @skip_on_missing_imports("discord", "commsagent-discord") # If packages are required, use this
class TestTimeToolAgent:
    def test_init(self, mock_credentials: Credentials) -> None:
        """Tests the initialization of the TimeToolAgent"""
        time_agent = TimeToolAgent(
            name="time_agent",
            llm_config=mock_credentials.llm_config,
            date_time_format="%Y-%m-%d %H:%M:%S",
        )

        assert time_agent._date_time_format == "%Y-%m-%d %H:%M:%S"

        expected_tools = [
            {
                "type": "function",
                "function": {
                    "description": "Get the current computer's date and time.",
                    "name": "date_time",
                    "parameters": {
                        "properties": {
                            "date_time_format": {
                                "description": "date/time Python format",
                                "default": time_agent._date_time_format,
                                "type": "string",
                            }
                        },
                        "required": [],
                        "type": "object",
                    },
                },
            },
        ]

        assert set(tool.name for tool in time_agent.tools) == {"date_time"}
        assert time_agent.llm_config["tools"] == expected_tools  # type: ignore[index]
        assert time_agent.system_message == (
            "You are a calendar agent that uses tools to return the date and time. "
            "When you reply, say 'Tick, tock, the current date/time is ' followed by the date and time in the exact format the tool provided."
        )

        time_agent = TimeToolAgent(
            name="time_agent",
            llm_config=mock_credentials.llm_config,
            system_message="Use your tool to get the time and output it as 'The time is: ' followed by the date/time.",
        )

        assert (
            time_agent.system_message
            == "Use your tool to get the time and output it as 'The time is: ' followed by the date/time."
        )
