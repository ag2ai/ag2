# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen import ConversableAgent
from autogen.agents.contrib import TimeReplyAgent

# from autogen.import_utils import skip_on_missing_imports
from ....conftest import Credentials


# @skip_on_missing_imports("discord", "commsagent-discord") # If packages are required, use this
class TestTimeReplyAgent:
    def test_init(self, mock_credentials: Credentials) -> None:
        """Tests the initialization of the TimeReplyAgent"""
        time_agent = TimeReplyAgent(
            name="time_agent",
            llm_config=mock_credentials.llm_config,
            date_time_format="%Y-%m-%d %H:%M:%S",
            output_prefix="The time is: ",
        )

        assert time_agent._date_time_format == "%Y-%m-%d %H:%M:%S"
        assert time_agent._output_prefix == "The time is: "
        assert time_agent.system_message == "You are a calendar agent that just returns the date and time."

    def test_output(self) -> None:
        """Tests the output of the TimeReplyAgent"""
        asking_agent = ConversableAgent(
            name="asking_agent",
        )

        time_agent = TimeReplyAgent(
            name="time_agent",
            date_time_format="%Y_%m_%d %H_%M_%S",
        )

        result = asking_agent.initiate_chat(
            recipient=time_agent,
            message="What is the current date and time?",
            max_turns=1,
        )

        # Confirm the output is right
        assert result.summary.startswith("Tick, tock, the current date/time is 20")

        # Confirm the output is prefixed correctly
        time_agent = TimeReplyAgent(
            name="time_agent",
            output_prefix="The time is: ",
        )

        result = asking_agent.initiate_chat(
            recipient=time_agent,
            message="What is the current date and time?",
            max_turns=1,
        )

        assert result.summary.startswith("The time is: 20")
