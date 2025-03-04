# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import tempfile

import autogen
from autogen.agentchat.contrib.capabilities.transform_messages import TransformMessages
from autogen.agentchat.contrib.capabilities.transforms import MessageHistoryLimiter, MessageTokenLimiter
from autogen.import_utils import run_for_optional_imports

from ....conftest import Credentials


@run_for_optional_imports("openai", "openai")
def test_transform_messages_capability(credentials_gpt_4o_mini: Credentials) -> None:
    """Test the TransformMessages capability to handle long contexts.

    This test is a replica of test_transform_chat_history_with_agents in test_context_handling.py
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = credentials_gpt_4o_mini.llm_config
        assistant = autogen.AssistantAgent("assistant", llm_config=llm_config, max_consecutive_auto_reply=1)

        context_handling = TransformMessages(
            transforms=[
                MessageHistoryLimiter(max_messages=10),
                MessageTokenLimiter(max_tokens=10, max_tokens_per_message=5),
            ]
        )
        context_handling.add_to_agent(assistant)
        user = autogen.UserProxyAgent(
            "user",
            code_execution_config={"work_dir": temp_dir},
            human_input_mode="NEVER",
            is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
            max_consecutive_auto_reply=1,
        )

        # Create a very long chat history that is bound to cause a crash
        # for gpt 3.5
        for i in range(1000):
            assitant_msg = {"role": "assistant", "content": "test " * 1000}
            user_msg = {"role": "user", "content": ""}

            assistant.send(assitant_msg, user, request_reply=False)
            user.send(user_msg, assistant, request_reply=False)

        try:
            user.initiate_chat(
                assistant,
                message="Plot a chart of nvidia and tesla stock prices for the last 5 years",
                clear_history=False,
            )
        except Exception as e:
            assert False, f"Chat initiation failed with error {e!s}"


if __name__ == "__main__":
    test_transform_messages_capability()
