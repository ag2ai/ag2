# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

import os
import warnings

from autogen import ConversableAgent, LLMConfig

warnings.simplefilter("always", UserWarning)

llm_config = LLMConfig(
    api_type="openai",
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    stream=True,  # Enable streaming globally
)

agent = ConversableAgent("test", llm_config=llm_config)

response = agent.run(
    message="Write a couple sentences about something niche in ML evals.", max_turns=1, user_input=False
)

# Output will be streamed, also works for multi-agent conversations
# TODO: check why IOStream reads it twice (likely sending final message somewhere)
# TODO: can check with pytest, currently doing manually
response.process()
