# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2labs
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

# Load LLM inference endpoints from an env variable or a file
# See https://ag2labs.github.io/autogen/docs/FAQ#set-your-api-endpoints
# and OAI_CONFIG_LIST_sample
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
assistant = AssistantAgent("assistant", llm_config={"config_list": config_list})
user_proxy = UserProxyAgent(
    "user_proxy", code_execution_config={"work_dir": "coding", "use_docker": False}
)  # IMPORTANT: set to True to run code in docker, recommended
user_proxy.initiate_chat(assistant, message="Plot a chart of NVDA and TESLA stock price change YTD.")
