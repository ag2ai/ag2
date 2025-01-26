# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import logging

from .agentchat import (
    AFTER_WORK,
    ON_CONDITION,
    UPDATE_SYSTEM_MESSAGE,
    AfterWorkOption,
    Agent,
    AssistantAgent,
    ChatResult,
    ConversableAgent,
    GroupChat,
    GroupChatManager,
    ReasoningAgent,
    SwarmAgent,
    SwarmResult,
    ThinkNode,
    UserProxyAgent,
    a_initiate_swarm_chat,
    gather_usage_summary,
    initiate_chats,
    initiate_swarm_chat,
    register_function,
    visualize_tree,
)
from .code_utils import DEFAULT_MODEL, FAST_MODEL
from .exception_utils import (
    AgentNameConflict,
    InvalidCarryOverType,
    NoEligibleSpeaker,
    SenderRequired,
    UndefinedNextAgent,
)
from .oai import (
    Cache,
    ChatCompletion,
    Completion,
    ModelClient,
    OpenAIWrapper,
    config_list_from_dotenv,
    config_list_from_json,
    config_list_from_models,
    config_list_gpt4_gpt35,
    config_list_openai_aoai,
    filter_config,
    get_config_list,
)
from .version import __version__

# Set the root logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


__all__ = [
    "AFTER_WORK",
    "DEFAULT_MODEL",
    "FAST_MODEL",
    "ON_CONDITION",
    "UPDATE_SYSTEM_MESSAGE",
    "AfterWorkOption",
    "Agent",
    "AgentNameConflict",
    "AssistantAgent",
    "Cache",
    "ChatCompletion",
    "ChatResult",
    "Completion",
    "ConversableAgent",
    "GroupChat",
    "GroupChatManager",
    "InvalidCarryOverType",
    "ModelClient",
    "NoEligibleSpeaker",
    "OpenAIWrapper",
    "ReasoningAgent",
    "SenderRequired",
    "SwarmAgent",
    "SwarmResult",
    "ThinkNode",
    "UndefinedNextAgent",
    "UserProxyAgent",
    "__version__",
    "a_initiate_swarm_chat",
    "config_list_from_dotenv",
    "config_list_from_json",
    "config_list_from_models",
    "config_list_gpt4_gpt35",
    "config_list_openai_aoai",
    "filter_config",
    "gather_usage_summary",
    "get_config_list",
    "initiate_chats",
    "initiate_swarm_chat",
    "register_function",
    "visualize_tree",
]
