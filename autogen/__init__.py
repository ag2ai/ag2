# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
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
    "Agent",
    "ConversableAgent",
    "AssistantAgent",
    "UserProxyAgent",
    "GroupChat",
    "GroupChatManager",
    "register_function",
    "initiate_chats",
    "gather_usage_summary",
    "ChatResult",
    "initiate_swarm_chat",
    "a_initiate_swarm_chat",
    "SwarmAgent",
    "SwarmResult",
    "ON_CONDITION",
    "AFTER_WORK",
    "AfterWorkOption",
    "UPDATE_SYSTEM_MESSAGE",
    "ReasoningAgent",
    "visualize_tree",
    "ThinkNode",
    "DEFAULT_MODEL",
    "FAST_MODEL",
    "__version__",
    "AgentNameConflict",
    "NoEligibleSpeaker",
    "SenderRequired",
    "InvalidCarryOverType",
    "UndefinedNextAgent",
    "OpenAIWrapper",
    "ModelClient",
    "Completion",
    "ChatCompletion",
    "get_config_list",
    "config_list_gpt4_gpt35",
    "config_list_openai_aoai",
    "config_list_from_models",
    "config_list_from_json",
    "config_list_from_dotenv",
    "filter_config",
    "Cache",
]
