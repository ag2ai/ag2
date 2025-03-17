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
    AfterWork,
    AfterWorkOption,
    Agent,
    AssistantAgent,
    ChatResult,
    ContextExpression,
    ConversableAgent,
    GroupChat,
    GroupChatManager,
    OnCondition,
    OnContextCondition,
    SwarmAgent,
    SwarmResult,
    UpdateSystemMessage,
    UserProxyAgent,
    a_initiate_swarm_chat,
    gather_usage_summary,
    initiate_chats,
    initiate_swarm_chat,
    register_function,
    register_hand_off,
)
from .code_utils import DEFAULT_MODEL, FAST_MODEL
from .exception_utils import (
    AgentNameConflictError,
    InvalidCarryOverTypeError,
    NoEligibleSpeakerError,
    SenderRequiredError,
    UndefinedNextAgentError,
)
from .llm_config import LLMConfig
from .oai import (
    Cache,
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
    "AfterWork",
    "AfterWorkOption",
    "Agent",
    "AgentNameConflictError",
    "AssistantAgent",
    "Cache",
    "ChatResult",
    "ContextExpression",
    "ConversableAgent",
    "GroupChat",
    "GroupChatManager",
    "InvalidCarryOverTypeError",
    "LLMConfig",
    "ModelClient",
    "NoEligibleSpeakerError",
    "OnCondition",
    "OnContextCondition",
    "OpenAIWrapper",
    "SenderRequiredError",
    "SwarmAgent",
    "SwarmResult",
    "UndefinedNextAgentError",
    "UpdateSystemMessage",
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
    "register_hand_off",
]
