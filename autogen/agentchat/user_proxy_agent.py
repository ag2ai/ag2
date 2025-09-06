# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from collections.abc import Callable
from typing import Any, Literal

from ..doc_utils import export_module
from ..llm_config import LLMConfig
from ..runtime_logging import log_new_agent, logging_enabled
from .conversable_agent import ConversableAgent


@export_module("autogen")
class UserProxyAgent(ConversableAgent):
    """(In preview) A proxy agent for the user, that can execute code and provide feedback to the other agents.\n
    \n
        UserProxyAgent is a subclass of ConversableAgent configured with `human_input_mode` to ALWAYS\n
        and `llm_config` to False. By default, the agent will prompt for human input every time a message is received.\n
        Code execution is enabled by default. LLM-based auto reply is disabled by default.\n
        To modify auto reply, register a method with [`register_reply`](../ConversableAgent#register-reply).\n
        To modify the way to get human input, override `get_human_input` method.\n
        To modify the way to execute code blocks, single code block, or function call, override `execute_code_blocks`,\n
        `run_code`, and `execute_function` methods respectively.\n
    """

    # Default UserProxyAgent.description values, based on human_input_mode
    DEFAULT_USER_PROXY_AGENT_DESCRIPTIONS = {
        "ALWAYS": "An attentive HUMAN user who can answer questions about the task, and can perform tasks such as running Python code or inputting command line commands at a Linux terminal and reporting back the execution results.",
        "TERMINATE": "A user that can run Python code or input command line commands at a Linux terminal and report back the execution results.",
        "NEVER": "A computer terminal that performs no other action than running Python scripts (provided to it quoted in ```python code blocks), or sh shell scripts (provided to it quoted in ```sh code blocks).",
    }

    def __init__(
        self,
        name: str,
        is_termination_msg: Callable[[dict[str, Any]], bool] | None = None,
        max_consecutive_auto_reply: int | None = None,
        human_input_mode: Literal["ALWAYS", "TERMINATE", "NEVER"] = "ALWAYS",
        function_map: dict[str, Callable[..., Any]] | None = None,
        code_execution_config: dict[str, Any] | Literal[False] = {},
        default_auto_reply: str | dict[str, Any] | None = "",
        llm_config: LLMConfig | dict[str, Any] | Literal[False] | None = False,
        system_message: str | list[str] | None = "",
        description: str | None = None,
        **kwargs: Any,
    ):
        """Initialize a UserProxyAgent.

        Args:
        name (str): name of the agent.\n
        is_termination_msg (function): a function that takes a message in the form of a dictionary\n
            and returns a boolean value indicating if this received message is a termination message.\n
            The dict can contain the following keys: "content", "role", "name", "function_call".\n
        max_consecutive_auto_reply (int): the maximum number of consecutive auto replies.\n
            default to None (no limit provided, class attribute MAX_CONSECUTIVE_AUTO_REPLY will be used as the limit in this case).\n
            The limit only plays a role when human_input_mode is not "ALWAYS".\n
        human_input_mode (str): whether to ask for human inputs every time a message is received.\n
            Possible values are "ALWAYS", "TERMINATE", "NEVER".\n
            (1) When "ALWAYS", the agent prompts for human input every time a message is received.\n
                Under this mode, the conversation stops when the human input is "exit",\n
                or when is_termination_msg is True and there is no human input.\n
            (2) When "TERMINATE", the agent only prompts for human input only when a termination message is received or\n
                the number of auto reply reaches the max_consecutive_auto_reply.\n
            (3) When "NEVER", the agent will never prompt for human input. Under this mode, the conversation stops\n
                when the number of auto reply reaches the max_consecutive_auto_reply or when is_termination_msg is True.\n
        function_map (dict[str, callable]): Mapping function names (passed to openai) to callable functions.\n
        code_execution_config (dict or False): config for the code execution.\n
            To disable code execution, set to False. Otherwise, set to a dictionary with the following keys:\n
            - work_dir (Optional, str): The working directory for the code execution.\n
                If None, a default working directory will be used.\n
                The default working directory is the "extensions" directory under\n
                "path_to_autogen".\n
            - use_docker (Optional, list, str or bool): The docker image to use for code execution.\n
                Default is True, which means the code will be executed in a docker container. A default list of images will be used.\n
                If a list or a str of image name(s) is provided, the code will be executed in a docker container\n
                with the first image successfully pulled.\n
                If False, the code will be executed in the current environment.\n
                We strongly recommend using docker for code execution.\n
            - timeout (Optional, int): The maximum execution time in seconds.\n
            - last_n_messages (Experimental, Optional, int): The number of messages to look back for code execution. Default to 1.\n
        default_auto_reply (str or dict or None): the default auto reply message when no code execution or llm based reply is generated.\n
        llm_config (LLMConfig or dict or False or None): llm inference configuration.\n
            Please refer to [OpenAIWrapper.create](https://docs.ag2.ai/latest/docs/api-reference/autogen/OpenAIWrapper/#autogen.OpenAIWrapper.create)\n
            for available options.\n
            Default to False, which disables llm-based auto reply.\n
            When set to None, will use self.DEFAULT_CONFIG, which defaults to False.\n
        system_message (str or List): system message for ChatCompletion inference.\n
            Only used when llm_config is not False. Use it to reprogram the agent.\n
        description (str): a short description of the agent. This description is used by other agents\n
            (e.g. the GroupChatManager) to decide when to call upon this agent. (Default: system_message)\n
        **kwargs (dict): Please refer to other kwargs in\n
            [ConversableAgent](https://docs.ag2.ai/latest/docs/api-reference/autogen/ConversableAgent).\n
        """
        super().__init__(
            name=name,
            system_message=system_message,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            function_map=function_map,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply,
            description=(
                description if description is not None else self.DEFAULT_USER_PROXY_AGENT_DESCRIPTIONS[human_input_mode]
            ),
            **kwargs,
        )

        if logging_enabled():
            log_new_agent(self, locals())
