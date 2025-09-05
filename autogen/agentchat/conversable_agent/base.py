# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


import copy
import logging
import warnings
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Optional

from ...code_utils import check_can_use_docker_or_throw, decide_use_docker
from ...coding.base import CodeExecutor
from ...coding.factory import CodeExecutorFactory
from ...llm_config import LLMConfig
from ...oai.client import OpenAIWrapper
from ...runtime_logging import log_new_agent, logging_enabled
from ...tools import Tool
from ..agent import Agent, LLMAgent
from .types import UpdateSystemMessage

if TYPE_CHECKING:
    from ..group.context_variables import ContextVariables
    from ..group.guardrails import Guardrail
    from ..group.handoffs import Handoffs
    from .conversable_agent import ConversableAgent

logger = logging.getLogger(__name__)


class ConversableAgentBase(LLMAgent):
    """Base class containing initialization and core properties for ConversableAgent"""

    DEFAULT_CONFIG = False  # False or dict, the default config for llm inference
    MAX_CONSECUTIVE_AUTO_REPLY = 100  # maximum number of consecutive auto replies (subject to future change)
    DEFAULT_SUMMARY_PROMPT = "Summarize the takeaway from the conversation. Do not add any introductory phrases."
    DEFAULT_SUMMARY_METHOD = "last_msg"

    llm_config: dict[str, Any] | Literal[False]

    def __init__(
        self,
        name: str,
        system_message: str | list | None = "You are a helpful AI Assistant.",
        is_termination_msg: Callable[[dict[str, Any]], bool] | None = None,
        max_consecutive_auto_reply: int | None = None,
        human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = "TERMINATE",
        function_map: dict[str, Callable[..., Any]] | None = None,
        code_execution_config: dict[str, Any] | Literal[False] = False,
        llm_config: LLMConfig | dict[str, Any] | Literal[False] | None = None,
        default_auto_reply: str | dict[str, Any] = "",
        description: str | None = None,
        chat_messages: dict[Agent, list[dict[str, Any]]] | None = None,
        silent: bool | None = None,
        context_variables: Optional["ContextVariables"] = None,
        functions: list[Callable[..., Any]] | Callable[..., Any] = None,
        update_agent_state_before_reply: list[Callable | UpdateSystemMessage]
        | Callable
        | UpdateSystemMessage
        | None = None,
        handoffs: "Handoffs" | None = None,
    ):
        """Initialize base components of ConversableAgent"""
        from ..group.context_variables import ContextVariables
        from ..group.handoffs import Handoffs

        self.handoffs = handoffs if handoffs is not None else Handoffs()
        self.input_guardrails: list[Guardrail] = []
        self.output_guardrails: list[Guardrail] = []

        # we change code_execution_config below and we have to make sure we don't change the input
        # in case of UserProxyAgent, without this we could even change the default value {}
        code_execution_config = (
            code_execution_config.copy() if hasattr(code_execution_config, "copy") else code_execution_config
        )

        # a dictionary of conversations, default value is list
        if chat_messages is None:
            self._oai_messages = defaultdict(list)
        else:
            self._oai_messages = chat_messages

        self._oai_system_message = [{"content": system_message, "role": "system"}]
        self._description = description if description is not None else system_message
        self._is_termination_msg = (
            is_termination_msg
            if is_termination_msg is not None
            else (lambda x: self._content_str(x.get("content")) == "TERMINATE")
        )
        self.silent = silent
        self.run_executor: ConversableAgent | None = None

        # Take a copy to avoid modifying the given dict
        if isinstance(llm_config, dict):
            try:
                llm_config = copy.deepcopy(llm_config)
            except TypeError as e:
                raise TypeError(
                    "Please implement __deepcopy__ method for each value class in llm_config to support deepcopy."
                    " Refer to the docs for more details: https://docs.ag2.ai/docs/user-guide/advanced-concepts/llm-configuration-deep-dive/#adding-http-client-in-llm_config-for-proxy"
                ) from e

        self.llm_config = self._validate_llm_config(llm_config)
        self.client = self._create_client(self.llm_config)
        self._validate_name(name)
        self._name = name

        if logging_enabled():
            log_new_agent(self, locals())

        # Initialize standalone client cache object.
        self.client_cache = None

        # To track UI tools
        self._ui_tools: list[Tool] = []

        self.human_input_mode = human_input_mode
        self._max_consecutive_auto_reply = (
            max_consecutive_auto_reply if max_consecutive_auto_reply is not None else self.MAX_CONSECUTIVE_AUTO_REPLY
        )
        self._consecutive_auto_reply_counter = defaultdict(int)
        self._max_consecutive_auto_reply_dict = defaultdict(self.max_consecutive_auto_reply)
        self._function_map = (
            {}
            if function_map is None
            else {name: callable for name, callable in function_map.items() if self._assert_valid_name(name)}
        )
        self._default_auto_reply = default_auto_reply
        self._reply_func_list = []
        self._human_input = []
        self.reply_at_receive = defaultdict(bool)

        self.context_variables = context_variables if context_variables is not None else ContextVariables()

        self._tools: list[Tool] = []

        # Register functions to the agent
        if isinstance(functions, list):
            if not all(isinstance(func, Callable) for func in functions):
                raise TypeError("All elements in the functions list must be callable")
            self._add_functions(functions)
        elif isinstance(functions, Callable):
            self._add_single_function(functions)
        elif functions is not None:
            raise TypeError("Functions must be a callable or a list of callables")

        # Setting up code execution.
        self._setup_code_execution(code_execution_config)

        # Registered hooks are kept in lists, indexed by hookable method, to be called in their order of registration.
        # New hookable methods should be added to this list as required to support new agent capabilities.
        self.hook_lists: dict[str, list[Callable[..., Any]]] = {
            "process_last_received_message": [],
            "process_all_messages_before_reply": [],
            "process_message_before_send": [],
            "update_agent_state": [],
        }

        # Associate agent update state hooks
        self._register_update_agent_state_before_reply(update_agent_state_before_reply)

    def _setup_code_execution(self, code_execution_config):
        """Setup code execution configuration"""
        # Do not register code execution reply if code execution is disabled.
        if code_execution_config is not False:
            # If code_execution_config is None, set it to an empty dict.
            if code_execution_config is None:
                warnings.warn(
                    "Using None to signal a default code_execution_config is deprecated. "
                    "Use {} to use default or False to disable code execution.",
                    stacklevel=2,
                )
                code_execution_config = {}
            if not isinstance(code_execution_config, dict):
                raise ValueError("code_execution_config must be a dict or False.")

            # We have got a valid code_execution_config.
            self._code_execution_config: dict[str, Any] | Literal[False] = code_execution_config

            if self._code_execution_config.get("executor") is not None:
                if "use_docker" in self._code_execution_config:
                    raise ValueError(
                        "'use_docker' in code_execution_config is not valid when 'executor' is set. Use the appropriate arg in the chosen executor instead."
                    )

                if "work_dir" in self._code_execution_config:
                    raise ValueError(
                        "'work_dir' in code_execution_config is not valid when 'executor' is set. Use the appropriate arg in the chosen executor instead."
                    )

                if "timeout" in self._code_execution_config:
                    raise ValueError(
                        "'timeout' in code_execution_config is not valid when 'executor' is set. Use the appropriate arg in the chosen executor instead."
                    )

                # Use the new code executor.
                self._code_executor = CodeExecutorFactory.create(self._code_execution_config)
            else:
                # Legacy code execution using code_utils.
                use_docker = self._code_execution_config.get("use_docker", None)
                use_docker = decide_use_docker(use_docker)
                check_can_use_docker_or_throw(use_docker)
                self._code_execution_config["use_docker"] = use_docker
        else:
            # Code execution is disabled.
            self._code_execution_config = False

    def _content_str(self, content):
        """Helper to get content string"""
        from ...code_utils import content_str

        return content_str(content)

    def _validate_name(self, name: str) -> None:
        import re

        if not self.llm_config:
            return

        if any(entry for entry in self.llm_config.config_list if entry.api_type == "openai" and re.search(r"\s", name)):
            raise ValueError(f"The name of the agent cannot contain any whitespace. The name provided is: '{name}'")

    def _get_display_name(self):
        """Get the string representation of the agent.

        If you would like to change the standard string representation for an
        instance of ConversableAgent, you can point it to another function.
        In this example a function called _group_agent_str that returns a string:
        agent._get_display_name = MethodType(_group_agent_str, agent)
        """
        return self.name

    def __str__(self):
        return self._get_display_name()

    @classmethod
    def _validate_llm_config(
        cls, llm_config: LLMConfig | dict[str, Any] | Literal[False] | None
    ) -> LLMConfig | Literal[False]:
        if llm_config is None:
            llm_config = LLMConfig.get_current_llm_config()
            if llm_config is None:
                return cls.DEFAULT_CONFIG

        elif llm_config is False:
            return False

        return LLMConfig.ensure_config(llm_config)

    @classmethod
    def _create_client(cls, llm_config: LLMConfig | Literal[False]) -> OpenAIWrapper | None:
        return None if llm_config is False else OpenAIWrapper(**llm_config)

    @staticmethod
    def _is_silent(agent: Agent, silent: bool | None = False) -> bool:
        return agent.silent if agent.silent is not None else silent

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return self._name

    @property
    def description(self) -> str:
        """Get the description of the agent."""
        return self._description

    @description.setter
    def description(self, description: str):
        """Set the description of the agent."""
        self._description = description

    @property
    def code_executor(self) -> CodeExecutor | None:
        """The code executor used by this agent. Returns None if code execution is disabled."""
        if not hasattr(self, "_code_executor"):
            return None
        return self._code_executor

    @property
    def system_message(self) -> str:
        """Return the system message."""
        return self._oai_system_message[0]["content"]

    def update_system_message(self, system_message: str) -> None:
        """Update the system message.

        Args:
            system_message (str): system message for the ChatCompletion inference.
        """
        self._oai_system_message[0]["content"] = system_message

    def update_max_consecutive_auto_reply(self, value: int, sender: Agent | None = None):
        """Update the maximum number of consecutive auto replies.

        Args:
            value (int): the maximum number of consecutive auto replies.
            sender (Agent): when the sender is provided, only update the max_consecutive_auto_reply for that sender.
        """
        if sender is None:
            self._max_consecutive_auto_reply = value
            for k in self._max_consecutive_auto_reply_dict:
                self._max_consecutive_auto_reply_dict[k] = value
        else:
            self._max_consecutive_auto_reply_dict[sender] = value

    def max_consecutive_auto_reply(self, sender: Agent | None = None) -> int:
        """The maximum number of consecutive auto replies."""
        return self._max_consecutive_auto_reply if sender is None else self._max_consecutive_auto_reply_dict[sender]

    @property
    def chat_messages(self) -> dict[Agent, list[dict[str, Any]]]:
        """A dictionary of conversations from agent to list of messages."""
        return self._oai_messages

    def chat_messages_for_summary(self, agent: Agent) -> list[dict[str, Any]]:
        """A list of messages as a conversation to summarize."""
        return self._oai_messages[agent]

    def last_message(self, agent: Agent | None = None) -> dict[str, Any] | None:
        """The last message exchanged with the agent.

        Args:
            agent (Agent): The agent in the conversation.
                If None and more than one agent's conversations are found, an error will be raised.
                If None and only one conversation is found, the last message of the only conversation will be returned.

        Returns:
            The last message exchanged with the agent.
        """
        if agent is None:
            n_conversations = len(self._oai_messages)
            if n_conversations == 0:
                return None
            if n_conversations == 1:
                for conversation in self._oai_messages.values():
                    return conversation[-1]
            raise ValueError("More than one conversation is found. Please specify the sender to get the last message.")
        if agent not in self._oai_messages:
            raise KeyError(
                f"The agent '{agent.name}' is not present in any conversation. No history available for this agent."
            )
        return self._oai_messages[agent][-1]

    @property
    def use_docker(self) -> bool | str | None:
        """Bool value of whether to use docker to execute the code,
        or str value of the docker image name to use, or None when code execution is disabled.
        """
        return None if self._code_execution_config is False else self._code_execution_config.get("use_docker")

    @staticmethod
    def _assert_valid_name(name):
        """Ensure that configured names are valid, raises ValueError if not.

        For munging LLM responses use _normalize_name to ensure LLM specified names don't break the API.
        """
        import re

        if not re.match(r"^[a-zA-Z0-9_-]+$", name):
            raise ValueError(f"Invalid name: {name}. Only letters, numbers, '_' and '-' are allowed.")
        if len(name) > 64:
            raise ValueError(f"Invalid name: {name}. Name must be less than 64 characters.")
        return name

    @staticmethod
    def _normalize_name(name):
        """LLMs sometimes ask functions while ignoring their own format requirements, this function should be used to replace invalid characters with "_".

        Prefer _assert_valid_name for validating user configuration or input
        """
        import re

        return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:64]

    @property
    def tools(self) -> list[Tool]:
        """Get the agent's tools (registered for LLM)

        Note this is a copy of the tools list, use add_tool and remove_tool to modify the tools list.
        """
        return self._tools.copy()

    @property
    def function_map(self) -> dict[str, Callable[..., Any]]:
        """Return the function map."""
        return self._function_map
