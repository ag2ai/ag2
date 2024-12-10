from typing import List, Optional, Union, Literal, Callable, Dict
from ...conversable_agent import ConversableAgent, Agent

class ExecutorAgent(ConversableAgent):
    """An agent that executes code provided to it in markdown code blocks."""

    DEFAULT_DESCRIPTION = "A computer terminal that performs no other action than running Python scripts (provided to it quoted in ```python code blocks), or sh shell scripts (provided to it quoted in ```sh code blocks)"

    def __init__(
        self,
        name: str,
        confirm_execution: Union[Callable[[dict], bool], Literal["ACCEPT_ALL"]],
        description: Optional[str] = DEFAULT_DESCRIPTION,
        code_execution_config: Union[Dict, Literal[False]] = {
                "work_dir": "coding",
                "use_docker": False,
            },  # Enable code execution by default
        llm_config: Optional[Union[Dict, Literal[False]]] = False,  # Disable LLM by default
        **kwargs
    ):
        """Initialize an executor agent.

        Args:
            name (str): name of the agent
            confirm_execution (Callable or "ACCEPT_ALL"): function to confirm code execution or "ACCEPT_ALL" to execute all code
            description (str, optional): description of the agent
            code_execution_config (dict or bool, optional): config for code execution. Defaults to True.
            llm_config (dict or bool, optional): config for llm. Defaults to False to disable LLM.
            **kwargs: other arguments to pass to ConversableAgent
        """

        super().__init__(
            name=name,
            description=description,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            **kwargs
        )

        self._confirm_execution = confirm_execution

        # Modify the reply functions to support code execution confirmation
        self.register_reply([ConversableAgent, None], self.check_termination_and_human_reply)
        self.register_reply([ConversableAgent, None], self._generate_code_execution_reply)

    def _generate_code_execution_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Union[Dict, Literal[False]]] = None,
    ):
        """Generate a reply using code execution with confirmation."""
        if self._code_execution_config is False:
            return False, None

        if messages is None:
            messages = self._oai_messages[sender]

        # Use the existing code execution method from ConversableAgent
        final, reply = self.generate_code_execution_reply(messages, sender, config)

        # If code execution is successful, check if it should be confirmed
        if final:
            if self._confirm_execution == "ACCEPT_ALL":
                return final, reply
            elif callable(self._confirm_execution):
                # Call the confirmation function with the code execution result
                if self._confirm_execution({"reply": reply, "messages": messages, "sender": sender}):
                    return final, reply
                else:
                    return False, None

        return final, reply
