from typing import List, Optional, Union, Literal, Callable
from ...conversable_agent import ConversableAgent
from autogen_core import CodeExecutor

class ExecutorAgent(ConversableAgent):
    """An agent that executes code provided to it in markdown code blocks."""

    DEFAULT_DESCRIPTION = "A computer terminal that performs no other action than running Python scripts (provided to it quoted in ```python code blocks), or sh shell scripts (provided to it quoted in ```sh code blocks)"

    def __init__(
        self,
        name: str,
        executor: CodeExecutor,
        confirm_execution: Union[Callable[[dict], bool], Literal["ACCEPT_ALL"]],
        description: Optional[str] = DEFAULT_DESCRIPTION,
        code_execution_config: Optional[Union[dict, bool]] = True,  # Enable code execution by default
        llm_config: Optional[Union[dict, bool]] = False,  # Disable LLM by default
        check_last_n_message: int = 5,
        **kwargs
    ):
        """Initialize an executor agent.

        Args:
            name (str): name of the agent
            executor (CodeExecutor): code executor to use
            confirm_execution (Callable or "ACCEPT_ALL"): function to confirm code execution or "ACCEPT_ALL" to execute all code
            description (str, optional): description of the agent
            code_execution_config (dict or bool, optional): config for code execution. Defaults to True.
            llm_config (dict or bool, optional): config for llm. Defaults to False to disable LLM.
            check_last_n_message (int, optional): number of last messages to check for code blocks. Defaults to 5.
            **kwargs: other arguments to pass to ConversableAgent
        """
        if code_execution_config is None:
            code_execution_config = True  # Enable code execution by default

        super().__init__(
            name=name,
            description=description,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            **kwargs
        )

        self._executor = executor
        self._confirm_execution = confirm_execution
        self._check_last_n_message = check_last_n_message

        # Clear any existing reply functions and register our specific ones in order
        self._reply_func_list = []
        
        # Register reply functions in order:
        # 1. Check for termination/human input
        # 2. Try code execution
        self.register_reply([ConversableAgent, None], self.check_termination_and_human_reply)
        self.register_reply([ConversableAgent, None], self._generate_code_execution_reply_using_executor)

"""

                                                                                                                                                                                                   
from typing import Optional, Union, Literal, Callable                                                                                                                                             
from autogen.agentchat import ConversableAgent

def create_executor_agent(                                                                                                                                                                        
    name: str,                                                                                                                                                                                                                                                                                                                                                    
    confirm_execution: Union[Callable[[dict], bool], Literal["ACCEPT_ALL"]],                                                                                                                      
    check_last_n_message: int = 5,                                                                                                                                                                
    description: Optional[str] = "A computer terminal that performs no other action than running Python scripts (provided to it quoted in ```python code blocks), or sh shell scripts (provided t it quoted in ```sh code blocks)",                                                                                                                                                                 
    **kwargs                                                                                                                                                                                      
) -> ConversableAgent:                                                                                                                                                                            
    """Create an executor agent that only executes code blocks."""                                                                                                                                
                                                                                                                                                                                                
    agent = ConversableAgent(                                                                                                                                                                     
        name=name,                                                                                                                                                                                
        description=description,                                                                                                                                                                  
        code_execution_config={
                "work_dir": "coding",
                "use_docker": False,
            },  # Use provided executor                                                                                                                    
        llm_config=False,  # Disable LLM capabilities                                                                                                                                             
        **kwargs                                                                                                                                                                                  
    )                                                                                                                                                                                             
                                                                                                                                                                                                
    # Store additional configuration
    agent._confirm_execution = confirm_execution
    agent._check_last_n_message = check_last_n_message

    # Clear existing reply functions and register new ones in specific order
    agent._reply_func_list = []

    # Register reply functions in order:
    # 1. Check for termination/human input
    # 2. Try code execution
    agent.register_reply([ConversableAgent, None], agent.check_termination_and_human_reply)
    agent.register_reply([ConversableAgent, None], agent.generate_code_execution_reply)
                                                                                                                                                                                                
    return agent    
"""
