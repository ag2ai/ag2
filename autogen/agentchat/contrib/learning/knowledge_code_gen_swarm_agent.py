# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import base64
from typing import Any, Callable, List, Optional, Tuple, Union

import cloudpickle

from autogen.agentchat.contrib.learning.knowledge_function_swarm_agent import KnowledgeFunctionSwarmAgent
from autogen.agentchat.contrib.swarm_agent import (
    SwarmAgent,
)
from autogen.coding.base import CodeBlock, IPythonCodeResultOutputList
from autogen.coding.jupyter.docker_jupyter_server import DockerJupyterServer
from autogen.coding.jupyter.jupyter_code_executor import JupyterCodeExecutor

# TODO What I want to add is the ability to annotate a function and then have this code provide that function
# AND if that function errors, to use the params that error it as a test case to make it not error the next time, so its a self healing coding function.


class KnowledgeCodeGenSwarmAgent(KnowledgeFunctionSwarmAgent):
    def __init__(
        self,
        name: str,
        researcher_llm_config: dict,
        result_validator_llm_config: dict,
        docker_server: DockerJupyterServer,
        user_notes: List[str] = [],
        agent_system_message: str = "",
        max_rounds: int = 10,
        use_own_knowledge: bool = True,
        verbose: bool = True,
        knowledge_sources: Optional[List[SwarmAgent]] = None,
        **kwargs,
    ) -> None:
        self._docker_server = docker_server
        super().__init__(
            name=name,
            researcher_llm_config=researcher_llm_config,
            result_validator_llm_config=result_validator_llm_config,
            functions=[self.test_function_code],
            user_notes=user_notes,
            agent_system_message=agent_system_message,
            max_rounds=max_rounds,
            use_own_knowledge=use_own_knowledge,
            verbose=verbose,
            knowledge_sources=knowledge_sources,
            **kwargs,
        )

    def generate_function(
        self,
        question: str,
        function_name: str,
        # TODO Just did not figure out how to specify that its a type
        params: dict[str, Tuple[Any, Union[Any, Any]]],
        return_type: Any,
    ):
        incomplete_function = self._generate_incomplete_function(function_name, params, return_type)

        agent_text = f"""
You are responsible for generating and submitting python code following the guidelines of this skeleton:
{incomplete_function}

Do not wrap the code in anything, just submit valid python code as plain text.
"""

        self.set_agent_system_message(agent_text)
        self._input_pickle_block, self._invoke_and_print, self._pickle_function_and_output = self._generate_code_blocks(
            function_name, params
        )

        return self.auto_gen_func(question)

    def _generate_incomplete_function(
        self,
        function_name: str,
        # No idea how to do types of types (for now) because some are GenericAliases and stuff
        params: dict[str, Tuple[Any, Union[Any, Any]]],
        return_type: Any,
    ) -> str:
        param_str = ", ".join(f"{name}: {typ}" for name, (_, typ) in params.items())
        incomplete_def = f"def {function_name}({param_str}) -> {return_type}:\n    pass\n"

        return incomplete_def

    def _generate_code_blocks(
        self,
        function_name: str,
        params: dict[str, Tuple[Any, Union[Any, Any]]],
    ):
        # 1) Pickle the input data
        # We'll produce code that unpickles it in the kernel.
        pickled_data_map = {}
        for arg_name, (value, typ) in params.items():
            pickled_bytes = cloudpickle.dumps(value)
            b64_encoded = base64.b64encode(pickled_bytes).decode("utf-8")
            pickled_data_map[arg_name] = b64_encoded

        # We'll create Python code that reconstructs these arguments
        load_test_data_code = "import base64, cloudpickle\n"
        for arg_name, b64_str in pickled_data_map.items():
            load_test_data_code += f"""
{arg_name}_b64 = \"\"\"{b64_str}\"\"\"
{arg_name}_decoded = base64.b64decode({arg_name}_b64.encode('utf-8'))
{arg_name} = cloudpickle.loads({arg_name}_decoded)
"""
        input_pickle_block = CodeBlock(code=load_test_data_code, language="python")

        # 2) Invoke the function with the pickled parameters
        # We'll build a call signature like: `my_func(documents, names, ...)`
        # Then print the result
        call_args_str = ", ".join(params.keys())
        invoke_and_print_code = f"""
result = {function_name}({call_args_str})
print("Function returned:")
print(result)
"""
        invoke_and_print = CodeBlock(code=invoke_and_print_code, language="python")

        # 3) Pickle the function itself, returning the base64 string
        pickle_function_and_output = CodeBlock(
            code=f"""
import base64
import cloudpickle

pickled_fn = cloudpickle.dumps({function_name})
encoded_str = base64.b64encode(pickled_fn).decode('utf-8')
encoded_str  # This becomes the cell's "output"
""",
            language="python",
        )

        return (input_pickle_block, invoke_and_print, pickle_function_and_output)

    def test_function_code(self, context_variables: dict, function_code: str) -> Tuple[Callable, str]:
        """
        Use this to define a function which satisfies the users requirements.

        Args:
            table_summary (TableSummaryModel): The informational summary of the table.
        Returns:
            str: the summary
        """

        generated_code_block = CodeBlock(language="python", code=function_code)
        code_blocks = [
            self._input_pickle_block,
            generated_code_block,
            self._invoke_and_print,
            self._pickle_function_and_output,
        ]

        executor = JupyterCodeExecutor(self._docker_server)
        execution_result: IPythonCodeResultOutputList = executor.execute_code_blocks_output_list(
            code_blocks=code_blocks
        )

        if len(execution_result.outputs) < 4:
            # Should be the last one?
            error_cell_output = execution_result.outputs[-1]

            raise Exception(strip_ansi_codes(error_cell_output))

        # Output of last cell
        pickled_function_raw = execution_result.outputs[-1]
        function = unpickle_function(pickled_function_raw)

        # Should be second to last, so jank
        test_output = execution_result.outputs[-2]

        return function, f"Function Invocation Result:\n{test_output}"


def unpickle_function(base64_str: str):
    """
    Convert a base64-encoded string (pickled function) into a local Python callable.
    """
    decoded = base64.b64decode(base64_str.encode("utf-8"))
    func = cloudpickle.loads(decoded)
    return func


import re


def strip_ansi_codes(text):
    # Step 1: Decode the doubly escaped string
    decoded_output = text.encode("utf-8").decode("unicode_escape")

    # Step 2: Remove ANSI escape codes
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    clean_output = ansi_escape.sub("", decoded_output)

    return clean_output
