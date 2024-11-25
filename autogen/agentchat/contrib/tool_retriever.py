# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import importlib.util
import inspect
import os
from textwrap import dedent, indent

import pandas as pd
from sentence_transformers import SentenceTransformer, util

from autogen import AssistantAgent, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor


class ToolBuilder:
    TOOL_USING_PROMPT = """# Functions
    You have access to the following functions. They can be accessed from the module called 'functions' by their function names.
For example, if there is a function called `foo` you could import it by writing `from functions import foo`
{functions}
"""

    def __init__(self, corpus_path, retriever="all-mpnet-base-v2"):

        self.df = pd.read_csv(corpus_path, sep="\t")
        document_list = self.df["document_content"].tolist()

        self.model = SentenceTransformer(retriever)
        self.embeddings = self.model.encode(document_list)

    def retrieve(self, query, top_k=3):
        # Encode the query using the Sentence Transformer model
        query_embedding = self.model.encode([query])

        hits = util.semantic_search(query_embedding, self.embeddings, top_k=top_k)

        results = []
        for hit in hits[0]:
            results.append(self.df.iloc[hit["corpus_id"], 1])
        return results

    def bind(self, agent: AssistantAgent, functions: str):
        """Binds the function to the agent so that agent is aware of it."""
        sys_message = agent.system_message
        sys_message += self.TOOL_USING_PROMPT.format(functions=functions)
        agent.update_system_message(sys_message)
        return

    def bind_user_proxy(self, agent: UserProxyAgent, tool_root: str):
        """
        Updates user proxy agent with a executor so that code executor can successfully execute function-related code.
        Returns an updated user proxy.
        """
        # Find all the functions in the tool root
        functions = find_callables(tool_root)

        code_execution_config = agent._code_execution_config
        executor = LocalCommandLineCodeExecutor(
            timeout=code_execution_config.get("timeout", 180),
            work_dir=code_execution_config.get("work_dir", "coding"),
            functions=functions,
        )
        code_execution_config = {
            "executor": executor,
            "last_n_messages": code_execution_config.get("last_n_messages", 1),
        }
        updated_user_proxy = UserProxyAgent(
            name=agent.name,
            is_termination_msg=agent._is_termination_msg,
            code_execution_config=code_execution_config,
            human_input_mode="NEVER",
            default_auto_reply=agent._default_auto_reply,
        )
        return updated_user_proxy


def get_full_tool_description(py_file):
    """
    Retrieves the function signature for a given Python file.
    """
    with open(py_file, "r") as f:
        code = f.read()
        exec(code)
        function_name = os.path.splitext(os.path.basename(py_file))[0]
        if function_name in locals():
            func = locals()[function_name]
            content = f"def {func.__name__}{inspect.signature(func)}:\n"
            docstring = func.__doc__

            if docstring:
                docstring = dedent(docstring)
                docstring = '"""' + docstring + '"""'
                docstring = indent(docstring, "    ")
                content += docstring + "\n"
            return content
        else:
            raise ValueError(f"Function {function_name} not found in {py_file}")


def find_callables(directory):
    """
    Find all callable objects defined in Python files within the specified directory.
    """
    callables = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                module_name = os.path.splitext(file)[0]
                module_path = os.path.join(root, file)
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                for name, value in module.__dict__.items():
                    if callable(value) and name == module_name:
                        callables.append(value)
                        break
    return callables
