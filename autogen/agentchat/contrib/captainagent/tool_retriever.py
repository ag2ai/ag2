# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import contextlib
import functools
import importlib.util
import inspect
import io
import os
import traceback
from hashlib import md5
from pathlib import Path
from textwrap import dedent, indent
from typing import Optional, Union

from .... import AssistantAgent, UserProxyAgent
from ....coding import CodeExecutor, CodeExtractor, LocalCommandLineCodeExecutor, MarkdownCodeExtractor
from ....coding.base import CodeBlock, CodeResult
from ....doc_utils import export_module
from ....import_utils import optional_import_block, require_optional_import
from ....tools import Tool, get_function_schema, load_basemodels_if_needed

with optional_import_block():
    import pandas as pd
    from sentence_transformers import SentenceTransformer, util


@require_optional_import(["pandas", "sentence_transformers"], "retrievechat")
@export_module("autogen.agentchat.contrib.captainagent")
class ToolBuilder:
    TOOL_PROMPT_DEFAULT = """\n## Functions
You have access to the following functions. They can be accessed from the module called 'functions' by their function names.
For example, if there is a function called `foo` you could import it by writing `from functions import foo`
{functions}
"""
    TOOL_PROMPT_USER_DEFINED = """\n## Functions
You have access to the following functions. You can write python code to call these functions directly without importing them.
{functions}
"""

    def __init__(self, corpus_root, retriever="all-mpnet-base-v2", type="default"):
        if type == "default":
            corpus_path = os.path.join(corpus_root, "tool_description.tsv")
            self.df = pd.read_csv(corpus_path, sep="\t")
            document_list = self.df["document_content"].tolist()
            self.TOOL_PROMPT = self.TOOL_PROMPT_DEFAULT
        else:
            self.TOOL_PROMPT = self.TOOL_PROMPT_USER_DEFINED
            # user defined tools, retrieve is actually not needed, just for consistency
            document_list = []
            for tool in corpus_root:
                document_list.append(tool.description)

        self.model = SentenceTransformer(retriever)
        self.embeddings = self.model.encode(document_list)
        self.type = type

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
        sys_message += self.TOOL_PROMPT.format(functions=functions)
        agent.update_system_message(sys_message)
        return

    def bind_user_proxy(self, agent: UserProxyAgent, tool_root: Union[str, list]):
        """Updates user proxy agent with a executor so that code executor can successfully execute function-related code.
        Returns an updated user proxy.
        """
        if isinstance(tool_root, str):
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
        else:
            # second case: user defined tools
            code_execution_config = agent._code_execution_config
            executor = LocalExecutorWithTools(
                tools=tool_root,
                work_dir=code_execution_config.get("work_dir", "coding"),
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


class LocalExecutorWithTools(CodeExecutor):
    """An executor that executes code blocks with injected tools. In this executor, the func within the tools can be called directly without declaring in the code block.

    For example, for a tool converted from langchain, the relevant functions can be called directly.
    ```python
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
    from autogen.interop import Interoperability

    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=3000)
    langchain_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
    interop = Interoperability()
    ag2_tool = interop.convert_tool(tool=langchain_tool, type="langchain")

    # `ag2_tool.name` is wikipedia
    local_executor = LocalExecutorWithTools(tools=[ag2_tool], work_dir="./")

    code = '''
    result = wikipedia(tool_input={"query":"Christmas"})
    print(result)
    '''
    print(
        local_executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python", code=code),
            ]
        )
    )
    ```
    In this case, the `wikipedia` function can be called directly in the code block. This hides the complexity of the tool.

    Args:
        tools: The tools to inject into the code execution environment. Default is an empty list.
        work_dir: The working directory for the code execution. Default is the current directory.
    """

    @property
    def code_extractor(self) -> CodeExtractor:
        """(Experimental) Export a code extractor that can be used by an agent."""
        return MarkdownCodeExtractor()

    def __init__(self, tools: Optional[list[Tool]] = None, work_dir: Union[Path, str] = Path()):
        self.tools = tools if tools is not None else []
        self.work_dir = work_dir
        if not os.path.exists(work_dir):
            os.makedirs(work_dir, exist_ok=True)

    def execute_code_blocks(self, code_blocks: list[CodeBlock]) -> CodeResult:
        """Execute code blocks and return the result.

        Args:
            code_blocks (List[CodeBlock]): The code blocks to execute.

        Returns:
            CodeResult: The result of the code execution.
        """
        logs_all = ""
        exit_code = 0  # Success code
        code_file = None  # Path to the first saved codeblock content

        for idx, code_block in enumerate(code_blocks):
            code = code_block.code
            code_hash = md5(code.encode()).hexdigest()
            filename = f"tmp_code_{code_hash}.py"
            filepath = os.path.join(self.work_dir, filename)
            # Save code content to file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(code)

            if idx == 0:
                code_file = filepath

            # Create a new execution environment
            execution_env = {}
            # Inject the tools
            for tool in self.tools:
                execution_env[tool.name] = _wrap_function(tool.func)

            # Prepare to capture stdout and stderr
            stdout = io.StringIO()
            stderr = io.StringIO()

            # Execute the code block
            try:
                # Redirect stdout and stderr
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    # Exec the code in the execution environment
                    exec(code, execution_env)
            except Exception:
                # Capture exception traceback
                tb = traceback.format_exc()
                stderr.write(tb)
                exit_code = 1  # Non-zero exit code indicates failure

            # Collect outputs
            stdout_content = stdout.getvalue()
            stderr_content = stderr.getvalue()
            logs_all += stdout_content + stderr_content

        return CodeResult(exit_code=exit_code, output=logs_all, code_file=code_file)

    def restart(self):
        """Restart the code executor. Since this executor is stateless, no action is needed."""
        pass


@export_module("autogen.agentchat.contrib.captainagent")
def format_ag2_tool(tool: Tool):
    # get the args first
    schema = get_function_schema(tool.func, description=tool.description)

    arg_name = list(inspect.signature(tool.func).parameters.keys())[0]
    arg_info = schema["function"]["parameters"]["properties"][arg_name]["properties"]

    content = f'def {tool.name}({arg_name}):\n    """\n'
    content += indent(tool.description, "    ") + "\n"
    content += (
        indent(
            f"You must format all the arguments into a dictionary and pass them as **kwargs to {arg_name}. You should use print function to get the results.",
            "    ",
        )
        + "\n"
        + indent(f"For example:\n\tresult = {tool.name}({arg_name}={{'arg1': 'value1' }})", "    ")
        + "\n"
    )
    content += indent(f"Arguments passed in {arg_name}:\n", "    ")
    for arg, info in arg_info.items():
        content += indent(f"{arg} ({info['type']}): {info['description']}\n", "    " * 2)
    content += '    """\n'
    return content


def _wrap_function(func):
    """Wrap the function to dump the return value to json.

    Handles both sync and async functions.

    Args:
        func: the function to be wrapped.

    Returns:
        The wrapped function.
    """

    @load_basemodels_if_needed
    @functools.wraps(func)
    def _wrapped_func(*args, **kwargs):
        return func(*args, **kwargs)

    return _wrapped_func


@export_module("autogen.agentchat.contrib.captainagent")
def get_full_tool_description(py_file):
    """Retrieves the function signature for a given Python file."""
    with open(py_file) as f:
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


def _wrap_function(func):
    """Wrap the function to dump the return value to json.

    Handles both sync and async functions.

    Args:
        func: the function to be wrapped.

    Returns:
        The wrapped function.
    """

    @load_basemodels_if_needed
    @functools.wraps(func)
    def _wrapped_func(*args, **kwargs):
        return func(*args, **kwargs)

    return _wrapped_func


def find_callables(directory):
    """Find all callable objects defined in Python files within the specified directory."""
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
