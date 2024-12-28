# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
import sys

import pytest
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from autogen.agentchat.contrib.tool_retriever import LocalExecutorWithTools
from autogen.coding.base import CodeBlock
from autogen.interop import Interoperability


# skip if python version is not >= 3.9
@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="Only Python 3.9 and above are supported for LangchainInteroperability"
)
def test_execution():
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=3000)
    langchain_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
    interop = Interoperability()
    ag2_tool = interop.convert_tool(tool=langchain_tool, type="langchain")

    # `ag2_tool.name` is wikipedia
    local_executor = LocalExecutorWithTools(tools=[ag2_tool], work_dir="./")

    code = """
    result = wikipedia(tool_input={"query":"Christmas"})
    print(result)
    """
    result = local_executor.execute_code_blocks(
        code_blocks=[
            CodeBlock(language="python", code=code),
        ]
    )
    print(result)
