# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# !/usr/bin/env python3 -m pytest

import sys

import pytest

from autogen.agentchat.contrib.math_user_proxy_agent import (
    MathUserProxyAgent,
    _add_print_to_last_line,
    _remove_print,
)
from autogen.import_utils import run_for_optional_imports

from ..conftest import Credentials


@run_for_optional_imports("openai", "openai")
@run_for_optional_imports(["openai"], "openai")
def test_math_user_proxy_agent(
    credentials_gpt_4o_mini: Credentials,
):
    from autogen.agentchat.assistant_agent import AssistantAgent

    conversations = {}
    # autogen.ChatCompletion.start_logging(conversations)

    assistant = AssistantAgent(
        "assistant",
        system_message="You are a helpful assistant.",
        llm_config={
            "cache_seed": 42,
            "config_list": credentials_gpt_4o_mini.config_list,
        },
    )

    mathproxyagent = MathUserProxyAgent(name="MathChatAgent", human_input_mode="NEVER")
    assistant.reset()

    math_problem = "$x^3=125$. What is x?"
    res = mathproxyagent.initiate_chat(assistant, message=mathproxyagent.message_generator, problem=math_problem)
    print(conversations)
    print("Chat summary:", res.summary)
    print("Chat history:", res.chat_history)


def test_add_remove_print():
    # test add print
    code = "a = 4\nb = 5\na,b"
    assert _add_print_to_last_line(code) == "a = 4\nb = 5\nprint(a,b)"

    # test remove print
    code = """print("hello")\na = 4*5\nprint("world")"""
    assert _remove_print(code) == "a = 4*5"

    # test remove print. Only remove prints without indentation
    code = "if 4 > 5:\n\tprint('True')"
    assert _remove_print(code) == code


@pytest.mark.skipif(
    sys.platform in ["darwin", "win32"],
    reason="do not run on MacOS or windows",
)
def test_execute_one_python_code():
    mathproxyagent = MathUserProxyAgent(name="MathChatAgent", human_input_mode="NEVER")

    # no output found 1
    code = "x=3"
    assert mathproxyagent.execute_one_python_code(code)[0] == "No output found. Make sure you print the results."

    # no output found 2
    code = "if 4 > 5:\n\tprint('True')"

    assert mathproxyagent.execute_one_python_code(code)[0] == "No output found."

    # return error
    code = "2+'2'"
    assert "Error:" in mathproxyagent.execute_one_python_code(code)[0]

    # save previous status
    mathproxyagent.execute_one_python_code("x=3\ny=x*2")
    assert mathproxyagent.execute_one_python_code("print(y)")[0].strip() == "6"

    code = "print('*'*2001)"
    assert (
        mathproxyagent.execute_one_python_code(code)[0]
        == "Your requested query response is too long. You might have made a mistake. Please revise your reasoning and query."
    )


def test_execute_one_wolfram_query():
    mathproxyagent = MathUserProxyAgent(name="MathChatAgent", human_input_mode="NEVER")
    code = "2x=3"

    try:
        mathproxyagent.execute_one_wolfram_query(code)[0]
    except ValueError:
        print("Wolfram API key not found. Skip test.")


def test_generate_prompt():
    mathproxyagent = MathUserProxyAgent(name="MathChatAgent", human_input_mode="NEVER")

    assert "customized" in mathproxyagent.message_generator(
        mathproxyagent, None, {"problem": "2x=4", "prompt_type": "python", "customized_prompt": "customized"}
    )


if __name__ == "__main__":
    # test_add_remove_print()
    # test_execute_one_python_code()
    # test_generate_prompt()
    test_math_user_proxy_agent()
