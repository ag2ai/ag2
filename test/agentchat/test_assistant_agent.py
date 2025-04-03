# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# !/usr/bin/env python3 -m pytest

import os

import pytest

from autogen.agentchat import AssistantAgent, UserProxyAgent
from autogen.import_utils import run_for_optional_imports
from autogen.tools import tool

from ..conftest import Credentials, credentials_all_llms, suppress_gemini_resource_exhausted

here = os.path.abspath(os.path.dirname(__file__))


def _test_ai_user_proxy_agent(credentials: Credentials) -> None:
    conversations = {}
    # autogen.ChatCompletion.start_logging(conversations)

    config_list = credentials.config_list

    assistant = AssistantAgent(
        "assistant",
        system_message="You are a helpful assistant.",
        llm_config={
            "timeout": 600,
            "cache_seed": 42,
            "config_list": config_list,
        },
    )

    ai_user_proxy = UserProxyAgent(
        name="ai_user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=2,
        code_execution_config=False,
        llm_config={
            "config_list": config_list,
        },
        # In the system message the "user" always refers to the other agent.
        system_message="You ask a user for help. You check the answer from the user and provide feedback.",
    )
    assistant.reset()

    math_problem = "$x^3=125$. What is x?"
    res = ai_user_proxy.initiate_chat(
        assistant,
        message=math_problem,
    )
    print(conversations)
    print("Result summary:", res.summary)


@pytest.mark.parametrize("credentials_from_test_param", credentials_all_llms, indirect=True)
@suppress_gemini_resource_exhausted
def test_ai_user_proxy_agent(
    credentials_from_test_param: Credentials,
) -> None:
    _test_ai_user_proxy_agent(credentials_from_test_param)


@run_for_optional_imports("openai", "openai")
def test_gpt4omini(credentials_gpt_4o_mini: Credentials, human_input_mode="NEVER", max_consecutive_auto_reply=5):
    config_list = credentials_gpt_4o_mini.config_list
    llm_config = {
        "cache_seed": 42,
        "config_list": config_list,
        "max_tokens": 1024,
    }
    assistant = AssistantAgent(
        "coding_agent",
        llm_config=llm_config,
    )
    user = UserProxyAgent(
        "user",
        human_input_mode=human_input_mode,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        max_consecutive_auto_reply=max_consecutive_auto_reply,
        code_execution_config={
            "work_dir": f"{here}/test_agent_scripts",
            "use_docker": "python:3",
            "timeout": 60,
        },
        llm_config=llm_config,
        system_message="""Is code provided but not enclosed in ``` blocks?
If so, remind that code blocks need to be enclosed in ``` blocks.
Reply TERMINATE to end the conversation if the task is finished. Don't say appreciation.
If "Thank you" or "You\'re welcome" are said in the conversation, then say TERMINATE and that is your last message.""",
    )
    user.initiate_chat(assistant, message="TERMINATE")
    # should terminate without sending any message
    assert assistant.last_message()["content"] == assistant.last_message(user)["content"] == "TERMINATE"
    coding_task = "Print hello world to a file called hello.txt"
    user.initiate_chat(assistant, message=coding_task)
    # coding_task = "Create a powerpoint with the text hello world in it."
    # assistant.receive(coding_task, user)
    coding_task = "Save a pandas df with 3 rows and 3 columns to disk."
    user.initiate_chat(assistant, message=coding_task)
    assert not isinstance(user.use_docker, bool)  # None or str


@run_for_optional_imports("openai", "openai")
def test_create_execute_script(
    credentials_gpt_4o_mini: Credentials, human_input_mode="NEVER", max_consecutive_auto_reply=3
):
    config_list = credentials_gpt_4o_mini.config_list
    conversations = {}
    # autogen.ChatCompletion.start_logging(conversations)
    llm_config = {
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    }
    assistant = AssistantAgent(
        "assistant",
        llm_config=llm_config,
    )
    user = UserProxyAgent(
        "user",
        human_input_mode=human_input_mode,
        max_consecutive_auto_reply=max_consecutive_auto_reply,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    )
    user.initiate_chat(
        assistant,
        message="""Create and execute a script to plot a rocket without using matplotlib""",
    )
    assistant.reset()
    user = UserProxyAgent(
        "user",
        human_input_mode=human_input_mode,
        code_execution_config={"work_dir": f"{here}/test_agent_scripts"},
        max_consecutive_auto_reply=max_consecutive_auto_reply,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    )
    res = user.initiate_chat(
        assistant,
        message="""Create a temp.py file with the following content:
```
print('Hello world!')
```""",
    )
    print(conversations)
    print("Result summary:", res.summary)
    # autogen.ChatCompletion.print_usage_summary()
    # autogen.ChatCompletion.start_logging(compact=False)
    user.send("""Execute temp.py""", assistant)
    # print(autogen.ChatCompletion.logged_history)
    # autogen.ChatCompletion.print_usage_summary()
    # autogen.ChatCompletion.stop_logging()


@run_for_optional_imports("openai", "openai")
def test_tsp(credentials_gpt_4o_mini: Credentials, human_input_mode="NEVER", max_consecutive_auto_reply=2):
    config_list = credentials_gpt_4o_mini.config_list
    hard_questions = [
        "What if we must go from node 1 to node 2?",
        "Can we double all distances?",
        "Can we add a new point to the graph? It's distance should be randomly between 0 - 5 to each of the existing points.",
    ]

    def tsp_message(sender, recipient, context):
        filename = context.get("prompt_filename", "")
        with open(filename) as f:
            prompt = f.read()
        question = context.get("question", "")
        return prompt.format(question=question)

    # autogen.ChatCompletion.start_logging()
    assistant = AssistantAgent("assistant", llm_config={"temperature": 0, "config_list": config_list})
    user = UserProxyAgent(
        "user",
        code_execution_config={
            "work_dir": here,
        },
        human_input_mode=human_input_mode,
        max_consecutive_auto_reply=max_consecutive_auto_reply,
    )
    chat_res = user.initiate_chat(
        assistant, message=tsp_message, question=hard_questions[2], prompt_filename=f"{here}/tsp_prompt.txt"
    )
    # print(autogen.ChatCompletion.logged_history)
    # autogen.ChatCompletion.stop_logging()
    # print(chat_res.summary)
    print(chat_res.cost)


@run_for_optional_imports("openai", "openai")
def test_standalone(credentials_gpt_4o_mini: Credentials):
    config_list = credentials_gpt_4o_mini.config_list

    x_assistant = AssistantAgent(name="x_assistant", llm_config={"temperature": 0, "config_list": config_list})

    @tool()
    def get_twitter_hot_topic() -> str:
        return "Hot topic of the day on Twitter is #AI, and an influencer who is talking about it is @elonmusk"

    response = x_assistant.run(
        message="Find out today's hot topic and an influencer who is talking about it on X",
        tools=get_twitter_hot_topic,
        user_input=False,
    )

    response.process()

    assert "AI" in response.summary
    assert "elonmusk" in response.summary


@run_for_optional_imports("openai", "openai")
@pytest.mark.asyncio
async def test_standalone_async(credentials_gpt_4o_mini: Credentials):
    config_list = credentials_gpt_4o_mini.config_list

    x_assistant = AssistantAgent(name="x_assistant", llm_config={"temperature": 0, "config_list": config_list})

    @tool()
    def get_twitter_hot_topic() -> str:
        return "Hot topic of the day on Twitter is #AI, and an influencer who is talking about it is @elonmusk"

    response = await x_assistant.a_run(
        message="Find out today's hot topic and an influencer who is talking about it on X",
        tools=get_twitter_hot_topic,
        user_input=False,
    )

    await response.process()
    assert "AI" in await response.summary
    assert "elonmusk" in await response.summary


if __name__ == "__main__":
    # test_gpt35()
    # test_create_execute_script(human_input_mode="TERMINATE")
    # when GPT-4, i.e., the DEFAULT_MODEL, is used, conversation in the following test
    # should terminate in 2-3 rounds of interactions (because is_termination_msg should be true after 2-3 rounds)
    # although the max_consecutive_auto_reply is set to 10.
    test_tsp(human_input_mode="NEVER", max_consecutive_auto_reply=2)
    # test_ai_user_proxy_agent()
