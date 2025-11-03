# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# !/usr/bin/env python3 -m pytest

import inspect
import json

import pytest

import autogen
from autogen import ConversableAgent
from autogen.import_utils import run_for_optional_imports
from autogen.math_utils import eval_math_responses
from autogen.oai.client import TOOL_ENABLED
from autogen.tools import tool
from test.credentials import Credentials


def test_eval_math_responses(credentials_gpt_4o_mini: Credentials):
    config_list = credentials_gpt_4o_mini.config_list
    tools = [
        {
            "type": "function",
            "function": {
                "name": "eval_math_responses",
                "description": "Select a response for a math problem using voting, and check if the response is correct if the solution is provided",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "responses": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "The responses in a list",
                        },
                        "solution": {
                            "type": "string",
                            "description": "The canonical solution",
                        },
                    },
                    "required": ["responses"],
                },
            },
        },
    ]
    client = autogen.OpenAIWrapper(config_list=config_list)
    response = client.create(
        messages=[
            {
                "role": "user",
                "content": 'evaluate the math responses ["1", "5/2", "5/2"] against the true answer \\frac{5}{2}',
            },
        ],
        tools=tools,
    )
    print(response)
    responses = client.extract_text_or_completion_object(response)
    print(responses[0])
    tool_calls = responses[0].tool_calls
    function_call = tool_calls[0].function
    name, arguments = function_call.name, json.loads(function_call.arguments)
    assert name == "eval_math_responses"
    print(arguments["responses"])
    # if isinstance(arguments["responses"], str):
    #     arguments["responses"] = json.loads(arguments["responses"])
    arguments["responses"] = [f"\\boxed{{{x}}}" for x in arguments["responses"]]
    print(arguments["responses"])
    arguments["solution"] = f"\\boxed{{{arguments['solution']}}}"
    print(eval_math_responses(**arguments))


def test_eval_math_responses_api_style_function(credentials_gpt_4o_mini: Credentials):
    config_list = credentials_gpt_4o_mini.config_list
    functions = [
        {
            "name": "eval_math_responses",
            "description": "Select a response for a math problem using voting, and check if the response is correct if the solution is provided",
            "parameters": {
                "type": "object",
                "properties": {
                    "responses": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The responses in a list",
                    },
                    "solution": {
                        "type": "string",
                        "description": "The canonical solution",
                    },
                },
                "required": ["responses"],
            },
        },
    ]
    client = autogen.OpenAIWrapper(config_list=config_list)
    response = client.create(
        messages=[
            {
                "role": "user",
                "content": 'evaluate the math responses ["1", "5/2", "5/2"] against the true answer \\frac{5}{2}',
            },
        ],
        functions=functions,
    )
    print(response)
    responses = client.extract_text_or_completion_object(response)
    print(responses[0])
    function_call = responses[0].function_call
    name, arguments = function_call.name, json.loads(function_call.arguments)
    assert name == "eval_math_responses"
    print(arguments["responses"])
    # if isinstance(arguments["responses"], str):
    #     arguments["responses"] = json.loads(arguments["responses"])
    arguments["responses"] = [f"\\boxed{{{x}}}" for x in arguments["responses"]]
    print(arguments["responses"])
    arguments["solution"] = f"\\boxed{{{arguments['solution']}}}"
    print(eval_math_responses(**arguments))


@run_for_optional_imports(["openai"], "openai")
def test_update_tool(credentials_gpt_4o: Credentials):
    llm_config = {
        "config_list": credentials_gpt_4o.config_list,
        "seed": 42,
        "tools": [],
    }

    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        is_termination_msg=lambda x: "TERMINATE" in x.get("content"),
    )
    assistant = autogen.AssistantAgent(name="test", llm_config=llm_config)

    # Define a new function *after* the assistant has been created
    assistant.update_tool_signature(
        {
            "type": "function",
            "function": {
                "name": "greet_user",
                "description": "Greets the user.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        is_remove=False,
    )
    res = user_proxy.initiate_chat(
        assistant,
        message="What functions do you know about in the context of this conversation? End your response with 'TERMINATE'.",
    )
    messages1 = assistant.chat_messages[user_proxy][-1]["content"]
    print("Summary:", res.summary)
    assert messages1.replace("TERMINATE", "") == res.summary, (
        "Message (removing TERMINATE) and summary should be the same"
    )

    assistant.update_tool_signature("greet_user", is_remove=True)
    res = user_proxy.initiate_chat(
        assistant,
        message="What functions do you know about in the context of this conversation? End your response with 'TERMINATE'.",
        summary_method="reflection_with_llm",
    )
    messages2 = assistant.chat_messages[user_proxy][-1]["content"]
    # The model should know about the function in the context of the conversation
    assert "greet_user" in messages1
    assert "greet_user" not in messages2
    print("Summary2:", res.summary)


@pytest.mark.skipif(not TOOL_ENABLED, reason="openai>=1.1.0 not installed")
def test_multi_tool_call():
    class FakeAgent(autogen.Agent):
        def __init__(self, name):
            self._name = name
            self.received = []
            self.silent = False

        @property
        def name(self):
            return self._name

        @property
        def description(self):
            return self._name

        def receive(
            self,
            message,
            sender,
            request_reply=None,
            silent=False,
        ):
            message = message if isinstance(message, list) else [message]
            self.received.extend(message)

    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        is_termination_msg=lambda x: "TERMINATE" in x.get("content"),
    )
    user_proxy.register_function({"echo": lambda str: str})

    fake_agent = FakeAgent("fake_agent")

    user_proxy.receive(
        message=[
            {
                "content": "test multi tool call",
                "tool_calls": [
                    {
                        "id": "tool_1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": json.JSONEncoder().encode({"str": "hello world"})},
                    },
                    {
                        "id": "tool_2",
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "arguments": json.JSONEncoder().encode({"str": "goodbye and thanks for all the fish"}),
                        },
                    },
                    {
                        "id": "tool_3",
                        "type": "function",
                        "function": {
                            "name": "multi_tool_call_echo",  # normalized "multi_tool_call.echo"
                            "arguments": json.JSONEncoder().encode({"str": "goodbye and thanks for all the fish"}),
                        },
                    },
                ],
            }
        ],
        sender=fake_agent,
        request_reply=True,
    )

    assert fake_agent.received == [
        {
            "role": "tool",
            "tool_responses": [
                {"tool_call_id": "tool_1", "role": "tool", "content": "hello world"},
                {
                    "tool_call_id": "tool_2",
                    "role": "tool",
                    "content": "goodbye and thanks for all the fish",
                },
                {
                    "tool_call_id": "tool_3",
                    "role": "tool",
                    "content": "Error: Function multi_tool_call_echo not found.",
                },
            ],
            "content": inspect.cleandoc(
                """
                hello world

                goodbye and thanks for all the fish

                Error: Function multi_tool_call_echo not found.
                """
            ),
        }
    ]


@pytest.mark.skipif(not TOOL_ENABLED, reason="openai>=1.1.0 not installed")
@pytest.mark.asyncio
async def test_async_multi_tool_call():
    class FakeAgent(autogen.Agent):
        def __init__(self, name):
            self._name = name
            self.received = []
            self.silent = False

        @property
        def name(self):
            return self._name

        @property
        def description(self):
            return self._name

        async def a_receive(
            self,
            message,
            sender,
            request_reply=None,
            silent=False,
        ):
            message = message if isinstance(message, list) else [message]
            self.received.extend(message)
            return ""

    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        is_termination_msg=lambda x: "TERMINATE" in x.get("content"),
    )

    def echo(str):
        return str

    async def a_echo(str):
        return str

    user_proxy.register_function({"a_echo": a_echo, "echo": echo})

    fake_agent = FakeAgent("fake_agent")

    await user_proxy.a_receive(
        message=[
            {
                "content": "test multi tool call",
                "tool_calls": [
                    {
                        "id": "tool_1",
                        "type": "function",
                        "function": {"name": "a_echo", "arguments": json.JSONEncoder().encode({"str": "hello world"})},
                    },
                    {
                        "id": "tool_2",
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "arguments": json.JSONEncoder().encode({"str": "goodbye and thanks for all the fish"}),
                        },
                    },
                    {
                        "id": "tool_3",
                        "type": "function",
                        "function": {
                            "name": "multi_tool_call_echo",  # normalized "multi_tool_call.echo"
                            "arguments": json.JSONEncoder().encode({"str": "goodbye and thanks for all the fish"}),
                        },
                    },
                ],
            }
        ],
        sender=fake_agent,
        request_reply=True,
    )

    assert fake_agent.received == [
        {
            "role": "tool",
            "tool_responses": [
                {"tool_call_id": "tool_1", "role": "tool", "content": "hello world"},
                {
                    "tool_call_id": "tool_2",
                    "role": "tool",
                    "content": "goodbye and thanks for all the fish",
                },
                {
                    "tool_call_id": "tool_3",
                    "role": "tool",
                    "content": "Error: Function multi_tool_call_echo not found.",
                },
            ],
            "content": inspect.cleandoc(
                """
                hello world

                goodbye and thanks for all the fish

                Error: Function multi_tool_call_echo not found.
                """
            ),
        }
    ]


@tool(name="calculator", description="Basic calculator")
def calculator(operation: str, a: float, b: float) -> float:
    ops = {"add": a + b, "multiply": a * b, "divide": a / b if b != 0 else 0}
    return ops.get(operation, 0)


@tool(name="weather_check", description="Check weather")
def weather_check(location: str) -> str:
    return f"Weather in {location}: Sunny, 25°C"


@pytest.mark.integration
@run_for_optional_imports("openai", "openai")
def test_handoffs_and_tools_integration(credentials_gpt_4o_mini: Credentials) -> None:
    """Integration test: handoffs with tools."""
    from autogen.agentchat import initiate_group_chat
    from autogen.agentchat.group.llm_condition import StringLLMCondition
    from autogen.agentchat.group.on_condition import OnCondition
    from autogen.agentchat.group.patterns import ManualPattern
    from autogen.agentchat.group.targets.transition_target import AgentTarget

    llm_config = credentials_gpt_4o_mini.llm_config

    triage_agent = ConversableAgent("triage_agent", llm_config=llm_config)
    math_agent = ConversableAgent("math_agent", llm_config=llm_config)
    weather_agent = ConversableAgent("weather_agent", llm_config=llm_config)
    general_agent = ConversableAgent("general_agent", llm_config=llm_config)
    user = ConversableAgent("user", llm_config=llm_config, human_input_mode="NEVER")

    # Setup handoffs
    triage_agent.handoffs.add_llm_conditions([
        OnCondition(
            condition=StringLLMCondition("contains math or calculate"),
            target=AgentTarget(math_agent),
            condition_llm_config=llm_config,
        ),
        OnCondition(
            condition=StringLLMCondition("contains weather"),
            target=AgentTarget(weather_agent),
            condition_llm_config=llm_config,
        ),
    ])

    triage_agent.handoffs.set_after_work(target=AgentTarget(general_agent))

    pattern = ManualPattern(
        initial_agent=triage_agent,
        agents=[triage_agent, math_agent, weather_agent, general_agent],
        user_agent=user,
        group_manager_args={"llm_config": llm_config},
    )

    messages = [{"role": "user", "content": "Calculate 25 * 4 for me"}]
    result, context, last_agent = initiate_group_chat(pattern=pattern, messages=messages, max_rounds=1)

    # Cleanup
    triage_agent.handoffs.clear()

    assert result is not None
    assert last_agent is not None


@pytest.mark.integration
@run_for_optional_imports("openai", "openai")
def test_mixed_tools_scenario_integration(credentials_gpt_4o_mini: Credentials) -> None:
    """Integration test: mixed tools in group chat."""
    from autogen.agentchat import initiate_group_chat
    from autogen.agentchat.group.patterns import RoundRobinPattern

    llm_config = credentials_gpt_4o_mini.llm_config

    triage_agent = ConversableAgent("triage_agent", llm_config=llm_config)
    math_agent = ConversableAgent("math_agent", llm_config=llm_config)
    weather_agent = ConversableAgent("weather_agent", llm_config=llm_config)
    user = ConversableAgent("user", llm_config=llm_config, human_input_mode="NEVER")

    pattern = RoundRobinPattern(
        initial_agent=triage_agent,
        agents=[triage_agent, math_agent, weather_agent],
        user_agent=user,
        group_manager_args={"llm_config": llm_config},
    )

    messages = [{"role": "user", "content": "Calculate 100/4 and check weather in London"}]
    result, context, last_agent = initiate_group_chat(pattern=pattern, messages=messages, max_rounds=1)

    assert result is not None
    assert last_agent is not None


if __name__ == "__main__":
    test_update_tool()
    # test_eval_math_responses()
    # test_multi_tool_call()
    # test_eval_math_responses_api_style_function()
