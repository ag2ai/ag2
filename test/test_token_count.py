# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# !/usr/bin/env python3 -m pytest

import pytest

from autogen.import_utils import run_for_optional_imports
from autogen.token_count_utils import (
    _num_token_from_messages,
    count_token,
    get_max_token_limit,
    num_tokens_from_functions,
    percentile_used,
    token_left,
)

func1 = {
    "name": "sh",
    "description": "run a shell script and return the execution result.",
    "parameters": {
        "type": "object",
        "properties": {
            "script": {
                "type": "string",
                "description": "Valid shell script to execute.",
            }
        },
        "required": ["script"],
    },
}
func2 = {
    "name": "query_wolfram",
    "description": "Return the API query result from the Wolfram Alpha. the return is a tuple of (result, is_success).",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}
func3 = {
    "name": "python",
    "description": "run cell in ipython and return the execution result.",
    "parameters": {
        "type": "object",
        "properties": {
            "cell": {
                "type": "string",
                "description": "Valid Python cell to execute.",
            }
        },
        "required": ["cell"],
    },
}


@pytest.mark.parametrize(
    "input_functions, expected_count", [([func1], 44), ([func2], 46), ([func3], 45), ([func1, func2], 78)]
)
def test_num_tokens_from_functions(input_functions, expected_count):
    assert num_tokens_from_functions(input_functions) == expected_count


@pytest.mark.parametrize(
    "model, expected_count",
    [
        ("mistral-", 524),
        ("deepseek-chat", 524),
        ("claude", 524),
        ("gemini", 524),
    ],
)
def test_num_token_from_messages(model: str, expected_count: int) -> None:
    messages = [
        {
            "content": "You are a helpful AI assistant.\nSolve tasks using your coding and language skills.\nIn the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.\n    1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.\n    2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.\nSolve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.\nWhen using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.\nIf you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.\nIf the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.\nWhen you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.\nReply \"TERMINATE\" in the end when everything is done.\n    ",
            "role": "system",
        },
        {"content": "Hi", "role": "user", "name": "user_proxy"},
        {
            "content": "Hello! How can I assist you today? If you have a specific task or question, feel free to share it, and I'll do my best to help.",
            "role": "assistant",
            "name": "assistant",
        },
        {"content": "okkk", "role": "user", "name": "user_proxy"},
    ]
    assert _num_token_from_messages(messages=messages, model=model) == expected_count


@run_for_optional_imports("PIL", "unknown")
def test_num_tokens_from_gpt_image():
    # mock num_tokens_from_gpt_image function
    base64_encoded_image = (
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4"
        "//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="
    )

    messages = [
        {
            "role": "system",
            "content": "you are a helpful assistant. af3758 *3 33(3)",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello asdfjj qeweee"},
                {"type": "image_url", "image_url": {"url": base64_encoded_image}},
            ],
        },
    ]
    tokens = count_token(messages, model="gpt-4-vision-preview")

    # The total number of tokens is text + image
    # where text = 34, as shown in the previous test case
    # the image token is: 85 + 170 = 255
    assert tokens == 34 + 255

    # Test low quality
    messages = [
        {
            "role": "system",
            "content": "you are a helpful assistant. af3758 *3 33(3)",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello asdfjj qeweee"},
                {"type": "image_url", "image_url": {"url": base64_encoded_image, "detail": "low"}},
            ],
        },
    ]
    tokens = count_token(messages, model="gpt-4o")
    assert tokens == 34 + 85


def test_count_token():
    messages = [
        {
            "role": "system",
            "content": "you are a helpful assistant. af3758 *3 33(3)",
        },
        {
            "role": "user",
            "content": "hello asdfjj qeweee",
        },
    ]
    assert count_token(messages) == 34
    assert percentile_used(messages) == 34 / 4096
    assert token_left(messages) == 4096 - 34

    text = "I'm sorry, but I'm not able to"
    assert count_token(text) == 10
    assert token_left(text) == 4096 - 10
    assert percentile_used(text) == 10 / 4096


def test_model_aliases():
    assert get_max_token_limit("gpt35-turbo") == get_max_token_limit("gpt-3.5-turbo")
    assert get_max_token_limit("gpt-35-turbo") == get_max_token_limit("gpt-3.5-turbo")
    assert get_max_token_limit("gpt4") == get_max_token_limit("gpt-4")
    assert get_max_token_limit("gpt4-32k") == get_max_token_limit("gpt-4-32k")
    assert get_max_token_limit("gpt4o") == get_max_token_limit("gpt-4o")
    assert get_max_token_limit("gpt4o-mini") == get_max_token_limit("gpt-4o-mini")


if __name__ == "__main__":
    #    test_num_tokens_from_functions()
    #    test_count_token()
    test_model_aliases()
