# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ValidationError

import autogen
from autogen.import_utils import optional_import_block

with optional_import_block() as result:
    import openai  # noqa: F401
    from openai.types.chat.parsed_chat_completion import ChatCompletion, ChatCompletionMessage, Choice


from ..conftest import (
    Credentials,
    credentials_gemini_flash,
    credentials_gpt_4o_mini,
    suppress_gemini_resource_exhausted,
)

credentials_structured_output = [
    pytest.param(
        credentials_gpt_4o_mini.__name__,
        marks=[pytest.mark.openai, pytest.mark.aux_neg_flag],
    ),
    pytest.param(
        credentials_gemini_flash.__name__,
        marks=[pytest.mark.gemini, pytest.mark.aux_neg_flag],
    ),
]


class ResponseModel(BaseModel):
    question: str
    short_answer: str
    reasoning: str
    difficulty: float


@pytest.mark.parametrize(
    "credentials_from_test_param",
    credentials_structured_output,
    indirect=True,
)
@pytest.mark.parametrize(
    "response_format",
    [
        ResponseModel,
        ResponseModel.model_json_schema(),
    ],
)
@suppress_gemini_resource_exhausted
def test_structured_output(credentials_from_test_param, response_format):
    config_list = credentials_from_test_param.config_list

    for config in config_list:
        config["response_format"] = response_format

    llm_config = {"config_list": config_list, "cache_seed": 43}

    user_proxy = autogen.UserProxyAgent(
        name="User_proxy",
        system_message="A human admin.",
        human_input_mode="NEVER",
    )

    assistant = autogen.AssistantAgent(
        name="Assistant",
        llm_config=llm_config,
    )

    chat_result = user_proxy.initiate_chat(
        assistant,
        message="What is the air-speed velocity of an unladen swallow?",
        max_turns=1,
        summary_method="last_msg",
    )

    try:
        ResponseModel.model_validate_json(chat_result.chat_history[-1]["content"])
    except ValidationError as e:
        raise AssertionError(f"Agent did not return a structured report. Exception: {e}")


@pytest.mark.parametrize(
    "credentials_from_test_param",
    credentials_structured_output,
    indirect=True,
)
@pytest.mark.parametrize("response_format", [ResponseModel, ResponseModel.model_json_schema()])
@suppress_gemini_resource_exhausted
def test_structured_output_global(credentials_from_test_param, response_format):
    config_list = credentials_from_test_param.config_list

    llm_config = {
        "config_list": config_list,
        "cache_seed": 43,
        "response_format": response_format,
    }

    user_proxy = autogen.UserProxyAgent(
        name="User_proxy",
        system_message="A human admin.",
        human_input_mode="NEVER",
    )

    assistant = autogen.AssistantAgent(
        name="Assistant",
        llm_config=llm_config,
    )

    chat_result = user_proxy.initiate_chat(
        assistant,
        message="What is the air-speed velocity of an unladen swallow?",
        max_turns=1,
        summary_method="last_msg",
    )

    try:
        ResponseModel.model_validate_json(chat_result.chat_history[-1]["content"])
    except ValidationError as e:
        raise AssertionError(f"Agent did not return a structured report. Exception: {e}")


# Helper classes for testing
class Step(BaseModel):
    explanation: str
    output: str


class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str

    def format(self) -> str:
        steps_output = "\n".join(
            f"Step {i + 1}: {step.explanation}\n  Output: {step.output}" for i, step in enumerate(self.steps)
        )
        return f"{steps_output}\n\nFinal Answer: {self.final_answer}"


@pytest.fixture
def mock_assistant(mock_credentials: Credentials) -> autogen.AssistantAgent:
    """Set up a mocked AssistantAgent with a predefined response format."""
    config_list = mock_credentials.config_list

    for config in config_list:
        config["response_format"] = MathReasoning

    llm_config = {"config_list": config_list, "cache_seed": 43}

    assistant = autogen.AssistantAgent(
        name="Assistant",
        llm_config=llm_config,
    )

    oai_client_mock = MagicMock()
    oai_client_mock.chat.completions.create.return_value = ChatCompletion(
        id="some-id",
        created=1733302346,
        model="gpt-4o",
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content='{"steps":[{"explanation":"some explanation","output":"some output"}],"final_answer":"final answer"}',
                    role="assistant",
                ),
            )
        ],
    )
    assistant.client._clients[0]._oai_client = oai_client_mock

    return assistant


def test_structured_output_formatting(mock_assistant: autogen.AssistantAgent) -> None:
    """Test that the AssistantAgent correctly formats structured output."""
    user_proxy = autogen.UserProxyAgent(
        name="User_proxy",
        system_message="A human admin.",
        human_input_mode="NEVER",
    )

    chat_result = user_proxy.initiate_chat(
        mock_assistant,
        message="What is the square root of 4?",
        max_turns=1,
        summary_method="last_msg",
    )

    expected_output = "Step 1: some explanation\n  Output: some output\n\nFinal Answer: final answer"
    assert chat_result.chat_history[-1]["content"] == expected_output
