# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

import base64
import os
import warnings
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from autogen.import_utils import optional_import_block, run_for_optional_imports
from autogen.llm_config import LLMConfig
from autogen.oai.gemini import GeminiClient, GeminiLLMConfigEntry

with optional_import_block() as result:
    from google.api_core.exceptions import InternalServerError
    from google.auth.credentials import Credentials
    from google.cloud.aiplatform.initializer import global_config as vertexai_global_config
    from google.genai.types import GenerateContentResponse, GoogleSearch, HttpOptions, Schema, Tool
    from vertexai.generative_models import GenerationResponse as VertexAIGenerationResponse
    from vertexai.generative_models import HarmBlockThreshold as VertexAIHarmBlockThreshold
    from vertexai.generative_models import HarmCategory as VertexAIHarmCategory
    from vertexai.generative_models import SafetySetting as VertexAISafetySetting


def test_gemini_llm_config_entry():
    gemini_llm_config = GeminiLLMConfigEntry(
        model="gemini-2.5-flash",
        api_key="dummy_api_key",
        project_id="fake-project-id",
        location="us-west1",
        proxy="http://mock-test-proxy:90/",
    )
    expected = {
        "api_type": "google",
        "model": "gemini-2.5-flash",
        "api_key": "dummy_api_key",
        "project_id": "fake-project-id",
        "location": "us-west1",
        "stream": False,
        "tags": [],
        "proxy": "http://mock-test-proxy:90/",
    }
    actual = gemini_llm_config.model_dump()
    assert actual == expected, actual

    assert LLMConfig(gemini_llm_config).model_dump() == {
        "config_list": [expected],
    }


@pytest.mark.parametrize("thinking_level", ["High", "Medium", "Low", "Minimal"])
def test_gemini_llm_config_entry_thinking_level(thinking_level):
    """Test that GeminiLLMConfigEntry accepts all valid thinking_level values"""
    gemini_llm_config = GeminiLLMConfigEntry(
        model="gemini-2.5-flash",
        api_key="dummy_api_key",
        thinking_level=thinking_level,
    )
    actual = gemini_llm_config.model_dump()
    assert actual["thinking_level"] == thinking_level


def test_gemini_llm_config_entry_thinking_config():
    """Test GeminiLLMConfigEntry with full thinking configuration"""
    gemini_llm_config = GeminiLLMConfigEntry(
        model="gemini-2.5-flash",
        api_key="dummy_api_key",
        include_thoughts=True,
        thinking_budget=1024,
        thinking_level="Medium",
    )
    actual = gemini_llm_config.model_dump()
    assert actual["include_thoughts"] is True
    assert actual["thinking_budget"] == 1024
    assert actual["thinking_level"] == "Medium"


@run_for_optional_imports(["vertexai", "PIL", "google.auth", "google.api", "google.cloud", "google.genai"], "gemini")
class TestGeminiClient:
    # Fixtures for mock data
    @pytest.fixture
    def mock_response(self):
        class MockResponse:
            def __init__(self, text, choices, usage, cost, model):
                self.text = text
                self.choices = choices
                self.usage = usage
                self.cost = cost
                self.model = model

        return MockResponse

    @pytest.fixture
    def gemini_client(self):
        system_message = [
            "You are a helpful AI assistant.",
        ]
        return GeminiClient(api_key="fake_api_key", system_message=system_message)

    @pytest.fixture
    def gemini_google_auth_default_client(self):
        system_message = [
            "You are a helpful AI assistant.",
        ]
        return GeminiClient(system_message=system_message)

    @pytest.fixture
    def gemini_client_with_credentials(self):
        mock_credentials = MagicMock(Credentials)
        return GeminiClient(credentials=mock_credentials)

    # Test compute location initialization and configuration
    def test_compute_location_initialization(self):
        with pytest.raises(AssertionError):
            GeminiClient(
                api_key="fake_api_key", location="us-west1"
            )  # Should raise an AssertionError due to specifying API key and compute location

    # Test project initialization and configuration
    def test_project_initialization(self):
        with pytest.raises(AssertionError):
            GeminiClient(
                api_key="fake_api_key", project_id="fake-project-id"
            )  # Should raise an AssertionError due to specifying API key and compute location

    def test_valid_initialization(self, gemini_client):
        assert gemini_client.api_key == "fake_api_key", "API Key should be correctly set"

    def test_google_application_credentials_initialization(self):
        GeminiClient(google_application_credentials="credentials.json", project_id="fake-project-id")
        assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == "credentials.json", (
            "Incorrect Google Application Credentials initialization"
        )

    def test_vertexai_initialization(self):
        mock_credentials = MagicMock(Credentials)
        GeminiClient(credentials=mock_credentials, project_id="fake-project-id", location="us-west1")
        assert vertexai_global_config.location == "us-west1", "Incorrect VertexAI location initialization"
        assert vertexai_global_config.project == "fake-project-id", "Incorrect VertexAI project initialization"
        assert vertexai_global_config.credentials == mock_credentials, "Incorrect VertexAI credentials initialization"

    def test_extract_system_instruction(self, gemini_client):
        # Test: valid system instruction
        messages = [{"role": "system", "content": "You are my personal assistant."}]
        assert gemini_client._extract_system_instruction(messages) == "You are my personal assistant."

        # Test: empty system instruction
        messages = [{"role": "system", "content": " "}]
        assert gemini_client._extract_system_instruction(messages) is None

        # Test: the first message is not a system instruction
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "system", "content": "You are my personal assistant."},
        ]
        assert gemini_client._extract_system_instruction(messages) is None

        # Test: empty message list
        assert gemini_client._extract_system_instruction([]) is None

        # Test: None input
        assert gemini_client._extract_system_instruction(None) is None

        # Test: system message without "content" key
        messages = [{"role": "system"}]
        with pytest.raises(KeyError):
            gemini_client._extract_system_instruction(messages)

    def test_gemini_message_handling(self, gemini_client):
        messages = [
            {"role": "system", "content": "You are my personal assistant."},
            {"role": "model", "content": "How can I help you?"},
            {"role": "user", "content": "Which planet is the nearest to the sun?"},
            {"role": "user", "content": "Which planet is the farthest from the sun?"},
            {"role": "model", "content": "Mercury is the closest planet to the sun."},
            {"role": "model", "content": "Neptune is the farthest planet from the sun."},
            {"role": "user", "content": "How can we determine the mass of a black hole?"},
        ]

        # The datastructure below defines what the structure of the messages
        # should resemble after converting to Gemini format.
        # Historically it has merged messages and ensured alternating roles,
        # this no longer appears to be required by the Gemini API
        expected_gemini_struct = [
            # system role is converted to user role
            {"role": "user", "parts": ["You are my personal assistant."]},
            {"role": "model", "parts": ["How can I help you?"]},
            {"role": "user", "parts": ["Which planet is the nearest to the sun?"]},
            {"role": "user", "parts": ["Which planet is the farthest from the sun?"]},
            {"role": "model", "parts": ["Mercury is the closest planet to the sun."]},
            {"role": "model", "parts": ["Neptune is the farthest planet from the sun."]},
            {"role": "user", "parts": ["How can we determine the mass of a black hole?"]},
        ]

        converted_messages = gemini_client._oai_messages_to_gemini_messages(messages)

        assert len(converted_messages) == len(expected_gemini_struct), "The number of messages is not as expected"

        for i, expected_msg in enumerate(expected_gemini_struct):
            assert expected_msg["role"] == converted_messages[i].role, "Incorrect mapped message role"
            for j, part in enumerate(expected_msg["parts"]):
                assert converted_messages[i].parts[j].text == part, "Incorrect mapped message text"

    def test_gemini_empty_message_handling(self, gemini_client):
        messages = [
            {"role": "system", "content": "You are my personal assistant."},
            {"role": "model", "content": "How can I help you?"},
            {"role": "user", "content": ""},
            {
                "role": "model",
                "content": "Please provide me with some context or a request! I need more information to assist you.",
            },
            {"role": "user", "content": ""},
        ]

        converted_messages = gemini_client._oai_messages_to_gemini_messages(messages)
        assert converted_messages[-3].parts[0].text == "empty", "Empty message is not converted to 'empty' correctly"
        assert converted_messages[-1].parts[0].text == "empty", "Empty message is not converted to 'empty' correctly"

    def test_gemini_message_without_role_defaults_to_user(self, gemini_client):
        """Test that messages without a 'role' field default to 'user' role (e.g., A2A messages)."""
        messages = [
            {"content": "Hello, this message has no role field"},
            {"role": "model", "content": "How can I help you?"},
            {"content": "Another message without role"},
        ]

        converted_messages = gemini_client._oai_messages_to_gemini_messages(messages)

        # Messages without role should be treated as "user"
        assert converted_messages[0].role == "user", "Message without role should default to 'user'"
        assert converted_messages[0].parts[0].text == "Hello, this message has no role field"
        assert converted_messages[1].role == "model"
        assert converted_messages[2].role == "user", "Message without role should default to 'user'"
        assert converted_messages[2].parts[0].text == "Another message without role"

    def test_parallel_function_responses_merged(self, gemini_client):
        """Test that consecutive tool response messages are merged into a single Content object.

        Gemini requires that the number of FunctionResponse parts equals the number of
        FunctionCall parts in the preceding model turn. When parallel function calls are
        made, AG2 unrolls the tool_responses into separate messages, but they must be
        recombined into a single Content for Gemini.
        """

        # Set up tool_call_function_map as would happen during a real conversation
        gemini_client.tool_call_function_map["1777"] = "timer"
        gemini_client.tool_call_function_map["1778"] = "stopwatch"

        messages = [
            {"role": "user", "content": "Create a timer for 1 second and then a stopwatch for 2 seconds."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1777",
                        "type": "function",
                        "function": {"name": "timer", "arguments": '{"num_seconds": "1"}'},
                    },
                    {
                        "id": "1778",
                        "type": "function",
                        "function": {"name": "stopwatch", "arguments": '{"num_seconds": "2"}'},
                    },
                ],
            },
            # These are the unrolled tool_responses — two separate messages
            {"role": "tool", "tool_call_id": "1777", "content": "Timer is done!"},
            {"role": "tool", "tool_call_id": "1778", "content": "Stopwatch is done!"},
        ]

        converted = gemini_client._oai_messages_to_gemini_messages(messages)

        # Should be: user message, model tool_call, single user with 2 FunctionResponse parts
        assert len(converted) == 3, f"Expected 3 Content objects, got {len(converted)}"

        # The third Content should have 2 FunctionResponse parts merged together
        tool_response_content = converted[2]
        assert tool_response_content.role == "user"
        assert len(tool_response_content.parts) == 2, (
            f"Expected 2 FunctionResponse parts in one Content, got {len(tool_response_content.parts)}"
        )
        assert tool_response_content.parts[0].function_response.name == "timer"
        assert tool_response_content.parts[1].function_response.name == "stopwatch"

    def test_single_function_response_not_affected(self, gemini_client):
        """Test that a single tool response still works correctly."""
        gemini_client.tool_call_function_map["123"] = "my_func"

        messages = [
            {"role": "user", "content": "Call my_func"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "123",
                        "type": "function",
                        "function": {"name": "my_func", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "123", "content": "done"},
        ]

        converted = gemini_client._oai_messages_to_gemini_messages(messages)

        assert len(converted) == 3
        tool_response_content = converted[2]
        assert tool_response_content.role == "user"
        assert len(tool_response_content.parts) == 1
        assert tool_response_content.parts[0].function_response.name == "my_func"

    def test_vertexai_safety_setting_conversion(self):
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ]
        converted_safety_settings = GeminiClient._to_vertexai_safety_settings(safety_settings)
        harm_categories = [
            VertexAIHarmCategory.HARM_CATEGORY_HARASSMENT,
            VertexAIHarmCategory.HARM_CATEGORY_HATE_SPEECH,
            VertexAIHarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            VertexAIHarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        ]
        expected_safety_settings = [
            VertexAISafetySetting(category=category, threshold=VertexAIHarmBlockThreshold.BLOCK_ONLY_HIGH)
            for category in harm_categories
        ]

        def compare_safety_settings(converted_safety_settings, expected_safety_settings):
            for i, expected_setting in enumerate(expected_safety_settings):
                converted_setting = converted_safety_settings[i]
                yield expected_setting.to_dict() == converted_setting.to_dict()

        assert len(converted_safety_settings) == len(expected_safety_settings), (
            "The length of the safety settings is incorrect"
        )
        settings_comparison = compare_safety_settings(converted_safety_settings, expected_safety_settings)
        assert all(settings_comparison), "Converted safety settings are incorrect"

    def test_vertexai_default_safety_settings_dict(self):
        safety_settings = {
            VertexAIHarmCategory.HARM_CATEGORY_HARASSMENT: VertexAIHarmBlockThreshold.BLOCK_ONLY_HIGH,
            VertexAIHarmCategory.HARM_CATEGORY_HATE_SPEECH: VertexAIHarmBlockThreshold.BLOCK_ONLY_HIGH,
            VertexAIHarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: VertexAIHarmBlockThreshold.BLOCK_ONLY_HIGH,
            VertexAIHarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: VertexAIHarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        converted_safety_settings = GeminiClient._to_vertexai_safety_settings(safety_settings)

        expected_safety_settings = dict.fromkeys(safety_settings, VertexAIHarmBlockThreshold.BLOCK_ONLY_HIGH)

        def compare_safety_settings(converted_safety_settings, expected_safety_settings):
            for expected_setting_key in expected_safety_settings:
                expected_setting = expected_safety_settings[expected_setting_key]
                converted_setting = converted_safety_settings[expected_setting_key]
                yield expected_setting == converted_setting

        assert len(converted_safety_settings) == len(expected_safety_settings), (
            "The length of the safety settings is incorrect"
        )
        settings_comparison = compare_safety_settings(converted_safety_settings, expected_safety_settings)
        assert all(settings_comparison), "Converted safety settings are incorrect"

    def test_vertexai_safety_setting_list(self):
        harm_categories = [
            VertexAIHarmCategory.HARM_CATEGORY_HARASSMENT,
            VertexAIHarmCategory.HARM_CATEGORY_HATE_SPEECH,
            VertexAIHarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            VertexAIHarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        ]

        expected_safety_settings = safety_settings = [
            VertexAISafetySetting(category=category, threshold=VertexAIHarmBlockThreshold.BLOCK_ONLY_HIGH)
            for category in harm_categories
        ]

        print(safety_settings)

        converted_safety_settings = GeminiClient._to_vertexai_safety_settings(safety_settings)

        def compare_safety_settings(converted_safety_settings, expected_safety_settings):
            for i, expected_setting in enumerate(expected_safety_settings):
                converted_setting = converted_safety_settings[i]
                yield expected_setting.to_dict() == converted_setting.to_dict()

        assert len(converted_safety_settings) == len(expected_safety_settings), (
            "The length of the safety settings is incorrect"
        )
        settings_comparison = compare_safety_settings(converted_safety_settings, expected_safety_settings)
        assert all(settings_comparison), "Converted safety settings are incorrect"

    # Test error handling
    @patch("autogen.oai.gemini.genai")
    def test_internal_server_error_retry(self, mock_genai, gemini_client):
        mock_genai.GenerativeModel.side_effect = [InternalServerError("Test Error"), None]  # First call fails
        # Mock successful response
        mock_chat = MagicMock()
        mock_chat.send_message.return_value = "Successful response"
        mock_genai.GenerativeModel.return_value.start_chat.return_value = mock_chat

        with patch.object(gemini_client, "create", return_value="Retried Successfully"):
            response = gemini_client.create({"model": "gemini-pro", "messages": [{"content": "Hello"}]})
            assert response == "Retried Successfully", "Should retry on InternalServerError"

    # Test cost calculation
    def test_cost_calculation(self, gemini_client, mock_response):
        response = mock_response(
            text="Example response",
            choices=[{"message": "Test message 1"}],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            cost=0.01,
            model="gemini-pro",
        )
        assert gemini_client.cost(response) > 0, "Cost should be correctly calculated as zero"

    @patch("autogen.oai.gemini.genai.Client")
    # @patch("autogen.oai.gemini.genai.configure")
    @patch("autogen.oai.gemini.calculate_gemini_cost")
    def test_create_response_with_text(self, mock_calculate_cost, mock_generative_client, gemini_client):
        mock_calculate_cost.return_value = 0.002

        mock_chat = MagicMock()
        mock_generative_client.return_value.chats.create.return_value = mock_chat
        assert mock_generative_client().chats.create() == mock_chat

        mock_text_part = MagicMock()
        mock_text_part.text = "Example response"
        mock_text_part.function_call = None

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 100
        mock_usage_metadata.candidates_token_count = 50

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_text_part]

        mock_response = MagicMock(spec=GenerateContentResponse)
        mock_response.usage_metadata = mock_usage_metadata
        mock_response.candidates = [mock_candidate]

        mock_chat.send_message.return_value = mock_response
        assert isinstance(mock_response, GenerateContentResponse)

        assert isinstance(mock_chat.send_message("dkdk"), GenerateContentResponse)

        response = gemini_client.create({
            "model": "gemini-pro",
            "messages": [{"content": "Hello", "role": "user"}],
            "stream": False,
        })

        # Assertions to check if response is structured as expected
        assert response.choices[0].message.content == "Example response", (
            "Response content should match expected output"
        )
        assert not response.choices[0].message.tool_calls, "There should be no tool calls"
        assert response.usage.prompt_tokens == 100, "Prompt tokens should match the mocked value"
        assert response.usage.completion_tokens == 50, "Completion tokens should match the mocked value"
        assert response.usage.total_tokens == 150, "Total tokens should be the sum of prompt and completion tokens"
        assert response.cost == 0.002, "Cost should match the mocked calculate_gemini_cost return value"

        # Verify that calculate_gemini_cost was called with the correct arguments
        mock_calculate_cost.assert_called_once_with(False, 100, 50, "gemini-pro")

    @patch("autogen.oai.gemini.GenerativeModel")
    @patch("autogen.oai.gemini.vertexai.init")
    @patch("autogen.oai.gemini.calculate_gemini_cost")
    def test_vertexai_create_response(
        self, mock_calculate_cost, mock_init, mock_generative_model, gemini_client_with_credentials
    ):
        # Mock the genai model configuration and creation process
        mock_chat = MagicMock()
        mock_model = MagicMock()
        mock_init.return_value = None
        mock_generative_model.return_value = mock_model
        mock_model.start_chat.return_value = mock_chat

        # Set up mock token counts with real integers
        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 100
        mock_usage_metadata.candidates_token_count = 50

        mock_text_part = MagicMock()
        mock_text_part.text = "Example response"
        mock_text_part.function_call = None

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_text_part]

        mock_response = MagicMock(spec=VertexAIGenerationResponse)
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = mock_usage_metadata
        mock_chat.send_message.return_value = mock_response

        # Mock the calculate_gemini_cost function
        mock_calculate_cost.return_value = 0.002

        # Call the create method
        response = gemini_client_with_credentials.create({
            "model": "gemini-pro",
            "messages": [{"content": "Hello", "role": "user"}],
            "stream": False,
        })

        # Assertions to check if response is structured as expected
        assert response.choices[0].message.content == "Example response", (
            "Response content should match expected output"
        )
        assert not response.choices[0].message.tool_calls, "There should be no tool calls"
        assert response.usage.prompt_tokens == 100, "Prompt tokens should match the mocked value"
        assert response.usage.completion_tokens == 50, "Completion tokens should match the mocked value"
        assert response.usage.total_tokens == 150, "Total tokens should be the sum of prompt and completion tokens"
        assert response.cost == 0.002, "Cost should match the mocked calculate_gemini_cost return value"

        # Verify that calculate_gemini_cost was called with the correct arguments
        mock_calculate_cost.assert_called_once_with(True, 100, 50, "gemini-pro")

    def test_extract_json_response(self, gemini_client):
        # Define test Pydantic model
        class Step(BaseModel):
            explanation: str
            output: str

        class MathReasoning(BaseModel):
            steps: list[Step]
            final_answer: str

        # Set up the response format
        gemini_client._response_format = MathReasoning

        # Test case 1: JSON within tags - CORRECT
        tagged_response = """{
                    "steps": [
                        {"explanation": "Step 1", "output": "8x = -30"},
                        {"explanation": "Step 2", "output": "x = -3.75"}
                    ],
                    "final_answer": "x = -3.75"
                }"""

        result = gemini_client._convert_json_response(tagged_response)
        assert isinstance(result, MathReasoning)
        assert len(result.steps) == 2
        assert result.final_answer == "x = -3.75"

        # Test case 2: Invalid JSON - RAISE ERROR
        invalid_response = """{
                    "steps": [
                        {"explanation": "Step 1", "output": "8x = -30"},
                        {"explanation": "Missing closing brace"
                    ],
                    "final_answer": "x = -3.75"
                """

        with pytest.raises(
            ValueError, match="Failed to parse response as valid JSON matching the schema for Structured Output: "
        ):
            gemini_client._convert_json_response(invalid_response)

        # Test case 3: No JSON content - RAISE ERROR
        no_json_response = "This response contains no JSON at all."

        with pytest.raises(
            ValueError,
            match="Failed to parse response as valid JSON matching the schema for Structured Output: Expecting value:",
        ):
            gemini_client._convert_json_response(no_json_response)

    @pytest.fixture
    def nested_function_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "$defs": {
                        "Subquestion": {
                            "properties": {
                                "question": {
                                    "description": "The original question.",
                                    "title": "Question",
                                    "type": "string",
                                }
                            },
                            "required": ["question"],
                            "title": "Subquestion",
                            "type": "object",
                        }
                    },
                    "properties": {
                        "question": {
                            "description": "The original question.",
                            "title": "Question",
                            "type": "string",
                        },
                        "subquestions": {
                            "description": "The subquestions that need to be answered.",
                            "items": {"$ref": "#/$defs/Subquestion"},
                            "title": "Subquestions",
                            "type": "array",
                        },
                    },
                    "required": ["question", "subquestions"],
                    "title": "Task",
                    "type": "object",
                    "description": "task",
                }
            },
            "required": ["task"],
        }

    def test_unwrap_references(self, nested_function_parameters: dict[str, Any]) -> None:
        result = GeminiClient._unwrap_references(nested_function_parameters)

        expected_result = {
            "type": "object",
            "properties": {
                "task": {
                    "properties": {
                        "question": {"description": "The original question.", "title": "Question", "type": "string"},
                        "subquestions": {
                            "description": "The subquestions that need to be answered.",
                            "items": {
                                "properties": {
                                    "question": {
                                        "description": "The original question.",
                                        "title": "Question",
                                        "type": "string",
                                    }
                                },
                                "required": ["question"],
                                "title": "Subquestion",
                                "type": "object",
                            },
                            "title": "Subquestions",
                            "type": "array",
                        },
                    },
                    "required": ["question", "subquestions"],
                    "title": "Task",
                    "type": "object",
                    "description": "task",
                }
            },
            "required": ["task"],
        }
        assert result == expected_result, result

    @patch("autogen.oai.gemini.genai.Client")
    @patch("autogen.oai.gemini.GenerateContentConfig")
    def test_generation_config_with_proxy(self, mock_generate_content_config, mock_generative_client, gemini_client):
        """Test that proxy parameter is properly set in Gemini LLM Config"""
        # Mock setup
        mock_chat = MagicMock()
        mock_generative_client.return_value.chats.create.return_value = mock_chat

        mock_text_part = MagicMock()
        mock_text_part.text = "Mock response"
        mock_text_part.function_call = None

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 10
        mock_usage_metadata.candidates_token_count = 5

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_text_part]

        mock_response = MagicMock(spec=GenerateContentResponse)
        mock_response.usage_metadata = mock_usage_metadata
        mock_response.candidates = [mock_candidate]

        mock_chat.send_message.return_value = mock_response

        # Call create with proxy parameter
        gemini_client.create({
            "model": "gemini-2.5-flash",
            "messages": [{"content": "Hello", "role": "user"}],
            "proxy": "http://mock-test-proxy:90/",
            "temperature": 0.7,
            "max_tokens": 265,
            "top_p": 0.5,
            "top_k": 3,
        })

        # Verify Client was called with correct parameters
        client_kwargs = mock_generative_client.call_args.kwargs
        assert "http_options" in client_kwargs
        http_options = client_kwargs["http_options"]
        assert isinstance(http_options, HttpOptions)
        assert http_options.client_args == {"proxy": "http://mock-test-proxy:90/"}
        assert http_options.async_client_args == {"proxy": "http://mock-test-proxy:90/"}

        # Verify GenerateContentConfig was called with correct parameters
        config_kwargs = mock_generate_content_config.call_args.kwargs
        expected_config = {
            "temperature": 0.7,
            "max_output_tokens": 265,
            "top_p": 0.5,
            "top_k": 3,
        }
        for key, expected_value in expected_config.items():
            assert key in config_kwargs, f"Expected key '{key}' not found in config kwargs"
            assert config_kwargs[key] == expected_value, f"Expected {key}={expected_value}, got {config_kwargs[key]}"

    def test_create_gemini_function_parameters_with_nested_parameters(
        self, nested_function_parameters: dict[str, Any]
    ) -> None:
        result = GeminiClient._create_gemini_function_parameters(nested_function_parameters)

        expected_result = {
            "type": "OBJECT",
            "properties": {
                "task": {
                    "properties": {
                        "question": {"description": "The original question.", "type": "STRING"},
                        "subquestions": {
                            "description": "The subquestions that need to be answered.",
                            "items": {
                                "properties": {"question": {"description": "The original question.", "type": "STRING"}},
                                "required": ["question"],
                                "type": "OBJECT",
                            },
                            "type": "ARRAY",
                        },
                    },
                    "required": ["question", "subquestions"],
                    "type": "OBJECT",
                    "description": "task",
                }
            },
            "required": ["task"],
        }

        assert result == expected_result, result

    def test_create_gemini_function_declaration_returns_schema(self) -> None:
        """Test that _create_gemini_function_declaration returns proper Schema objects without Pydantic warnings."""
        tool = {
            "function": {
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                        "max_results": {"type": "integer", "description": "Maximum number of results"},
                    },
                    "required": ["query"],
                },
            }
        }

        func_decl = GeminiClient._create_gemini_function_declaration(tool)

        # Verify the function declaration is correct
        assert func_decl.name == "search_web"
        assert func_decl.description == "Search the web for information"
        assert isinstance(func_decl.parameters, Schema), "parameters should be a Schema object, not a dict"

        # Verify no Pydantic serialization warnings are raised when serializing
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            func_decl.model_dump()

            pydantic_warnings = [
                warning
                for warning in w
                if "PydanticSerializationUnexpectedValue" in str(warning.message)
                and "parameters" in str(warning.message)
            ]
            assert len(pydantic_warnings) == 0, (
                f"Pydantic serialization warnings were raised for parameters field: {pydantic_warnings}"
            )

    def test_create_gemini_function_declaration_schema_handles_required_and_enum(self) -> None:
        """Test that _create_gemini_function_declaration_schema handles required and enum fields."""
        json_data = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "The status",
                    "enum": ["active", "inactive", "pending"],
                },
                "count": {"type": "integer", "description": "The count"},
            },
            "required": ["status"],
        }

        schema = GeminiClient._create_gemini_function_declaration_schema(json_data)

        assert isinstance(schema, Schema)
        assert schema.required == ["status"]
        assert schema.properties["status"].enum == ["active", "inactive", "pending"]

    def test_create_gemini_function_declaration_schema_with_nested_refs(self) -> None:
        """Test that _create_gemini_function_declaration_schema handles nested $ref references."""
        # This schema has a $ref inside a property that references a $defs entry
        json_data = {
            "type": "object",
            "$defs": {
                "SubItem": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The name"},
                    },
                    "required": ["name"],
                }
            },
            "properties": {
                "items": {
                    "type": "array",
                    "description": "List of items",
                    "items": {"$ref": "#/$defs/SubItem"},
                },
            },
            "required": ["items"],
        }

        # This should not raise KeyError: 'type'
        schema = GeminiClient._create_gemini_function_declaration_schema(json_data)

        assert isinstance(schema, Schema)
        assert schema.required == ["items"]
        assert schema.properties["items"].description == "List of items"
        # The nested $ref should be resolved and have proper type
        assert schema.properties["items"].items is not None

    @patch("autogen.oai.gemini.genai.Client")
    @patch("autogen.oai.gemini.GenerateContentConfig")
    def test_generation_config_with_seed(self, mock_generate_content_config, mock_generative_client, gemini_client):
        """Test that seed parameter is properly passed to generation config"""
        # Mock setup
        mock_chat = MagicMock()
        mock_generative_client.return_value.chats.create.return_value = mock_chat

        mock_text_part = MagicMock()
        mock_text_part.text = "Test response"
        mock_text_part.function_call = None

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 10
        mock_usage_metadata.candidates_token_count = 5

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_text_part]

        mock_response = MagicMock(spec=GenerateContentResponse)
        mock_response.usage_metadata = mock_usage_metadata
        mock_response.candidates = [mock_candidate]

        mock_chat.send_message.return_value = mock_response

        # Call create with seed parameter
        gemini_client.create({
            "model": "gemini-pro",
            "messages": [{"content": "Hello", "role": "user"}],
            "seed": 42,
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
            "top_k": 5,
        })

        # Verify GenerateContentConfig was called with correct parameters
        mock_generate_content_config.assert_called_once()
        call_kwargs = mock_generate_content_config.call_args.kwargs

        # Check that generation config parameters are correctly mapped
        assert call_kwargs["seed"] == 42, "Seed parameter should be passed to generation config"
        assert call_kwargs["temperature"] == 0.7, "Temperature parameter should be passed to generation config"
        assert call_kwargs["max_output_tokens"] == 100, "max_tokens should be mapped to max_output_tokens"
        assert call_kwargs["top_p"] == 0.9, "top_p parameter should be passed to generation config"
        assert call_kwargs["top_k"] == 5, "top_k parameter should be passed to generation config"

    @patch("autogen.oai.gemini.genai.Client")
    @patch("autogen.oai.gemini.GenerateContentConfig")
    @patch("autogen.oai.gemini.ThinkingConfig")
    def test_generation_config_with_thinking_config(
        self, mock_thinking_config, mock_generate_content_config, mock_generative_client, gemini_client
    ):
        """Test that thinking parameters are properly passed to generation config"""
        mock_chat = MagicMock()
        mock_generative_client.return_value.chats.create.return_value = mock_chat

        mock_text_part = MagicMock()
        mock_text_part.text = "Thoughtful response"
        mock_text_part.function_call = None

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 12
        mock_usage_metadata.candidates_token_count = 6

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_text_part]

        mock_response = MagicMock(spec=GenerateContentResponse)
        mock_response.usage_metadata = mock_usage_metadata
        mock_response.candidates = [mock_candidate]

        mock_chat.send_message.return_value = mock_response

        gemini_client.create({
            "model": "gemini-pro",
            "messages": [{"content": "Hello", "role": "user"}],
            "include_thoughts": True,
            "thinking_budget": 1024,
            "thinking_level": "High",
        })

        # Note: thinking_level is defined in GeminiLLMConfigEntry but not yet supported
        # by google.genai.types.ThinkingConfig, so it's not passed to the constructor
        mock_thinking_config.assert_called_once_with(
            include_thoughts=True,
            thinking_budget=1024,
        )

        config_kwargs = mock_generate_content_config.call_args.kwargs
        assert config_kwargs["thinking_config"] == mock_thinking_config.return_value, (
            "thinking_config should be passed to GenerateContentConfig"
        )

    @patch("autogen.oai.gemini.genai.Client")
    @patch("autogen.oai.gemini.GenerateContentConfig")
    @patch("autogen.oai.gemini.ThinkingConfig")
    def test_generation_config_with_default_thinking_config(
        self, mock_thinking_config, mock_generate_content_config, mock_generative_client, gemini_client
    ):
        """Test that a default ThinkingConfig is created and passed when no thinking params are provided"""
        mock_chat = MagicMock()
        mock_generative_client.return_value.chats.create.return_value = mock_chat

        mock_text_part = MagicMock()
        mock_text_part.text = "Response"
        mock_text_part.function_call = None

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 5
        mock_usage_metadata.candidates_token_count = 3

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_text_part]

        mock_response = MagicMock(spec=GenerateContentResponse)
        mock_response.usage_metadata = mock_usage_metadata
        mock_response.candidates = [mock_candidate]

        mock_chat.send_message.return_value = mock_response

        # Call create without thinking params
        gemini_client.create({
            "model": "gemini-pro",
            "messages": [{"content": "Hello", "role": "user"}],
        })

        mock_thinking_config.assert_called_once_with(
            include_thoughts=None,
            thinking_budget=None,
        )

        config_kwargs = mock_generate_content_config.call_args.kwargs
        assert config_kwargs["thinking_config"] == mock_thinking_config.return_value, (
            "default thinking_config should still be passed to GenerateContentConfig"
        )

    # Note: thinking_level is defined in GeminiLLMConfigEntry but not yet supported
    # by google.genai.types.ThinkingConfig, so it's not passed to the constructor.
    # These tests verify include_thoughts and thinking_budget are passed correctly.
    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            ({"include_thoughts": True}, {"include_thoughts": True, "thinking_budget": None}),
            ({"thinking_budget": 256}, {"include_thoughts": None, "thinking_budget": 256}),
            # thinking_level is accepted in config but not passed to ThinkingConfig
            ({"thinking_level": "High"}, {"include_thoughts": None, "thinking_budget": None}),
            (
                {"include_thoughts": False, "thinking_budget": 512},
                {"include_thoughts": False, "thinking_budget": 512},
            ),
            (
                {"include_thoughts": True, "thinking_level": "Low"},
                {"include_thoughts": True, "thinking_budget": None},
            ),
            (
                {"thinking_budget": 1024, "thinking_level": "High"},
                {"include_thoughts": None, "thinking_budget": 1024},
            ),
            (
                {"include_thoughts": True, "thinking_budget": 2048, "thinking_level": "High"},
                {"include_thoughts": True, "thinking_budget": 2048},
            ),
            # Test with "Medium" thinking level (not passed to ThinkingConfig)
            (
                {"thinking_level": "Medium"},
                {"include_thoughts": None, "thinking_budget": None},
            ),
            (
                {"include_thoughts": True, "thinking_level": "Medium"},
                {"include_thoughts": True, "thinking_budget": None},
            ),
            (
                {"thinking_budget": 512, "thinking_level": "Medium"},
                {"include_thoughts": None, "thinking_budget": 512},
            ),
            # Test with "Minimal" thinking level (not passed to ThinkingConfig)
            (
                {"thinking_level": "Minimal"},
                {"include_thoughts": None, "thinking_budget": None},
            ),
            (
                {"include_thoughts": True, "thinking_level": "Minimal"},
                {"include_thoughts": True, "thinking_budget": None},
            ),
            (
                {"thinking_budget": 128, "thinking_level": "Minimal"},
                {"include_thoughts": None, "thinking_budget": 128},
            ),
        ],
    )
    @patch("autogen.oai.gemini.genai.Client")
    @patch("autogen.oai.gemini.GenerateContentConfig")
    @patch("autogen.oai.gemini.ThinkingConfig")
    def test_generation_config_thinking_param_variants(
        self,
        mock_thinking_config,
        mock_generate_content_config,
        mock_generative_client,
        gemini_client,
        kwargs,
        expected,
    ):
        """Test individual and combined thinking params are passed through to ThinkingConfig and GenerateContentConfig"""
        mock_chat = MagicMock()
        mock_generative_client.return_value.chats.create.return_value = mock_chat

        mock_text_part = MagicMock()
        mock_text_part.text = "Response"
        mock_text_part.function_call = None

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 5
        mock_usage_metadata.candidates_token_count = 3

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_text_part]

        mock_response = MagicMock(spec=GenerateContentResponse)
        mock_response.usage_metadata = mock_usage_metadata
        mock_response.candidates = [mock_candidate]

        mock_chat.send_message.return_value = mock_response

        params = {
            "model": "gemini-pro",
            "messages": [{"content": "Hello", "role": "user"}],
            **kwargs,
        }
        gemini_client.create(params)

        mock_thinking_config.assert_called_once_with(
            include_thoughts=expected["include_thoughts"],
            thinking_budget=expected["thinking_budget"],
        )

        config_kwargs = mock_generate_content_config.call_args.kwargs
        assert config_kwargs["thinking_config"] == mock_thinking_config.return_value, (
            "thinking_config should be passed to GenerateContentConfig"
        )

    @patch("autogen.oai.gemini.GenerativeModel")
    @patch("autogen.oai.gemini.GenerationConfig")
    def test_vertexai_generation_config_with_seed(
        self, mock_generation_config, mock_generative_model, gemini_client_with_credentials
    ):
        """Test that seed parameter is properly passed to VertexAI generation config"""
        # Mock setup
        mock_chat = MagicMock()
        mock_model = MagicMock()
        mock_generative_model.return_value = mock_model
        mock_model.start_chat.return_value = mock_chat

        mock_text_part = MagicMock()
        mock_text_part.text = "Test response"
        mock_text_part.function_call = None

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 10
        mock_usage_metadata.candidates_token_count = 5

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_text_part]

        mock_response = MagicMock(spec=VertexAIGenerationResponse)
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = mock_usage_metadata
        mock_chat.send_message.return_value = mock_response

        # Call create with seed parameter
        gemini_client_with_credentials.create({
            "model": "gemini-pro",
            "messages": [{"content": "Hello", "role": "user"}],
            "seed": 123,
            "temperature": 0.5,
            "max_tokens": 200,
        })

        # Verify GenerationConfig was called with correct parameters
        mock_generation_config.assert_called_once()
        call_kwargs = mock_generation_config.call_args.kwargs

        # Check that generation config parameters are correctly mapped
        assert call_kwargs["seed"] == 123, "Seed parameter should be passed to VertexAI generation config"
        assert call_kwargs["temperature"] == 0.5, "Temperature parameter should be passed to VertexAI generation config"
        assert call_kwargs["max_output_tokens"] == 200, "max_tokens should be mapped to max_output_tokens"

    @pytest.mark.parametrize("name", ["prebuilt_google_search", "google_search"])
    def test_check_if_prebuilt_google_search_tool_exists(self, name: str) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "description": "Google Search",
                    "name": name,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query."},
                            "num_results": {
                                "type": "integer",
                                "default": 10,
                                "description": "The number of results to return.",
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ]
        expected = name == "prebuilt_google_search"
        assert GeminiClient._check_if_prebuilt_google_search_tool_exists(tools) == expected

    @pytest.mark.parametrize("name", ["prebuilt_google_search", "google_search"])
    def test_tools_to_gemini_tools(self, gemini_client: GeminiClient, name: str) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "description": "Google Search",
                    "name": name,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query."},
                            "num_results": {
                                "type": "integer",
                                "default": 10,
                                "description": "The number of results to return.",
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ]
        result = gemini_client._tools_to_gemini_tools(tools)
        assert isinstance(result, list)
        assert isinstance(result[0], Tool)

        tools_list = [Tool(google_search=GoogleSearch())]
        if name == "prebuilt_google_search":
            assert result == tools_list
        else:
            assert result != tools_list

    @patch("autogen.oai.gemini.genai.Client")
    @patch("autogen.oai.gemini.GenerateContentConfig")
    def test_response_format_uses_response_json_schema_for_non_vertexai(
        self, mock_generate_content_config, mock_generative_client, gemini_client
    ):
        """Test that non-VertexAI path uses response_json_schema instead of response_schema.

        This is the fix for issue #2348: Pydantic models with dict[str, SomeModel]
        generate additionalProperties in the JSON schema, which the Google GenAI SDK
        rejects via response_schema but accepts via response_json_schema.
        """
        mock_chat = MagicMock()
        mock_generative_client.return_value.chats.create.return_value = mock_chat

        mock_text_part = MagicMock()
        mock_text_part.text = '{"title": "Test", "extras": {"a": {"value": "hello", "score": 0.5}}}'
        mock_text_part.function_call = None

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 10
        mock_usage_metadata.candidates_token_count = 5

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_text_part]

        mock_response = MagicMock(spec=GenerateContentResponse)
        mock_response.usage_metadata = mock_usage_metadata
        mock_response.candidates = [mock_candidate]

        mock_chat.send_message.return_value = mock_response

        class Extra(BaseModel):
            value: str
            score: float

        class Report(BaseModel):
            title: str
            extras: dict[str, Extra]

        gemini_client.create({
            "model": "gemini-2.0-flash",
            "messages": [{"content": "Give me a report", "role": "user"}],
            "response_format": Report,
        })

        config_kwargs = mock_generate_content_config.call_args.kwargs
        assert "response_json_schema" in config_kwargs, "Non-VertexAI path should use response_json_schema"
        assert "response_schema" not in config_kwargs, "Non-VertexAI path should NOT use response_schema"
        assert config_kwargs["response_mime_type"] == "application/json"

    @patch("autogen.oai.gemini.GenerativeModel")
    @patch("autogen.oai.gemini.GenerationConfig")
    def test_response_format_uses_response_schema_for_vertexai(
        self, mock_generation_config, mock_generative_model, gemini_client_with_credentials
    ):
        """Test that VertexAI path still uses response_schema (not response_json_schema)."""
        mock_chat = MagicMock()
        mock_model = MagicMock()
        mock_generative_model.return_value = mock_model
        mock_model.start_chat.return_value = mock_chat

        mock_text_part = MagicMock()
        mock_text_part.text = '{"title": "Test", "extras": {"a": {"value": "hello", "score": 0.5}}}'
        mock_text_part.function_call = None

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 10
        mock_usage_metadata.candidates_token_count = 5

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_text_part]

        mock_response = MagicMock(spec=VertexAIGenerationResponse)
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = mock_usage_metadata
        mock_chat.send_message.return_value = mock_response

        class Extra(BaseModel):
            value: str
            score: float

        class Report(BaseModel):
            title: str
            extras: dict[str, Extra]

        gemini_client_with_credentials.create({
            "model": "gemini-2.0-flash",
            "messages": [{"content": "Give me a report", "role": "user"}],
            "response_format": Report,
        })

        config_kwargs = mock_generation_config.call_args.kwargs
        assert "response_schema" in config_kwargs, "VertexAI path should use response_schema"
        assert "response_json_schema" not in config_kwargs, "VertexAI path should NOT use response_json_schema"

    def test_thought_signature_initialized_in_init(self, gemini_client):
        """Test that thought signature mapping is initialized in __init__"""
        assert hasattr(gemini_client, "tool_call_thought_signatures")
        assert isinstance(gemini_client.tool_call_thought_signatures, dict)
        assert len(gemini_client.tool_call_thought_signatures) == 0

    @patch("autogen.oai.gemini.genai.Client")
    @patch("autogen.oai.gemini.calculate_gemini_cost")
    def test_thought_signature_captured_from_response(self, mock_calculate_cost, mock_generative_client, gemini_client):
        """Test that thought_signature is captured when parsing function call responses"""
        mock_calculate_cost.return_value = 0.001

        mock_chat = MagicMock()
        mock_generative_client.return_value.chats.create.return_value = mock_chat

        # Create a mock function call part with thought_signature
        mock_fn_call = MagicMock()
        mock_fn_call.name = "get_weather"
        mock_fn_call.args = {"location": "NYC"}

        mock_part = MagicMock()
        mock_part.function_call = mock_fn_call
        mock_part.text = ""
        mock_part.thought_signature = b"test_thought_signature_bytes"

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 10
        mock_usage_metadata.candidates_token_count = 5

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        mock_response = MagicMock(spec=GenerateContentResponse)
        mock_response.usage_metadata = mock_usage_metadata
        mock_response.candidates = [mock_candidate]

        mock_chat.send_message.return_value = mock_response

        response = gemini_client.create({
            "model": "gemini-3-flash",
            "messages": [{"content": "What's the weather in NYC?", "role": "user"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
                    },
                }
            ],
        })

        # Verify thought_signature was captured
        assert len(gemini_client.tool_call_thought_signatures) == 1
        tool_call_id = response.choices[0].message.tool_calls[0].id
        assert tool_call_id in gemini_client.tool_call_thought_signatures
        assert gemini_client.tool_call_thought_signatures[tool_call_id] == b"test_thought_signature_bytes"

    @patch("autogen.oai.gemini.genai.Client")
    @patch("autogen.oai.gemini.calculate_gemini_cost")
    def test_thought_signature_retained_across_calls(self, mock_calculate_cost, mock_generative_client, gemini_client):
        """Test that thought_signature is retained across multiple create() calls"""
        mock_calculate_cost.return_value = 0.001

        mock_chat = MagicMock()
        mock_generative_client.return_value.chats.create.return_value = mock_chat

        # First call: Model returns a function call with thought_signature
        mock_fn_call = MagicMock()
        mock_fn_call.name = "get_weather"
        mock_fn_call.args = {"location": "NYC"}

        mock_part_with_fn = MagicMock()
        mock_part_with_fn.function_call = mock_fn_call
        mock_part_with_fn.text = ""
        mock_part_with_fn.thought_signature = b"signature_for_tool_call"

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 10
        mock_usage_metadata.candidates_token_count = 5

        mock_candidate_fn = MagicMock()
        mock_candidate_fn.content.parts = [mock_part_with_fn]

        mock_response_fn = MagicMock(spec=GenerateContentResponse)
        mock_response_fn.usage_metadata = mock_usage_metadata
        mock_response_fn.candidates = [mock_candidate_fn]

        mock_chat.send_message.return_value = mock_response_fn

        # First create call
        response1 = gemini_client.create({
            "model": "gemini-3-flash",
            "messages": [{"content": "What's the weather?", "role": "user"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
                    },
                }
            ],
        })

        tool_call_id = response1.choices[0].message.tool_calls[0].id

        # Verify thought_signature was captured
        assert tool_call_id in gemini_client.tool_call_thought_signatures
        captured_signature = gemini_client.tool_call_thought_signatures[tool_call_id]
        assert captured_signature == b"signature_for_tool_call"

        # Second call: Send function result back (simulating the tool response flow)
        mock_part_text = MagicMock()
        mock_part_text.function_call = None
        mock_part_text.text = "The weather in NYC is sunny."

        mock_candidate_text = MagicMock()
        mock_candidate_text.content.parts = [mock_part_text]

        mock_response_text = MagicMock(spec=GenerateContentResponse)
        mock_response_text.usage_metadata = mock_usage_metadata
        mock_response_text.candidates = [mock_candidate_text]

        mock_chat.send_message.return_value = mock_response_text

        # Prepare messages that include the previous tool call and result
        messages = [
            {"content": "What's the weather?", "role": "user"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": tool_call_id, "content": "Sunny, 72°F"},
        ]

        # Second create call - thought_signature should still be available
        gemini_client.create({
            "model": "gemini-3-flash",
            "messages": messages,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
                    },
                }
            ],
        })

        # Verify the thought_signature is still retained after second call
        assert tool_call_id in gemini_client.tool_call_thought_signatures
        assert gemini_client.tool_call_thought_signatures[tool_call_id] == b"signature_for_tool_call"

    def test_thought_signature_included_in_reconstructed_parts(self, gemini_client):
        """Test that thought_signature is included when reconstructing function call Parts"""
        from google.genai.types import Part

        # Manually set up a thought_signature mapping
        tool_call_id = "test_tool_123"
        test_signature = b"test_signature_bytes"
        gemini_client.tool_call_thought_signatures[tool_call_id] = test_signature
        gemini_client.tool_call_function_map[tool_call_id] = "test_function"

        # Create a message with tool_calls that should be converted
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "function": {"name": "test_function", "arguments": '{"arg1": "value1"}'},
                    "type": "function",
                }
            ],
        }

        # Convert to Gemini content
        parts, part_type = gemini_client._oai_content_to_gemini_content(message)

        assert part_type == "tool_call"
        assert len(parts) == 1

        # Verify the Part has the thought_signature
        part = parts[0]
        assert isinstance(part, Part)
        assert part.function_call is not None
        assert part.function_call.name == "test_function"
        assert part.thought_signature == test_signature

    def test_thought_signature_none_when_not_present(self, gemini_client):
        """Test that thought_signature is None when not available in mapping"""
        from google.genai.types import Part

        # Set up function map without thought_signature
        tool_call_id = "tool_without_signature"
        gemini_client.tool_call_function_map[tool_call_id] = "some_function"
        # Note: not adding to tool_call_thought_signatures

        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "function": {"name": "some_function", "arguments": "{}"},
                    "type": "function",
                }
            ],
        }

        parts, part_type = gemini_client._oai_content_to_gemini_content(message)

        assert part_type == "tool_call"
        part = parts[0]
        assert isinstance(part, Part)
        # thought_signature should be None (not set, which is fine for non-Gemini-3 models)
        assert part.thought_signature is None

    def test_thought_signature_captured_from_vertex_part_via_to_dict(self, gemini_client):
        """Vertex ``Part`` does not expose ``thought_signature`` as a direct attribute —
        it is only surfaced via ``part.to_dict()`` as a base64 string."""

        # Fake a Vertex-shaped part: no thought_signature attribute, but to_dict() exposes it.
        original = b"vertex_sig_value"
        vertex_part = MagicMock(spec=["function_call", "to_dict"])
        vertex_part.function_call = MagicMock(name="get_weather", args={"location": "Melbourne"})
        vertex_part.function_call.name = "get_weather"
        vertex_part.function_call.args = {"location": "Melbourne"}
        vertex_part.to_dict.return_value = {
            "function_call": {"name": "get_weather", "args": {"location": "Melbourne"}},
            "thought_signature": base64.b64encode(original).decode("ascii"),
        }

        tool_calls: list = []
        gemini_client._process_parts([vertex_part], tool_calls)

        assert len(tool_calls) == 1
        tc = tool_calls[0]
        assert tc.id in gemini_client.tool_call_thought_signatures
        assert gemini_client.tool_call_thought_signatures[tc.id] == original
        assert base64.b64decode(tc.thought_signature) == original

    def test_thought_signature_replayed_on_vertex_same_agent(self, gemini_client_with_credentials):
        """Vertex branch must attach thought_signature from the per-instance dict.

        Without this, Gemini 3 thinking models on Vertex reject the replayed
        function call with `400 INVALID_ARGUMENT ... missing a thought_signature`.
        """
        import base64

        client = gemini_client_with_credentials
        tool_call_id = "vertex_tool_same_agent"
        test_signature = b"vertex_sig_bytes"
        client.tool_call_thought_signatures[tool_call_id] = test_signature
        client.tool_call_function_map[tool_call_id] = "lookup"

        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "function": {"name": "lookup", "arguments": '{"q": "widget"}'},
                    "type": "function",
                }
            ],
        }

        parts, part_type = client._oai_content_to_gemini_content(message)

        assert part_type == "tool_call"
        assert len(parts) == 1
        raw = parts[0].to_dict()
        fc = raw.get("function_call") or raw.get("functionCall")
        assert fc and fc["name"] == "lookup"
        sig_b64 = raw.get("thought_signature") or raw.get("thoughtSignature")
        assert sig_b64 is not None
        assert base64.b64decode(sig_b64) == test_signature

    def test_thought_signature_replayed_on_vertex_cross_agent(self, gemini_client_with_credentials):
        """Vertex branch must honour the base64 signature carried on the tool_call dict
        (cross-agent handoff case — the receiving client has an empty instance dict)."""
        import base64

        client = gemini_client_with_credentials
        # Deliberately leave instance dict empty to simulate a different agent's client
        assert client.tool_call_thought_signatures == {}

        original_bytes = b"cross_agent_sig"
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "xid",
                    "function": {"name": "lookup", "arguments": "{}"},
                    "type": "function",
                    "thought_signature": base64.b64encode(original_bytes).decode("ascii"),
                }
            ],
        }

        parts, _ = client._oai_content_to_gemini_content(message)

        raw = parts[0].to_dict()
        sig_b64 = raw.get("thought_signature") or raw.get("thoughtSignature")
        assert sig_b64 is not None
        assert base64.b64decode(sig_b64) == original_bytes

    @patch("autogen.oai.gemini.genai.Client")
    @patch("autogen.oai.gemini.calculate_gemini_cost")
    def test_thought_signature_embedded_in_tool_call_for_cross_agent(
        self, mock_calculate_cost, mock_generative_client, gemini_client
    ):
        """Test that thought_signature is base64-encoded and embedded in the tool call object.

        This is required for cross-agent routing: when Agent B receives Agent A's
        function call history, the signature must travel with the tool_call dict,
        not just live in Agent A's instance dict.
        """
        import base64

        mock_calculate_cost.return_value = 0.001

        mock_chat = MagicMock()
        mock_generative_client.return_value.chats.create.return_value = mock_chat

        mock_fn_call = MagicMock()
        mock_fn_call.name = "get_weather"
        mock_fn_call.args = {"city": "Tokyo"}

        mock_part = MagicMock()
        mock_part.function_call = mock_fn_call
        mock_part.text = ""
        mock_part.thought_signature = b"cross_agent_signature_bytes"

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 10
        mock_usage_metadata.candidates_token_count = 5

        mock_response = MagicMock(spec=GenerateContentResponse)
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [mock_part]
        mock_response.candidates[0].finish_reason = None
        mock_response.usage_metadata = mock_usage_metadata
        mock_chat.send_message.return_value = mock_response

        response = gemini_client.create({
            "model": "gemini-3-flash",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                    },
                },
            ],
        })

        tool_call = response.choices[0].message.tool_calls[0]

        # Verify thought_signature is embedded as base64 string on the tool call
        assert hasattr(tool_call, "thought_signature")
        assert isinstance(tool_call.thought_signature, str)
        assert base64.b64decode(tool_call.thought_signature) == b"cross_agent_signature_bytes"

        # Also verify it's still in the instance dict
        assert tool_call.id in gemini_client.tool_call_thought_signatures
        assert gemini_client.tool_call_thought_signatures[tool_call.id] == b"cross_agent_signature_bytes"

    def test_thought_signature_reconstructed_from_tool_call_dict_cross_agent(self, gemini_client):
        """Test that reconstruction reads thought_signature from the tool call dict.

        Simulates the cross-agent scenario: Agent B's client has an empty
        tool_call_thought_signatures dict, but the tool_call dict carries the
        base64-encoded signature from Agent A.
        """
        import base64

        from google.genai.types import Part

        tool_call_id = "cross_agent_tool_456"
        original_signature = b"original_signature_bytes"
        encoded_signature = base64.b64encode(original_signature).decode("ascii")

        # Set up function map but NOT the instance dict (simulates Agent B)
        gemini_client.tool_call_function_map[tool_call_id] = "get_weather"
        # Note: NOT adding to tool_call_thought_signatures — empty like Agent B's client

        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
                    "type": "function",
                    "thought_signature": encoded_signature,  # Carried from Agent A
                }
            ],
        }

        parts, part_type = gemini_client._oai_content_to_gemini_content(message)

        assert part_type == "tool_call"
        assert len(parts) == 1
        part = parts[0]
        assert isinstance(part, Part)
        assert part.function_call.name == "get_weather"
        # Signature should be decoded from the tool call dict
        assert part.thought_signature == original_signature

    def test_thought_signature_instance_dict_fallback_when_not_in_tool_call(self, gemini_client):
        """Test that reconstruction falls back to instance dict when tool call has no signature.

        This is the same-agent scenario (original behavior) — the signature lives
        in the instance dict, not in the tool call dict.
        """
        from google.genai.types import Part

        tool_call_id = "same_agent_tool_789"
        test_signature = b"instance_dict_signature"

        gemini_client.tool_call_function_map[tool_call_id] = "get_weather"
        gemini_client.tool_call_thought_signatures[tool_call_id] = test_signature

        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
                    "type": "function",
                    # No thought_signature key — same-agent path
                }
            ],
        }

        parts, part_type = gemini_client._oai_content_to_gemini_content(message)

        assert part_type == "tool_call"
        part = parts[0]
        assert isinstance(part, Part)
        assert part.thought_signature == test_signature

    def test_thought_signature_not_embedded_when_absent(self, gemini_client):
        """Test that no thought_signature is added to tool call when model doesn't provide one."""
        mock_fn_call = MagicMock()
        mock_fn_call.name = "get_weather"
        mock_fn_call.args = {"city": "NYC"}

        mock_part = MagicMock()
        mock_part.function_call = mock_fn_call
        mock_part.text = ""
        mock_part.thought_signature = None  # No signature (non-thinking model)

        # Directly test _process_parts
        autogen_tool_calls = []
        gemini_client._process_parts([mock_part], autogen_tool_calls)

        assert len(autogen_tool_calls) == 1
        tool_call = autogen_tool_calls[0]
        # Should NOT have thought_signature attribute
        assert not hasattr(tool_call, "thought_signature") or tool_call.thought_signature is None

    @patch("autogen.oai.gemini.genai.Client")
    @patch("autogen.oai.gemini.calculate_gemini_cost")
    def test_streaming_text_response(self, mock_calculate_cost, mock_generative_client, gemini_client):
        """Test that streaming accumulates text chunks and emits StreamEvents."""
        mock_calculate_cost.return_value = 0.001

        mock_chat = MagicMock()
        mock_generative_client.return_value.chats.create.return_value = mock_chat

        # Create streaming chunks
        def make_chunk(text, prompt_tokens=0, completion_tokens=0):
            chunk = MagicMock(spec=GenerateContentResponse)
            mock_part = MagicMock()
            mock_part.text = text
            mock_part.function_call = None
            mock_part.thought_signature = None
            mock_candidate = MagicMock()
            mock_candidate.content.parts = [mock_part]
            mock_candidate.finish_reason = None
            chunk.candidates = [mock_candidate]
            chunk.usage_metadata = MagicMock()
            chunk.usage_metadata.prompt_token_count = prompt_tokens
            chunk.usage_metadata.candidates_token_count = completion_tokens
            return chunk

        chunks = [
            make_chunk("Hello", prompt_tokens=10, completion_tokens=1),
            make_chunk(", ", completion_tokens=2),
            make_chunk("world!", prompt_tokens=10, completion_tokens=3),
        ]

        mock_chat.send_message_stream.return_value = iter(chunks)

        response = gemini_client.create({
            "model": "gemini-pro",
            "messages": [{"content": "Hi", "role": "user"}],
            "stream": True,
        })

        assert response.choices[0].message.content == "Hello, world!"
        assert response.choices[0].finish_reason == "stop"
        assert not response.choices[0].message.tool_calls
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 3
        mock_chat.send_message_stream.assert_called_once()

    @patch("autogen.oai.gemini.genai.Client")
    @patch("autogen.oai.gemini.calculate_gemini_cost")
    def test_streaming_with_tool_calls(self, mock_calculate_cost, mock_generative_client, gemini_client):
        """Test that streaming handles function calls properly."""
        mock_calculate_cost.return_value = 0.001

        mock_chat = MagicMock()
        mock_generative_client.return_value.chats.create.return_value = mock_chat

        # Create a chunk with a function call
        chunk = MagicMock(spec=GenerateContentResponse)
        mock_fn_call = MagicMock()
        mock_fn_call.name = "get_weather"
        mock_fn_call.args = MagicMock()
        mock_fn_call.args.items.return_value = [("city", "London")]

        mock_part = MagicMock()
        mock_part.text = ""
        mock_part.function_call = mock_fn_call
        mock_part.thought_signature = None

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_candidate.finish_reason = None
        chunk.candidates = [mock_candidate]
        chunk.usage_metadata = MagicMock()
        chunk.usage_metadata.prompt_token_count = 20
        chunk.usage_metadata.candidates_token_count = 5

        mock_chat.send_message_stream.return_value = iter([chunk])

        response = gemini_client.create({
            "model": "gemini-pro",
            "messages": [{"content": "What's the weather?", "role": "user"}],
            "stream": True,
        })

        assert response.choices[0].finish_reason == "tool_calls"
        assert response.choices[0].message.tool_calls is not None
        assert len(response.choices[0].message.tool_calls) == 1
        assert response.choices[0].message.tool_calls[0].function.name == "get_weather"
        assert response.choices[0].message.content == ""

    @patch("autogen.oai.gemini.genai.Client")
    @patch("autogen.oai.gemini.calculate_gemini_cost")
    def test_streaming_emits_stream_events(self, mock_calculate_cost, mock_generative_client, gemini_client):
        """Test that StreamEvents are emitted during streaming."""

        mock_calculate_cost.return_value = 0.0

        mock_chat = MagicMock()
        mock_generative_client.return_value.chats.create.return_value = mock_chat

        chunk = MagicMock(spec=GenerateContentResponse)
        mock_part = MagicMock()
        mock_part.text = "streamed text"
        mock_part.function_call = None
        mock_part.thought_signature = None
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_candidate.finish_reason = None
        chunk.candidates = [mock_candidate]
        chunk.usage_metadata = MagicMock()
        chunk.usage_metadata.prompt_token_count = 5
        chunk.usage_metadata.candidates_token_count = 2

        mock_chat.send_message_stream.return_value = iter([chunk])

        mock_iostream = MagicMock()
        with patch("autogen.oai.gemini.IOStream.get_default", return_value=mock_iostream):
            gemini_client.create({
                "model": "gemini-pro",
                "messages": [{"content": "Hello", "role": "user"}],
                "stream": True,
            })

        # Verify StreamEvent was emitted
        mock_iostream.send.assert_called()
        sent_event = mock_iostream.send.call_args[0][0]
        # Navigate through wrapping to find the text content
        event = sent_event
        while hasattr(event, "content") and not isinstance(event.content, str):
            event = event.content
        assert event.content == "streamed text"

    @patch("autogen.oai.gemini.genai.Client")
    @patch("autogen.oai.gemini.calculate_gemini_cost")
    def test_non_streaming_does_not_call_send_message_stream(
        self, mock_calculate_cost, mock_generative_client, gemini_client
    ):
        """Test that non-streaming uses send_message, not send_message_stream."""
        mock_calculate_cost.return_value = 0.0

        mock_chat = MagicMock()
        mock_generative_client.return_value.chats.create.return_value = mock_chat

        mock_part = MagicMock()
        mock_part.text = "response"
        mock_part.function_call = None
        mock_part.thought_signature = None

        mock_response = MagicMock(spec=GenerateContentResponse)
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [mock_part]
        mock_response.candidates[0].finish_reason = None
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 5
        mock_response.usage_metadata.candidates_token_count = 2

        mock_chat.send_message.return_value = mock_response

        gemini_client.create({
            "model": "gemini-pro",
            "messages": [{"content": "Hi", "role": "user"}],
            "stream": False,
        })

        mock_chat.send_message.assert_called_once()
        mock_chat.send_message_stream.assert_not_called()

    def test_oai_content_to_gemini_content_missing_content_key(self, gemini_client):
        """Test that messages without a 'content' key are handled gracefully.

        Reproduces the KeyError crash when a DataPart-originated message
        (e.g. from A2UI action) enters the conversation history without
        a 'content' field.
        """
        from google.genai.types import Part

        message = {
            "role": "user",
            "version": "v0.9",
            "action": {"name": "approve_previews", "surfaceId": "marketing"},
        }

        parts, part_type = gemini_client._oai_content_to_gemini_content(message)

        assert part_type == "text"
        assert len(parts) == 1
        assert isinstance(parts[0], Part)
        # Should serialize non-role fields as JSON
        import json

        serialized = json.loads(parts[0].text)
        assert serialized["version"] == "v0.9"
        assert serialized["action"]["name"] == "approve_previews"
        assert "role" not in serialized

    @patch("autogen.oai.gemini.GenerativeModel")
    @patch("autogen.oai.gemini.vertexai.init")
    @patch("autogen.oai.gemini.calculate_gemini_cost")
    def test_vertexai_streaming(
        self, mock_calculate_cost, mock_init, mock_generative_model, gemini_client_with_credentials
    ):
        """Test that VertexAI streaming uses send_message with stream=True."""
        mock_calculate_cost.return_value = 0.001
        mock_init.return_value = None

        mock_chat = MagicMock()
        mock_model = MagicMock()
        mock_generative_model.return_value = mock_model
        mock_model.start_chat.return_value = mock_chat

        # VertexAI streaming returns an iterable of VertexAIGenerationResponse
        chunk = MagicMock(spec=VertexAIGenerationResponse)
        mock_part = MagicMock()
        mock_part.text = "vertex streamed"
        mock_part.function_call = None
        mock_part.thought_signature = None
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_candidate.finish_reason = None
        chunk.candidates = [mock_candidate]
        chunk.usage_metadata = MagicMock()
        chunk.usage_metadata.prompt_token_count = 10
        chunk.usage_metadata.candidates_token_count = 5

        mock_chat.send_message.return_value = iter([chunk])

        response = gemini_client_with_credentials.create({
            "model": "gemini-pro",
            "messages": [{"content": "Hello", "role": "user"}],
            "stream": True,
        })

        assert response.choices[0].message.content == "vertex streamed"
        # VertexAI passes stream=True to send_message
        mock_chat.send_message.assert_called_once()
        call_kwargs = mock_chat.send_message.call_args
        assert call_kwargs.kwargs.get("stream") is True or (
            len(call_kwargs.args) > 1 and call_kwargs[1].get("stream") is True
        )
