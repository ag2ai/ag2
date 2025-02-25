# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

import os
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel

from autogen.import_utils import optional_import_block, skip_on_missing_imports
from autogen.oai.gemini import GeminiClient

with optional_import_block() as result:
    from google.api_core.exceptions import InternalServerError
    from google.auth.credentials import Credentials
    from google.cloud.aiplatform.initializer import global_config as vertexai_global_config
    from google.genai.types import GenerateContentResponse
    from vertexai.generative_models import GenerationResponse as VertexAIGenerationResponse
    from vertexai.generative_models import HarmBlockThreshold as VertexAIHarmBlockThreshold
    from vertexai.generative_models import HarmCategory as VertexAIHarmCategory
    from vertexai.generative_models import SafetySetting as VertexAISafetySetting


@skip_on_missing_imports(["vertexai", "PIL", "google.auth", "google.api", "google.cloud", "google.genai"], "gemini")
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

        expected_safety_settings = {
            category: VertexAIHarmBlockThreshold.BLOCK_ONLY_HIGH for category in safety_settings
        }

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
        assert isinstance(response, ChatCompletion), (
            f"Response should be an instance of ChatCompletion - got {type(response)}"
        )
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
        # assert isinstance(response, ChatCompletion), "Response should be an instance of ChatCompletion"
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
            steps: List[Step]
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

    def test_convert_type_null_to_nullable(self):
        initial_schema = {
            "type": "object",
            "properties": {
                "additional_notes": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "description": "Additional notes",
                }
            },
            "required": [],
        }

        expected_schema = {
            "properties": {
                "additional_notes": {
                    "anyOf": [{"type": "string"}, {"nullable": True}],
                    "default": None,
                    "description": "Additional notes",
                }
            },
            "required": [],
            "type": "object",
        }
        converted_schema = GeminiClient._convert_type_null_to_nullable(initial_schema)
        assert converted_schema == expected_schema

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
