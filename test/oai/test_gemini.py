# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

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
        model="gemini-2.0-flash-lite",
        api_key="dummy_api_key",
        project_id="fake-project-id",
        location="us-west1",
        proxy="http://mock-test-proxy:90/",
    )
    expected = {
        "api_type": "gemini",
        "model": "gemini-2.0-flash-lite",
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

    def test_normalize_pydantic_schema_to_dict_with_refs(self, gemini_client):
        """Test _normalize_pydantic_schema_to_dict resolves $ref references."""

        # Define nested Pydantic models
        class Step(BaseModel):
            explanation: str
            output: str

        class MathReasoning(BaseModel):
            steps: list[Step]
            final_answer: str

        # Get normalized schema
        normalized = gemini_client.normalize_pydantic_schema_to_dict(MathReasoning, for_genai_api=False)

        # Verify $defs is removed
        assert "$defs" not in normalized

        # Verify $ref references are resolved
        assert "properties" in normalized
        assert "steps" in normalized["properties"]
        steps_schema = normalized["properties"]["steps"]

        # The $ref should be resolved to the actual Step schema
        assert "$ref" not in steps_schema.get("items", {})
        assert "properties" in steps_schema.get("items", {})
        assert "explanation" in steps_schema["items"]["properties"]
        assert "output" in steps_schema["items"]["properties"]

        # Verify final_answer is present
        assert "final_answer" in normalized["properties"]
        assert normalized["properties"]["final_answer"]["type"] == "string"

    def test_normalize_pydantic_schema_to_dict_with_additional_properties(self, gemini_client):
        """Test _normalize_pydantic_schema_to_dict converts additionalProperties for GenAI API."""

        # Define a model with dict[str, T] which creates additionalProperties
        class Extra(BaseModel):
            notes: str

        class Output(BaseModel):
            is_good: bool
            extra: dict[str, Extra]  # This creates additionalProperties in schema

        # Test with for_genai_api=True (should convert additionalProperties)
        normalized_genai = gemini_client.normalize_pydantic_schema_to_dict(Output, for_genai_api=True)

        # Verify $defs is removed
        assert "$defs" not in normalized_genai

        # Verify additionalProperties is converted to a regular property
        assert "properties" in normalized_genai
        assert "extra" in normalized_genai["properties"]
        extra_schema = normalized_genai["properties"]["extra"]

        # Should have properties (converted from additionalProperties)
        assert "properties" in extra_schema
        assert "additionalProperties" not in extra_schema

        # The converted property should be named "value" and contain the Extra schema
        assert "value" in extra_schema["properties"]
        value_schema = extra_schema["properties"]["value"]
        assert "properties" in value_schema
        assert "notes" in value_schema["properties"]

        # Test with for_genai_api=False (should keep additionalProperties for Vertex AI)
        normalized_vertexai = gemini_client.normalize_pydantic_schema_to_dict(Output, for_genai_api=False)

        # Verify $defs is removed
        assert "$defs" not in normalized_vertexai

        # For Vertex AI, additionalProperties might be kept (depending on implementation)
        # But $ref should still be resolved
        assert "properties" in normalized_vertexai
        assert "extra" in normalized_vertexai["properties"]
        extra_schema_vertexai = normalized_vertexai["properties"]["extra"]

        # $ref should be resolved
        assert "$ref" not in extra_schema_vertexai.get("additionalProperties", {})

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
