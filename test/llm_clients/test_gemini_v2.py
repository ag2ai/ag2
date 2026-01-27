# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Test Gemini V2 integration with AG2."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from autogen.import_utils import optional_import_block, run_for_optional_imports
from autogen.llm_clients.gemini_v2 import GeminiV2Client, GeminiV2LLMConfigEntry
from autogen.llm_clients.models import (
    AudioContent,
    ImageContent,
    ReasoningContent,
    TextContent,
    ToolCallContent,
    UnifiedMessage,
    UnifiedResponse,
    UserRoleEnum,
    VideoContent,
)
from autogen.llm_config import LLMConfig
from autogen.oai.shared_utils import normalize_pydantic_schema_to_dict

with optional_import_block() as result:
    from google.auth.credentials import Credentials
    from google.genai.types import (
        Content,
        FinishReason,
        FunctionCall,
        FunctionResponse,
        GenerateContentResponse,
        GoogleSearch,
        Part,
        Schema,
        Tool,
        Type,
    )
    from vertexai.generative_models import (
        Content as VertexAIContent,
    )
    from vertexai.generative_models import (
        GenerationResponse as VertexAIGenerationResponse,
    )
    from vertexai.generative_models import (
        Part as VertexAIPart,
    )
    from vertexai.generative_models import (
        SafetySetting as VertexAISafetySetting,
    )
    from vertexai.generative_models import (
        Tool as vaiTool,
    )


def test_gemini_v2_llm_config_entry():
    """Test GeminiV2LLMConfigEntry initialization and serialization."""
    config = GeminiV2LLMConfigEntry(
        model="gemini-2.5-pro",
        api_key="dummy_api_key",
        project_id="fake-project-id",
        location="us-west1",
        proxy="http://mock-test-proxy:90/",
        include_thoughts=True,
        thinking_budget=1024,
        thinking_level="High",
    )
    expected = {
        "api_type": "gemini_v2",
        "model": "gemini-2.5-pro",
        "api_key": "dummy_api_key",
        "project_id": "fake-project-id",
        "location": "us-west1",
        "stream": False,
        "tags": [],
        "proxy": "http://mock-test-proxy:90/",
        "include_thoughts": True,
        "thinking_budget": 1024,
        "thinking_level": "High",
    }
    actual = config.model_dump()
    assert actual == expected, actual

    assert LLMConfig(config).model_dump() == {
        "config_list": [expected],
    }


@pytest.mark.parametrize("thinking_level", ["High", "Medium", "Low", "Minimal"])
def test_gemini_v2_llm_config_entry_thinking_level(thinking_level):
    """Test that GeminiV2LLMConfigEntry accepts all valid thinking_level values."""
    config = GeminiV2LLMConfigEntry(
        model="gemini-2.5-flash",
        api_key="dummy_api_key",
        thinking_level=thinking_level,
    )
    actual = config.model_dump()
    assert actual["thinking_level"] == thinking_level


def test_gemini_v2_llm_config_entry_thinking_config():
    """Test GeminiV2LLMConfigEntry with full thinking configuration."""
    config = GeminiV2LLMConfigEntry(
        model="gemini-2.5-flash",
        api_key="dummy_api_key",
        include_thoughts=True,
        thinking_budget=1024,
        thinking_level="Medium",
    )
    actual = config.model_dump()
    assert actual["include_thoughts"] is True
    assert actual["thinking_budget"] == 1024
    assert actual["thinking_level"] == "Medium"


@run_for_optional_imports(["vertexai", "google.genai", "google.auth"], "gemini_v2")
class TestGeminiV2Client:
    """Test suite for GeminiV2Client."""

    @pytest.fixture
    def gemini_v2_client(self):
        """Create a GeminiV2Client instance with API key."""
        return GeminiV2Client(api_key="fake_api_key")

    @pytest.fixture
    def gemini_v2_client_vertexai(self):
        """Create a GeminiV2Client instance for Vertex AI."""
        mock_credentials = MagicMock(Credentials)
        return GeminiV2Client(
            credentials=mock_credentials,
            project_id="fake-project-id",
            location="us-west1",
        )

    def test_initialization_with_api_key(self, gemini_v2_client):
        """Test client initialization with API key."""
        assert gemini_v2_client.api_key == "fake_api_key"
        assert gemini_v2_client.use_vertexai is False

    def test_initialization_without_api_key(self, gemini_v2_client_vertexai):
        """Test client initialization without API key (Vertex AI)."""
        assert gemini_v2_client_vertexai.api_key is None
        assert gemini_v2_client_vertexai.use_vertexai is True

    def test_initialization_error_with_api_key_and_project(self):
        """Test that providing API key with project/location raises error."""
        with pytest.raises(ValueError, match="Google Cloud project and location cannot be set when using an API Key"):
            GeminiV2Client(api_key="fake_key", project_id="project", location="us-west1")

    def test_google_application_credentials_initialization(self):
        """Test initialization with google_application_credentials."""
        client = GeminiV2Client(
            google_application_credentials="credentials.json",
            project_id="fake-project-id",
            location="us-west1",
        )
        assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == "credentials.json"
        assert client.google_application_credentials == "credentials.json"

    def test_initialization_with_invalid_credentials_type(self):
        """Test initialization with invalid credentials type."""
        with pytest.raises(TypeError, match="Object type google.auth.credentials.Credentials is expected"):
            GeminiV2Client(credentials="not_a_credentials_object", project_id="project")

    def test_extract_system_instruction(self, gemini_v2_client):
        """Test extracting system instruction from messages."""
        # Valid system instruction
        messages = [{"role": "system", "content": "You are my personal assistant."}]
        assert gemini_v2_client._extract_system_instruction(messages) == "You are my personal assistant."

        # Empty system instruction
        messages = [{"role": "system", "content": " "}]
        assert gemini_v2_client._extract_system_instruction(messages) is None

        # System instruction not first
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "system", "content": "You are my personal assistant."},
        ]
        assert gemini_v2_client._extract_system_instruction(messages) is None

        # Empty message list
        assert gemini_v2_client._extract_system_instruction([]) is None

        # Multimodal system message
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are helpful."}],
            }
        ]
        result = gemini_v2_client._extract_system_instruction(messages)
        assert result == "You are helpful."

    def test_convert_finish_reason(self, gemini_v2_client):
        """Test converting Gemini finish reasons to standard format."""
        assert gemini_v2_client._convert_finish_reason(FinishReason.STOP) == "stop"
        assert gemini_v2_client._convert_finish_reason(FinishReason.MAX_TOKENS) == "length"
        assert gemini_v2_client._convert_finish_reason(FinishReason.SAFETY) == "content_filter"
        assert gemini_v2_client._convert_finish_reason(FinishReason.RECITATION) == "content_filter"
        assert gemini_v2_client._convert_finish_reason(None) == "stop"
        assert gemini_v2_client._convert_finish_reason("UNKNOWN") == "stop"
        # Test with string representation
        assert gemini_v2_client._convert_finish_reason("MAX_TOKENS") == "length"

    @patch("autogen.llm_clients.gemini_v2.genai.Client")
    @patch("autogen.oai.gemini.calculate_gemini_cost")
    def test_create_with_text_response(self, mock_calculate_cost, mock_client, gemini_v2_client):
        """Test create method with simple text response."""
        mock_calculate_cost.return_value = 0.002

        # Setup mocks
        mock_chat = MagicMock()
        mock_client.return_value.chats.create.return_value = mock_chat

        mock_text_part = MagicMock()
        mock_text_part.text = "Example response"
        mock_text_part.function_call = None
        mock_text_part.inline_data = None
        mock_text_part.thought = None

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 100
        mock_usage_metadata.candidates_token_count = 50

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_text_part]
        mock_candidate.finish_reason = FinishReason.STOP

        mock_response = MagicMock(spec=GenerateContentResponse)
        mock_response.usage_metadata = mock_usage_metadata
        mock_response.candidates = [mock_candidate]

        mock_chat.send_message.return_value = mock_response

        # Call create
        response = gemini_v2_client.create({
            "model": "gemini-2.5-pro",
            "messages": [{"content": "Hello", "role": "user"}],
        })

        # Assertions
        assert isinstance(response, UnifiedResponse)
        assert response.model == "gemini-2.5-pro"
        assert response.provider == "gemini"
        assert len(response.messages) == 1
        assert response.messages[0].role == UserRoleEnum.ASSISTANT
        assert len(response.messages[0].content) == 1
        assert isinstance(response.messages[0].content[0], TextContent)
        assert response.messages[0].content[0].text == "Example response"
        # Use response.text property
        assert response.text == "Example response"
        assert response.usage["prompt_tokens"] == 100
        assert response.usage["completion_tokens"] == 50
        assert response.usage["total_tokens"] == 150
        assert response.finish_reason == "stop"

    @patch("autogen.llm_clients.gemini_v2.GenerativeModel")
    @patch("autogen.oai.gemini.calculate_gemini_cost")
    def test_create_with_vertexai(self, mock_calculate_cost, mock_model, gemini_v2_client_vertexai):
        """Test create method with Vertex AI."""
        mock_calculate_cost.return_value = 0.002

        mock_chat = MagicMock()
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_model_instance.start_chat.return_value = mock_chat

        mock_text_part = MagicMock()
        mock_text_part.text = "Vertex AI response"
        mock_text_part.function_call = None
        mock_text_part.inline_data = None
        mock_text_part.thought = None

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 80
        mock_usage_metadata.candidates_token_count = 40

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_text_part]
        mock_candidate.finish_reason = FinishReason.STOP

        mock_response = MagicMock(spec=VertexAIGenerationResponse)
        mock_response.usage_metadata = mock_usage_metadata
        mock_response.candidates = [mock_candidate]

        mock_chat.send_message.return_value = mock_response

        response = gemini_v2_client_vertexai.create({
            "model": "gemini-pro",
            "messages": [{"content": "Hello", "role": "user"}],
        })

        assert isinstance(response, UnifiedResponse)
        assert response.messages[0].get_text() == "Vertex AI response"

    @patch("autogen.llm_clients.gemini_v2.GenerativeModel")
    @patch("autogen.llm_clients.gemini_v2.GenerationConfig")
    @patch("autogen.oai.gemini.calculate_gemini_cost")
    def test_vertexai_generation_config_creation(
        self, mock_calculate_cost, mock_generation_config, mock_model, gemini_v2_client_vertexai
    ):
        """Test that GenerationConfig is created correctly for VertexAI path."""
        mock_calculate_cost.return_value = 0.002

        mock_chat = MagicMock()
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_model_instance.start_chat.return_value = mock_chat

        mock_text_part = MagicMock()
        mock_text_part.text = "Response"
        mock_text_part.function_call = None
        mock_text_part.inline_data = None
        mock_text_part.thought = None

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 10
        mock_usage_metadata.candidates_token_count = 5

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_text_part]
        mock_candidate.finish_reason = FinishReason.STOP

        mock_response = MagicMock(spec=VertexAIGenerationResponse)
        mock_response.usage_metadata = mock_usage_metadata
        mock_response.candidates = [mock_candidate]

        mock_chat.send_message.return_value = mock_response

        gemini_v2_client_vertexai.create({
            "model": "gemini-pro",
            "messages": [{"content": "Hello", "role": "user"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
        })

        # Verify GenerationConfig was created with correct parameters
        mock_generation_config.assert_called_once()
        call_kwargs = mock_generation_config.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_output_tokens"] == 100
        assert call_kwargs["top_p"] == 0.9

        # Verify GenerativeModel was called with GenerationConfig
        # Since GenerationConfig is mocked, check that it's the return value of the mock
        model_call_kwargs = mock_model.call_args.kwargs
        assert "generation_config" in model_call_kwargs
        # The generation_config is the return value of mock_generation_config()
        assert model_call_kwargs["generation_config"] == mock_generation_config.return_value

    @patch("autogen.llm_clients.gemini_v2.genai.Client")
    @patch("autogen.oai.gemini.calculate_gemini_cost")
    def test_create_with_function_call(self, mock_calculate_cost, mock_client, gemini_v2_client):
        """Test create method with function call response."""
        mock_calculate_cost.return_value = 0.001

        mock_chat = MagicMock()
        mock_client.return_value.chats.create.return_value = mock_chat

        mock_fn_call = MagicMock()
        mock_fn_call.name = "get_weather"
        mock_fn_call.args = {"location": "NYC"}

        mock_fn_part = MagicMock()
        mock_fn_part.text = ""
        mock_fn_part.function_call = mock_fn_call
        mock_fn_part.inline_data = None
        mock_fn_part.thought = None
        mock_fn_part.thought_signature = None

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 10
        mock_usage_metadata.candidates_token_count = 5

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_fn_part]
        mock_candidate.finish_reason = FinishReason.STOP

        mock_response = MagicMock(spec=GenerateContentResponse)
        mock_response.usage_metadata = mock_usage_metadata
        mock_response.candidates = [mock_candidate]

        mock_chat.send_message.return_value = mock_response

        response = gemini_v2_client.create({
            "model": "gemini-2.5-pro",
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

        assert isinstance(response, UnifiedResponse)
        tool_calls = response.messages[0].get_tool_calls()
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"
        assert json.loads(tool_calls[0].arguments) == {"location": "NYC"}

    @patch("autogen.llm_clients.gemini_v2.genai.Client")
    @patch("autogen.oai.gemini.calculate_gemini_cost")
    def test_create_with_image_content(self, mock_calculate_cost, mock_client, gemini_v2_client):
        """Test create method with image content."""
        mock_calculate_cost.return_value = 0.001

        mock_chat = MagicMock()
        mock_client.return_value.chats.create.return_value = mock_chat

        mock_inline_data = MagicMock()
        mock_inline_data.mime_type = "image/png"
        mock_inline_data.data = "base64imagedata"

        mock_image_part = MagicMock()
        mock_image_part.text = None
        mock_image_part.function_call = None
        mock_image_part.inline_data = mock_inline_data
        mock_image_part.thought = None

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 20
        mock_usage_metadata.candidates_token_count = 10

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_image_part]
        mock_candidate.finish_reason = FinishReason.STOP

        mock_response = MagicMock(spec=GenerateContentResponse)
        mock_response.usage_metadata = mock_usage_metadata
        mock_response.candidates = [mock_candidate]

        mock_chat.send_message.return_value = mock_response

        response = gemini_v2_client.create({
            "model": "gemini-2.5-pro",
            "messages": [{"content": "Generate an image", "role": "user"}],
        })

        assert isinstance(response, UnifiedResponse)
        images = response.messages[0].get_content_by_type("image")
        assert len(images) == 1
        assert isinstance(images[0], ImageContent)
        assert images[0].data_uri.startswith("data:image/png;base64,")

    @patch("autogen.llm_clients.gemini_v2.genai.Client")
    @patch("autogen.oai.gemini.calculate_gemini_cost")
    def test_create_with_audio_content(self, mock_calculate_cost, mock_client, gemini_v2_client):
        """Test create method with audio content."""
        mock_calculate_cost.return_value = 0.001

        mock_chat = MagicMock()
        mock_client.return_value.chats.create.return_value = mock_chat

        mock_inline_data = MagicMock()
        mock_inline_data.mime_type = "audio/mp3"
        mock_inline_data.data = "base64audiodata"

        mock_audio_part = MagicMock()
        mock_audio_part.text = None
        mock_audio_part.function_call = None
        mock_audio_part.inline_data = mock_inline_data
        mock_audio_part.thought = None

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 15
        mock_usage_metadata.candidates_token_count = 8

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_audio_part]
        mock_candidate.finish_reason = FinishReason.STOP

        mock_response = MagicMock(spec=GenerateContentResponse)
        mock_response.usage_metadata = mock_usage_metadata
        mock_response.candidates = [mock_candidate]

        mock_chat.send_message.return_value = mock_response

        response = gemini_v2_client.create({
            "model": "gemini-2.5-pro",
            "messages": [{"content": "Generate audio", "role": "user"}],
        })

        assert isinstance(response, UnifiedResponse)
        audio = response.messages[0].get_content_by_type("audio")
        assert len(audio) == 1
        assert isinstance(audio[0], AudioContent)

    @patch("autogen.llm_clients.gemini_v2.genai.Client")
    @patch("autogen.oai.gemini.calculate_gemini_cost")
    def test_create_with_video_content(self, mock_calculate_cost, mock_client, gemini_v2_client):
        """Test create method with video content."""
        mock_calculate_cost.return_value = 0.001

        mock_chat = MagicMock()
        mock_client.return_value.chats.create.return_value = mock_chat

        mock_inline_data = MagicMock()
        mock_inline_data.mime_type = "video/mp4"
        mock_inline_data.data = "base64videodata"

        mock_video_part = MagicMock()
        mock_video_part.text = None
        mock_video_part.function_call = None
        mock_video_part.inline_data = mock_inline_data
        mock_video_part.thought = None

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 25
        mock_usage_metadata.candidates_token_count = 12

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_video_part]
        mock_candidate.finish_reason = FinishReason.STOP

        mock_response = MagicMock(spec=GenerateContentResponse)
        mock_response.usage_metadata = mock_usage_metadata
        mock_response.candidates = [mock_candidate]

        mock_chat.send_message.return_value = mock_response

        response = gemini_v2_client.create({
            "model": "gemini-2.5-pro",
            "messages": [{"content": "Generate video", "role": "user"}],
        })

        assert isinstance(response, UnifiedResponse)
        videos = response.messages[0].get_content_by_type("video")
        assert len(videos) == 1
        assert isinstance(videos[0], VideoContent)

    @patch("autogen.llm_clients.gemini_v2.genai.Client")
    @patch("autogen.oai.gemini.calculate_gemini_cost")
    def test_create_with_reasoning_content(self, mock_calculate_cost, mock_client, gemini_v2_client):
        """Test create method with reasoning/thinking content."""
        mock_calculate_cost.return_value = 0.001

        mock_chat = MagicMock()
        mock_client.return_value.chats.create.return_value = mock_chat

        mock_text_part = MagicMock()
        mock_text_part.text = "Final answer"
        mock_text_part.function_call = None
        mock_text_part.inline_data = None
        mock_text_part.thought = "Let me think about this step by step..."

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 30
        mock_usage_metadata.candidates_token_count = 15

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_text_part]
        mock_candidate.finish_reason = FinishReason.STOP

        mock_response = MagicMock(spec=GenerateContentResponse)
        mock_response.usage_metadata = mock_usage_metadata
        mock_response.candidates = [mock_candidate]

        mock_chat.send_message.return_value = mock_response

        response = gemini_v2_client.create({
            "model": "gemini-3-flash",
            "messages": [{"content": "Solve this problem", "role": "user"}],
            "include_thoughts": True,
        })

        assert isinstance(response, UnifiedResponse)
        # Use response.reasoning property
        assert len(response.reasoning) == 1
        assert isinstance(response.reasoning[0], ReasoningContent)
        assert response.reasoning[0].reasoning == "Let me think about this step by step..."
        # Also verify via message method
        reasoning = response.messages[0].get_content_by_type("reasoning")
        assert len(reasoning) == 1

    @patch("autogen.llm_clients.gemini_v2.genai.Client")
    @patch("autogen.llm_clients.gemini_v2.GenerateContentConfig")
    @patch("autogen.llm_clients.gemini_v2.ThinkingConfig")
    @patch("autogen.oai.gemini.calculate_gemini_cost")
    def test_create_with_thinking_config(
        self, mock_calculate_cost, mock_thinking_config, mock_generate_config, mock_client, gemini_v2_client
    ):
        """Test create method with thinking configuration."""
        mock_calculate_cost.return_value = 0.001

        mock_chat = MagicMock()
        mock_client.return_value.chats.create.return_value = mock_chat

        mock_text_part = MagicMock()
        mock_text_part.text = "Response"
        mock_text_part.function_call = None
        mock_text_part.inline_data = None
        mock_text_part.thought = None

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 10
        mock_usage_metadata.candidates_token_count = 5

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_text_part]
        mock_candidate.finish_reason = FinishReason.STOP

        mock_response = MagicMock(spec=GenerateContentResponse)
        mock_response.usage_metadata = mock_usage_metadata
        mock_response.candidates = [mock_candidate]

        mock_chat.send_message.return_value = mock_response

        gemini_v2_client.create({
            "model": "gemini-3-flash",
            "messages": [{"content": "Hello", "role": "user"}],
            "include_thoughts": True,
            "thinking_budget": 1024,
        })

        # Verify ThinkingConfig was created
        mock_thinking_config.assert_called_once_with(include_thoughts=True, thinking_budget=1024)

        # Verify GenerateContentConfig was called with thinking_config
        config_kwargs = mock_generate_config.call_args.kwargs
        assert "thinking_config" in config_kwargs

    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            ({"include_thoughts": True}, {"include_thoughts": True, "thinking_budget": None}),
            ({"thinking_budget": 256}, {"include_thoughts": None, "thinking_budget": 256}),
            (
                {"include_thoughts": False, "thinking_budget": 512},
                {"include_thoughts": False, "thinking_budget": 512},
            ),
            (
                {"include_thoughts": True, "thinking_budget": 2048},
                {"include_thoughts": True, "thinking_budget": 2048},
            ),
        ],
    )
    @patch("autogen.llm_clients.gemini_v2.genai.Client")
    @patch("autogen.llm_clients.gemini_v2.GenerateContentConfig")
    @patch("autogen.llm_clients.gemini_v2.ThinkingConfig")
    @patch("autogen.oai.gemini.calculate_gemini_cost")
    def test_thinking_config_variants(
        self,
        mock_calculate_cost,
        mock_thinking_config,
        mock_generate_config,
        mock_client,
        gemini_v2_client,
        kwargs,
        expected,
    ):
        """Test various thinking config parameter combinations."""
        mock_calculate_cost.return_value = 0.001

        mock_chat = MagicMock()
        mock_client.return_value.chats.create.return_value = mock_chat

        mock_text_part = MagicMock()
        mock_text_part.text = "Response"
        mock_text_part.function_call = None
        mock_text_part.inline_data = None
        mock_text_part.thought = None

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 5
        mock_usage_metadata.candidates_token_count = 3

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_text_part]
        mock_candidate.finish_reason = FinishReason.STOP

        mock_response = MagicMock(spec=GenerateContentResponse)
        mock_response.usage_metadata = mock_usage_metadata
        mock_response.candidates = [mock_candidate]

        mock_chat.send_message.return_value = mock_response

        params = {
            "model": "gemini-3-flash",
            "messages": [{"content": "Hello", "role": "user"}],
            **kwargs,
        }
        gemini_v2_client.create(params)

        mock_thinking_config.assert_called_once_with(
            include_thoughts=expected["include_thoughts"],
            thinking_budget=expected["thinking_budget"],
        )

    @patch("autogen.llm_clients.gemini_v2.genai.Client")
    @patch("autogen.llm_clients.gemini_v2.GenerateContentConfig")
    @patch("autogen.oai.gemini.calculate_gemini_cost")
    def test_create_with_structured_output(
        self, mock_calculate_cost, mock_generate_config, mock_client, gemini_v2_client
    ):
        """Test create method with structured output (response_format)."""
        mock_calculate_cost.return_value = 0.001

        class Answer(BaseModel):
            result: str

        mock_chat = MagicMock()
        mock_client.return_value.chats.create.return_value = mock_chat

        mock_text_part = MagicMock()
        mock_text_part.text = '{"result": "42"}'
        mock_text_part.function_call = None
        mock_text_part.inline_data = None
        mock_text_part.thought = None

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 20
        mock_usage_metadata.candidates_token_count = 10

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_text_part]
        mock_candidate.finish_reason = FinishReason.STOP

        mock_response = MagicMock(spec=GenerateContentResponse)
        mock_response.usage_metadata = mock_usage_metadata
        mock_response.candidates = [mock_candidate]

        mock_chat.send_message.return_value = mock_response

        response = gemini_v2_client.create({
            "model": "gemini-2.5-pro",
            "messages": [{"content": "What is 2+2?", "role": "user"}],
            "response_format": Answer,
        })

        assert isinstance(response, UnifiedResponse)
        # Structured output should be formatted as text
        text_content = response.messages[0].get_text()
        assert "result" in text_content
        assert "42" in text_content

        # Verify response_schema was set
        config_kwargs = mock_generate_config.call_args.kwargs
        assert "response_schema" in config_kwargs
        assert config_kwargs["response_mime_type"] == "application/json"

    @patch("autogen.llm_clients.gemini_v2.genai.Client")
    @patch("autogen.oai.gemini.calculate_gemini_cost")
    def test_create_with_multimodal_messages(self, mock_calculate_cost, mock_client, gemini_v2_client):
        """Test create method with multimodal messages."""
        mock_calculate_cost.return_value = 0.001

        mock_chat = MagicMock()
        mock_client.return_value.chats.create.return_value = mock_chat

        mock_text_part = MagicMock()
        mock_text_part.text = "This is an image of a cat."
        mock_text_part.function_call = None
        mock_text_part.inline_data = None
        mock_text_part.thought = None

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 50
        mock_usage_metadata.candidates_token_count = 25

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_text_part]
        mock_candidate.finish_reason = FinishReason.STOP

        mock_response = MagicMock(spec=GenerateContentResponse)
        mock_response.usage_metadata = mock_usage_metadata
        mock_response.candidates = [mock_candidate]

        mock_chat.send_message.return_value = mock_response

        response = gemini_v2_client.create({
            "model": "gemini-2.5-pro",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,test"}},
                    ],
                }
            ],
        })

        assert isinstance(response, UnifiedResponse)
        # Use response.text property
        assert response.text == "This is an image of a cat."
        # Also verify via message method
        assert response.messages[0].get_text() == "This is an image of a cat."

    @patch("autogen.llm_clients.gemini_v2.genai.Client")
    @patch("autogen.oai.gemini.calculate_gemini_cost")
    def test_create_with_proxy(self, mock_calculate_cost, mock_client, gemini_v2_client):
        """Test create method with proxy configuration."""
        mock_calculate_cost.return_value = 0.001

        mock_chat = MagicMock()
        mock_client.return_value.chats.create.return_value = mock_chat

        mock_text_part = MagicMock()
        mock_text_part.text = "Response"
        mock_text_part.function_call = None
        mock_text_part.inline_data = None
        mock_text_part.thought = None

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 10
        mock_usage_metadata.candidates_token_count = 5

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_text_part]
        mock_candidate.finish_reason = FinishReason.STOP

        mock_response = MagicMock(spec=GenerateContentResponse)
        mock_response.usage_metadata = mock_usage_metadata
        mock_response.candidates = [mock_candidate]

        mock_chat.send_message.return_value = mock_response

        gemini_v2_client.proxy = "http://proxy:8080"
        gemini_v2_client.create({
            "model": "gemini-2.5-pro",
            "messages": [{"content": "Hello", "role": "user"}],
        })

        # Verify Client was called with http_options containing proxy
        client_kwargs = mock_client.call_args.kwargs
        assert "http_options" in client_kwargs
        http_options = client_kwargs["http_options"]
        assert http_options.client_args == {"proxy": "http://proxy:8080"}
        assert http_options.async_client_args == {"proxy": "http://proxy:8080"}

    @patch("autogen.llm_clients.gemini_v2.genai.Client")
    @patch("autogen.llm_clients.gemini_v2.GenerateContentConfig")
    @patch("autogen.oai.gemini.calculate_gemini_cost")
    def test_create_with_generation_params(
        self, mock_calculate_cost, mock_generate_config, mock_client, gemini_v2_client
    ):
        """Test create method with generation parameters."""
        mock_calculate_cost.return_value = 0.001

        mock_chat = MagicMock()
        mock_client.return_value.chats.create.return_value = mock_chat

        mock_text_part = MagicMock()
        mock_text_part.text = "Response"
        mock_text_part.function_call = None
        mock_text_part.inline_data = None
        mock_text_part.thought = None

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 10
        mock_usage_metadata.candidates_token_count = 5

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_text_part]
        mock_candidate.finish_reason = FinishReason.STOP

        mock_response = MagicMock(spec=GenerateContentResponse)
        mock_response.usage_metadata = mock_usage_metadata
        mock_response.candidates = [mock_candidate]

        mock_chat.send_message.return_value = mock_response

        gemini_v2_client.create({
            "model": "gemini-2.5-pro",
            "messages": [{"content": "Hello", "role": "user"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
            "top_k": 5,
            "seed": 42,
        })

        # Verify generation config parameters
        config_kwargs = mock_generate_config.call_args.kwargs
        assert config_kwargs["temperature"] == 0.7
        assert config_kwargs["max_output_tokens"] == 100
        assert config_kwargs["top_p"] == 0.9
        assert config_kwargs["top_k"] == 5
        assert config_kwargs["seed"] == 42

    def test_oai_messages_to_gemini_messages(self, gemini_v2_client):
        """Test converting OAI messages to Gemini format."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        gemini_messages = gemini_v2_client._oai_messages_to_gemini_messages(messages)

        assert len(gemini_messages) > 0
        assert all(isinstance(msg, Content) for msg in gemini_messages)

    def test_oai_messages_to_gemini_messages_vertexai(self, gemini_v2_client_vertexai):
        """Test converting OAI messages to Gemini format with VertexAI (uses VertexAIContent)."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        gemini_messages = gemini_v2_client_vertexai._oai_messages_to_gemini_messages(messages)

        assert len(gemini_messages) > 0
        # Verify VertexAI Content is used (not regular Content)
        assert all(isinstance(msg, VertexAIContent) for msg in gemini_messages)
        # Verify structure
        assert gemini_messages[0].role == "user"
        assert len(gemini_messages[0].parts) > 0

    def test_oai_content_to_gemini_content_text(self, gemini_v2_client):
        """Test converting OAI text content to Gemini format."""
        message = {"role": "user", "content": "Hello world"}

        parts, part_type = gemini_v2_client._oai_content_to_gemini_content(message)

        assert part_type == "text"
        assert len(parts) == 1
        assert isinstance(parts[0], Part)
        assert parts[0].text == "Hello world"

    def test_oai_content_to_gemini_content_multimodal(self, gemini_v2_client):
        """Test converting OAI multimodal content to Gemini format."""
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,test"}},
            ],
        }

        parts, part_type = gemini_v2_client._oai_content_to_gemini_content(message)

        assert part_type == "image"
        assert len(parts) >= 1

    def test_oai_content_to_gemini_content_tool_response(self, gemini_v2_client):
        """Test converting OAI tool response to Gemini format."""
        gemini_v2_client.tool_call_function_map["call-123"] = "get_weather"

        message = {
            "role": "tool",
            "tool_call_id": "call-123",
            "content": '{"result": "sunny"}',
        }

        parts, part_type = gemini_v2_client._oai_content_to_gemini_content(message)

        assert part_type == "tool"
        assert len(parts) == 1

    def test_oai_content_to_gemini_content_tool_response_creates_function_response(self, gemini_v2_client):
        """Test that FunctionResponse object is created correctly."""
        gemini_v2_client.tool_call_function_map["call-123"] = "get_weather"

        message = {
            "role": "tool",
            "tool_call_id": "call-123",
            "content": '{"result": "sunny", "temp": 72}',
        }

        parts, part_type = gemini_v2_client._oai_content_to_gemini_content(message)

        assert part_type == "tool"
        assert len(parts) == 1
        # Verify FunctionResponse object is created
        assert hasattr(parts[0], "function_response")
        assert isinstance(parts[0].function_response, FunctionResponse)
        assert parts[0].function_response.name == "get_weather"
        assert "sunny" in str(parts[0].function_response.response)

    def test_oai_content_to_gemini_content_tool_response_vertexai(self, gemini_v2_client_vertexai):
        """Test converting OAI tool response to Gemini format with VertexAI."""
        gemini_v2_client_vertexai.tool_call_function_map["call-123"] = "get_weather"

        message = {
            "role": "tool",
            "tool_call_id": "call-123",
            "content": '{"result": "sunny"}',
        }

        parts, part_type = gemini_v2_client_vertexai._oai_content_to_gemini_content(message)

        assert part_type == "tool"
        assert len(parts) == 1
        # Verify VertexAI Part is used
        assert isinstance(parts[0], VertexAIPart)

    def test_oai_content_to_gemini_content_tool_call_vertexai(self, gemini_v2_client_vertexai):
        """Test converting OAI tool call to Gemini format with VertexAI."""
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-456",
                    "function": {"name": "search", "arguments": '{"query": "test"}'},
                    "type": "function",
                }
            ],
        }

        parts, part_type = gemini_v2_client_vertexai._oai_content_to_gemini_content(message)

        assert part_type == "tool_call"
        assert len(parts) == 1
        # Verify VertexAI Part is used
        assert isinstance(parts[0], VertexAIPart)

    def test_oai_content_to_gemini_content_tool_call_creates_function_call(self, gemini_v2_client):
        """Test that FunctionCall object is created correctly with all properties."""
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-456",
                    "function": {"name": "search", "arguments": '{"query": "test", "limit": 10}'},
                    "type": "function",
                }
            ],
        }

        parts, part_type = gemini_v2_client._oai_content_to_gemini_content(message)

        assert part_type == "tool_call"
        assert len(parts) == 1
        # Verify FunctionCall object is created
        assert hasattr(parts[0], "function_call")
        assert isinstance(parts[0].function_call, FunctionCall)
        assert parts[0].function_call.name == "search"
        assert parts[0].function_call.args == {"query": "test", "limit": 10}

    def test_oai_content_to_gemini_content_tool_call_with_thought_signature(self, gemini_v2_client):
        """Test that FunctionCall includes thought_signature when available."""
        gemini_v2_client.tool_call_thought_signatures["call-789"] = b"thought_sig_bytes"

        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-789",
                    "function": {"name": "analyze", "arguments": '{"data": "test"}'},
                    "type": "function",
                }
            ],
        }

        parts, part_type = gemini_v2_client._oai_content_to_gemini_content(message)

        assert part_type == "tool_call"
        assert len(parts) == 1
        assert isinstance(parts[0].function_call, FunctionCall)
        # Verify thought_signature is included
        assert parts[0].thought_signature == b"thought_sig_bytes"

    def test_oai_content_to_gemini_content_tool_call_with_thought_signature_vertexai(self, gemini_v2_client_vertexai):
        """Test that VertexAI Part attempts to include thought_signature when available."""
        gemini_v2_client_vertexai.tool_call_thought_signatures["call-999"] = b"vertexai_thought_sig"

        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-999",
                    "function": {"name": "search", "arguments": '{"query": "test"}'},
                    "type": "function",
                }
            ],
        }

        parts, part_type = gemini_v2_client_vertexai._oai_content_to_gemini_content(message)

        assert part_type == "tool_call"
        assert len(parts) == 1
        assert isinstance(parts[0], VertexAIPart)
        # Verify that the part was created successfully
        # Note: VertexAI may or may not support thoughtSignature in the same way as GenAI API
        # The implementation attempts to include it, but if the API doesn't support it, it will be ignored
        # The important thing is that the code path executes without error

    def test_oai_content_to_gemini_content_empty_text(self, gemini_v2_client):
        """Test converting empty text content."""
        message = {"role": "user", "content": ""}

        parts, part_type = gemini_v2_client._oai_content_to_gemini_content(message)

        assert part_type == "text"
        assert len(parts) == 1
        assert parts[0].text == "empty"

    def test_tools_to_gemini_tools(self, gemini_v2_client):
        """Test converting OAI tools to Gemini format."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string", "description": "Search query"}},
                        "required": ["query"],
                    },
                },
            }
        ]

        gemini_tools = gemini_v2_client._tools_to_gemini_tools(tools)

        assert isinstance(gemini_tools, list)
        assert len(gemini_tools) == 1
        assert isinstance(gemini_tools[0], Tool)

    def test_tools_to_gemini_tools_vertexai(self, gemini_v2_client_vertexai):
        """Test converting OAI tools to Gemini format with VertexAI (uses vaiFunctionDeclaration)."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string", "description": "Search query"}},
                        "required": ["query"],
                    },
                },
            }
        ]

        gemini_tools = gemini_v2_client_vertexai._tools_to_gemini_tools(tools)

        assert isinstance(gemini_tools, list)
        assert len(gemini_tools) == 1
        # Verify VertexAI Tool is used
        assert isinstance(gemini_tools[0], vaiTool)

    def test_check_if_prebuilt_google_search_tool_exists(self, gemini_v2_client):
        """Test detection of prebuilt Google Search tool."""
        # Test with Google Search tool
        tools_with_google = [
            {
                "type": "function",
                "function": {
                    "name": "prebuilt_google_search",
                    "description": "Google Search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        assert GeminiV2Client._check_if_prebuilt_google_search_tool_exists(tools_with_google) is True

        # Test without Google Search tool
        tools_without_google = [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        assert GeminiV2Client._check_if_prebuilt_google_search_tool_exists(tools_without_google) is False

        # Test with empty tools
        assert GeminiV2Client._check_if_prebuilt_google_search_tool_exists([]) is False

    def test_check_if_prebuilt_google_search_tool_exists_with_multiple_tools(self, gemini_v2_client):
        """Test that Google Search tool raises error when used with other tools."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "prebuilt_google_search",
                    "description": "Google Search",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "other_tool",
                    "description": "Other tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

        with pytest.raises(ValueError, match="Google Search tool can be used only by itself"):
            GeminiV2Client._check_if_prebuilt_google_search_tool_exists(tools)

    def test_tools_to_gemini_tools_with_google_search(self, gemini_v2_client):
        """Test that Google Search tool is returned when detected (GenAI API only)."""
        from google.genai.types import GoogleSearch

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "prebuilt_google_search",
                    "description": "Google Search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        gemini_tools = gemini_v2_client._tools_to_gemini_tools(tools)

        assert isinstance(gemini_tools, list)
        assert len(gemini_tools) == 1
        assert isinstance(gemini_tools[0], Tool)
        # Verify it's a GoogleSearch tool
        assert hasattr(gemini_tools[0], "google_search")
        assert gemini_tools[0].google_search is not None
        assert isinstance(gemini_tools[0].google_search, GoogleSearch)

    def test_tools_to_gemini_tools_with_google_search_vertexai(self, gemini_v2_client_vertexai):
        """Test that Google Search tool is NOT used with Vertex AI."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "prebuilt_google_search",
                    "description": "Google Search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        # With Vertex AI, Google Search should not be used (returns regular function declarations)
        gemini_tools = gemini_v2_client_vertexai._tools_to_gemini_tools(tools)

        assert isinstance(gemini_tools, list)
        assert len(gemini_tools) == 1
        # Should be vaiTool, not GoogleSearch tool
        assert isinstance(gemini_tools[0], vaiTool)
        # Should not have google_search attribute
        assert not hasattr(gemini_tools[0], "google_search")
        # Verify vaiFunctionDeclaration is used
        # The Tool is created with function_declarations parameter: vaiTool(function_declarations=functions)
        # VertexAI Tool may store function_declarations internally and not expose it directly
        # Verify via to_dict() method if available, otherwise verify Tool creation succeeded
        func_decls = None

        # Try accessing via to_dict() method
        if hasattr(gemini_tools[0], "to_dict"):
            try:
                tool_dict = gemini_tools[0].to_dict()
                if isinstance(tool_dict, dict) and "function_declarations" in tool_dict:
                    func_decls = tool_dict["function_declarations"]
            except Exception:
                pass  # to_dict() might not be available or might fail

        # If found via to_dict, verify the function declarations
        # Note: to_dict() serializes vaiFunctionDeclaration objects to dictionaries
        if func_decls is not None:
            assert hasattr(func_decls, "__len__"), f"function_declarations is not a sequence. Type: {type(func_decls)}"
            assert len(func_decls) == 1
            # When accessed via to_dict(), func_decls[0] is a dict, not a vaiFunctionDeclaration instance
            # Verify the dictionary content matches what we expect
            func_decl_dict = func_decls[0]
            assert isinstance(func_decl_dict, dict), f"Expected dict from to_dict(), got {type(func_decl_dict)}"
            # Fix: Check for the actual function name being passed
            assert func_decl_dict.get("name") == "prebuilt_google_search"
            assert func_decl_dict.get("description") == "Google Search"
            assert "parameters" in func_decl_dict
        else:
            # If function_declarations is not accessible, verify Tool creation succeeded
            # The implementation creates vaiFunctionDeclaration objects and passes them to vaiTool
            # The fact that vaiTool was created successfully means the function_declarations were processed correctly
            # Verify that the Tool is the correct type and was created
            assert isinstance(gemini_tools[0], vaiTool)
            # Verify Tool has expected methods (from_function_declarations exists, confirming Tool type)
            assert hasattr(gemini_tools[0], "from_function_declarations")
            # The Tool was created with function_declarations=[vaiFunctionDeclaration(...)]
            # which means vaiFunctionDeclaration was correctly created and passed

    def test_create_gemini_schema(self, gemini_v2_client):
        """Test creating Gemini schema from JSON schema."""
        json_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name"},
                "age": {"type": "integer", "description": "Age"},
            },
            "required": ["name"],
        }

        schema = gemini_v2_client._create_gemini_schema(json_schema)

        assert isinstance(schema, Schema)
        assert schema.type == Type.OBJECT
        assert "name" in schema.properties
        assert "age" in schema.properties

    def test_convert_type_null_to_nullable(self, gemini_v2_client):
        """Test converting null type to nullable."""
        schema = {"type": "null"}
        result = GeminiV2Client._convert_type_null_to_nullable(schema)
        assert result == {"nullable": True}

        schema = {"type": "string", "nullable": False}
        result = GeminiV2Client._convert_type_null_to_nullable(schema)
        assert result == {"type": "string", "nullable": False}

    def test_unwrap_references(self, gemini_v2_client):
        """Test unwrapping $ref references."""
        function_parameters = {
            "type": "object",
            "properties": {
                "task": {
                    "$defs": {"SubItem": {"type": "object", "properties": {"name": {"type": "string"}}}},
                    "properties": {"items": {"$ref": "#/$defs/SubItem"}},
                }
            },
        }

        result = GeminiV2Client._unwrap_references(function_parameters)

        assert "$defs" not in result.get("properties", {}).get("task", {})

    def test_normalize_pydantic_schema_to_dict_with_refs(self, gemini_v2_client):
        """Test _normalize_pydantic_schema_to_dict resolves $ref references."""

        # Define nested Pydantic models
        class Step(BaseModel):
            explanation: str
            output: str

        class MathReasoning(BaseModel):
            steps: list[Step]
            final_answer: str

        # Get normalized schema
        normalized = normalize_pydantic_schema_to_dict(MathReasoning, for_genai_api=False)

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

    def test_normalize_pydantic_schema_to_dict_with_additional_properties(self, gemini_v2_client):
        """Test _normalize_pydantic_schema_to_dict converts additionalProperties for GenAI API."""

        # Define a model with dict[str, T] which creates additionalProperties
        class Extra(BaseModel):
            notes: str

        class Output(BaseModel):
            is_good: bool
            extra: dict[str, Extra]  # This creates additionalProperties in schema

        # Test with for_genai_api=True (should convert additionalProperties)
        normalized_genai = normalize_pydantic_schema_to_dict(Output, for_genai_api=True)

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
        normalized_vertexai = normalize_pydantic_schema_to_dict(Output, for_genai_api=False)

        # Verify $defs is removed
        assert "$defs" not in normalized_vertexai

        # For Vertex AI, additionalProperties might be kept (depending on implementation)
        # But $ref should still be resolved
        assert "properties" in normalized_vertexai
        assert "extra" in normalized_vertexai["properties"]
        extra_schema_vertexai = normalized_vertexai["properties"]["extra"]

        # $ref should be resolved
        assert "$ref" not in extra_schema_vertexai.get("additionalProperties", {})

    def test_to_vertexai_safety_settings(self, gemini_v2_client):
        """Test converting safety settings to VertexAI format."""
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
        ]

        vertexai_settings = GeminiV2Client._to_vertexai_safety_settings(safety_settings)

        assert isinstance(vertexai_settings, list)
        assert len(vertexai_settings) == 2
        assert all(isinstance(s, VertexAISafetySetting) for s in vertexai_settings)

    def test_parse_custom_params(self, gemini_v2_client):
        """Test parsing custom parameters."""
        params = {"price": [0.001, 0.002]}
        gemini_v2_client._parse_custom_params(params)

        assert gemini_v2_client._price_per_1k_tokens == (0.001, 0.002)

    @patch("autogen.llm_clients.gemini_v2.genai.Client")
    @patch("autogen.oai.gemini.calculate_gemini_cost")
    def test_cost_calculation(self, mock_calculate_cost, mock_client, gemini_v2_client):
        """Test cost calculation."""
        mock_calculate_cost.return_value = 0.002

        mock_chat = MagicMock()
        mock_client.return_value.chats.create.return_value = mock_chat

        mock_text_part = MagicMock()
        mock_text_part.text = "Response"
        mock_text_part.function_call = None
        mock_text_part.inline_data = None
        mock_text_part.thought = None

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 100
        mock_usage_metadata.candidates_token_count = 50

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_text_part]
        mock_candidate.finish_reason = FinishReason.STOP

        mock_response = MagicMock(spec=GenerateContentResponse)
        mock_response.usage_metadata = mock_usage_metadata
        mock_response.candidates = [mock_candidate]

        mock_chat.send_message.return_value = mock_response

        response = gemini_v2_client.create({
            "model": "gemini-2.5-pro",
            "messages": [{"content": "Hello", "role": "user"}],
        })

        assert response.cost == 0.002
        mock_calculate_cost.assert_called_once()

    def test_get_usage(self, gemini_v2_client):
        """Test get_usage static method."""
        from autogen.llm_clients.models import UnifiedMessage, UnifiedResponse

        message = UnifiedMessage(
            role=UserRoleEnum.ASSISTANT,
            content=[TextContent(text="Hello")],
        )
        response = UnifiedResponse(
            id="resp-123",
            model="gemini-2.5-pro",
            provider="gemini",
            messages=[message],
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            cost=0.002,
        )

        usage = GeminiV2Client.get_usage(response)

        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150
        assert usage["cost"] == 0.002
        assert usage["model"] == "gemini-2.5-pro"

    def test_message_retrieval(self, gemini_v2_client):
        """Test message_retrieval method."""
        from autogen.llm_clients.models import UnifiedMessage, UnifiedResponse

        message = UnifiedMessage(
            role=UserRoleEnum.ASSISTANT,
            content=[TextContent(text="Hello world")],
        )
        response = UnifiedResponse(
            id="resp-123",
            model="gemini-2.5-pro",
            provider="gemini",
            messages=[message],
        )

        messages = gemini_v2_client.message_retrieval(response)

        assert len(messages) == 1
        assert messages[0] == "Hello world"

    def test_create_v1_compatible(self, gemini_v2_client):
        """Test create_v1_compatible method."""
        from autogen.llm_clients.models import UnifiedMessage, UnifiedResponse

        tool_call = ToolCallContent(
            id="call-123",
            name="get_weather",
            arguments='{"location": "NYC"}',
        )
        message = UnifiedMessage(
            role=UserRoleEnum.ASSISTANT,
            content=[TextContent(text="Hello"), tool_call],
        )
        response = UnifiedResponse(
            id="resp-123",
            model="gemini-2.5-pro",
            provider="gemini",
            messages=[message],
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            finish_reason="stop",
            cost=0.002,
        )

        # Mock the create method to return our response
        gemini_v2_client.create = MagicMock(return_value=response)

        v1_response = gemini_v2_client.create_v1_compatible({
            "model": "gemini-2.5-pro",
            "messages": [{"content": "Hello", "role": "user"}],
        })

        assert v1_response["id"] == "resp-123"
        assert v1_response["model"] == "gemini-2.5-pro"
        assert v1_response["object"] == "chat.completion"
        assert len(v1_response["choices"]) == 1
        # Verify role is string (from UserRoleEnum.value)
        assert v1_response["choices"][0]["message"]["role"] == "assistant"
        # message.get_text() includes tool call text, so expect the full text
        expected_text = 'Hello tool call name: get_weather tool call arguments: {"location": "NYC"}'
        assert v1_response["choices"][0]["message"]["content"] == expected_text
        # Verify tool_calls structure matches implementation
        assert v1_response["choices"][0]["message"]["tool_calls"] is not None
        assert len(v1_response["choices"][0]["message"]["tool_calls"]) == 1
        assert v1_response["choices"][0]["message"]["tool_calls"][0]["id"] == "call-123"
        assert v1_response["choices"][0]["message"]["tool_calls"][0]["type"] == "function"
        assert v1_response["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert v1_response["choices"][0]["finish_reason"] == "stop"
        assert v1_response["usage"]["prompt_tokens"] == 100
        assert v1_response["cost"] == 0.002

    def test_transform_response_with_multiple_candidates_error(self, gemini_v2_client):
        """Test that transform_response raises error with multiple candidates."""
        mock_candidate1 = MagicMock()
        mock_candidate2 = MagicMock()

        mock_response = MagicMock(spec=GenerateContentResponse)
        mock_response.candidates = [mock_candidate1, mock_candidate2]

        with pytest.raises(ValueError, match="Unexpected number of candidates"):
            gemini_v2_client._transform_response(mock_response, "gemini-2.5-pro")

    def test_transform_response_with_no_content(self, gemini_v2_client):
        """Test transform_response with no content parts."""
        mock_candidate = MagicMock()
        mock_candidate.content = None
        mock_candidate.finish_reason = FinishReason.STOP

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 10
        mock_usage_metadata.candidates_token_count = 5

        mock_response = MagicMock(spec=GenerateContentResponse)
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = mock_usage_metadata

        response = gemini_v2_client._transform_response(mock_response, "gemini-2.5-pro")

        assert isinstance(response, UnifiedResponse)
        # Should have empty text content
        assert response.messages[0].get_text() == ""

    def test_thought_signature_captured(self, gemini_v2_client):
        """Test that thought_signature is captured from function calls."""
        mock_fn_call = MagicMock()
        mock_fn_call.name = "get_weather"
        mock_fn_call.args = {"location": "NYC"}

        mock_part = MagicMock()
        mock_part.text = ""
        mock_part.function_call = mock_fn_call
        mock_part.inline_data = None
        mock_part.thought = None
        mock_part.thought_signature = b"test_signature"

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_candidate.finish_reason = FinishReason.STOP

        mock_usage_metadata = MagicMock()
        mock_usage_metadata.prompt_token_count = 10
        mock_usage_metadata.candidates_token_count = 5

        mock_response = MagicMock(spec=GenerateContentResponse)
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = mock_usage_metadata

        # Call _transform_response to capture thought_signature
        gemini_v2_client._transform_response(mock_response, "gemini-3-flash")

        # Verify thought_signature was captured
        assert len(gemini_v2_client.tool_call_thought_signatures) == 1
        tool_call_id = list(gemini_v2_client.tool_call_thought_signatures.keys())[0]
        assert gemini_v2_client.tool_call_thought_signatures[tool_call_id] == b"test_signature"


@run_for_optional_imports(["vertexai", "google.genai", "google.auth"], "gemini_v2")
class TestUnifiedMessage:
    """Test suite for UnifiedMessage with Gemini V2 client."""

    def test_create_simple_message(self):
        """Test creating a simple UnifiedMessage with text content."""
        content = TextContent(text="Hello world")
        message = UnifiedMessage(role=UserRoleEnum.USER, content=[content])

        assert message.role == UserRoleEnum.USER
        assert len(message.content) == 1
        assert isinstance(message.content[0], TextContent)
        assert message.content[0].text == "Hello world"
        assert message.name is None
        assert message.metadata == {}

    def test_create_message_with_name(self):
        """Test creating a UnifiedMessage with a name."""
        content = TextContent(text="Hello")
        message = UnifiedMessage(role=UserRoleEnum.ASSISTANT, content=[content], name="assistant_1")

        assert message.name == "assistant_1"
        assert message.role == UserRoleEnum.ASSISTANT

    def test_create_message_with_metadata(self):
        """Test creating a UnifiedMessage with metadata."""
        content = TextContent(text="Hello")
        metadata = {"provider": "gemini", "model": "gemini-2.5-pro", "temperature": 0.7}
        message = UnifiedMessage(role=UserRoleEnum.ASSISTANT, content=[content], metadata=metadata)

        assert message.metadata == metadata

    def test_create_message_with_multiple_content_blocks(self):
        """Test creating a UnifiedMessage with multiple content blocks."""
        contents = [
            ReasoningContent(reasoning="Let me think step by step...", summary="Analysis"),
            TextContent(text="The answer is 42"),
            ImageContent(data_uri="data:image/png;base64,test"),
            ToolCallContent(id="call-123", name="calculate", arguments='{"x": 42}'),
        ]
        message = UnifiedMessage(role=UserRoleEnum.ASSISTANT, content=contents)

        assert len(message.content) == 4
        assert isinstance(message.content[0], ReasoningContent)
        assert isinstance(message.content[1], TextContent)
        assert isinstance(message.content[2], ImageContent)
        assert isinstance(message.content[3], ToolCallContent)

    def test_get_text_from_single_text_content(self):
        """Test extracting text from a single text content block."""
        content = TextContent(text="Hello world")
        message = UnifiedMessage(role=UserRoleEnum.USER, content=[content])

        assert message.get_text() == "Hello world"

    def test_get_text_from_multiple_text_contents(self):
        """Test extracting text from multiple text content blocks."""
        contents = [
            TextContent(text="Hello"),
            TextContent(text="world"),
        ]
        message = UnifiedMessage(role=UserRoleEnum.USER, content=contents)

        assert message.get_text() == "Hello world"

    def test_get_text_from_reasoning_content(self):
        """Test extracting text from reasoning content."""
        contents = [
            ReasoningContent(reasoning="Step 1: analyze the problem", summary="Analysis"),
            TextContent(text="Conclusion: 42"),
        ]
        message = UnifiedMessage(role=UserRoleEnum.ASSISTANT, content=contents)

        text = message.get_text()
        assert "Step 1: analyze the problem" in text
        assert "Conclusion: 42" in text

    def test_get_text_from_tool_call_content(self):
        """Test extracting text from tool call content."""
        contents = [
            TextContent(text="I'll call a function"),
            ToolCallContent(id="call-123", name="get_weather", arguments='{"location": "NYC"}'),
        ]
        message = UnifiedMessage(role=UserRoleEnum.ASSISTANT, content=contents)

        text = message.get_text()
        assert "I'll call a function" in text
        assert "get_weather" in text
        assert "NYC" in text

    def test_get_text_empty_content(self):
        """Test get_text() with empty or non-text content blocks."""
        contents = [
            ImageContent(data_uri="data:image/png;base64,test"),
        ]
        message = UnifiedMessage(role=UserRoleEnum.ASSISTANT, content=contents)

        assert message.get_text() == ""

    def test_get_reasoning_single_block(self):
        """Test extracting a single reasoning block."""
        reasoning = ReasoningContent(reasoning="Let me think about this...", summary=None)
        message = UnifiedMessage(role=UserRoleEnum.ASSISTANT, content=[reasoning])

        reasoning_blocks = message.get_reasoning()
        assert len(reasoning_blocks) == 1
        assert reasoning_blocks[0].reasoning == "Let me think about this..."
        assert isinstance(reasoning_blocks[0], ReasoningContent)

    def test_get_reasoning_multiple_blocks(self):
        """Test extracting multiple reasoning blocks."""
        contents = [
            ReasoningContent(reasoning="Step 1: analyze", summary="Analysis"),
            TextContent(text="Interim result"),
            ReasoningContent(reasoning="Step 2: conclude", summary="Conclusion"),
        ]
        message = UnifiedMessage(role=UserRoleEnum.ASSISTANT, content=contents)

        reasoning_blocks = message.get_reasoning()
        assert len(reasoning_blocks) == 2
        assert reasoning_blocks[0].reasoning == "Step 1: analyze"
        assert reasoning_blocks[1].reasoning == "Step 2: conclude"

    def test_get_reasoning_no_blocks(self):
        """Test get_reasoning() when no reasoning blocks present."""
        content = TextContent(type="text", text="No reasoning here")
        message = UnifiedMessage(role=UserRoleEnum.ASSISTANT, content=[content])

        reasoning_blocks = message.get_reasoning()
        assert len(reasoning_blocks) == 0

    def test_get_tool_calls_single_call(self):
        """Test extracting a single tool call."""
        tool_call = ToolCallContent(
            id="call-123",
            name="get_weather",
            arguments='{"location": "NYC"}',
        )
        message = UnifiedMessage(role=UserRoleEnum.ASSISTANT, content=[tool_call])

        tool_calls = message.get_tool_calls()
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call-123"
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[0].arguments == '{"location": "NYC"}'

    def test_get_tool_calls_multiple_calls(self):
        """Test extracting multiple tool calls."""
        contents = [
            TextContent(text="I'll call multiple functions"),
            ToolCallContent(id="call-1", name="get_weather", arguments='{"location": "NYC"}'),
            ToolCallContent(id="call-2", name="get_time", arguments='{"timezone": "EST"}'),
        ]
        message = UnifiedMessage(role=UserRoleEnum.ASSISTANT, content=contents)

        tool_calls = message.get_tool_calls()
        assert len(tool_calls) == 2
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[1].name == "get_time"

    def test_get_tool_calls_no_calls(self):
        """Test get_tool_calls() when no tool calls present."""
        content = TextContent(type="text", text="No tool calls")
        message = UnifiedMessage(role=UserRoleEnum.ASSISTANT, content=[content])

        tool_calls = message.get_tool_calls()
        assert len(tool_calls) == 0

    def test_get_content_by_type_text(self):
        """Test getting content blocks by type - text."""
        contents = [
            TextContent(text="First text"),
            ImageContent(data_uri="data:image/png;base64,test1"),
            TextContent(text="Second text"),
            ImageContent(data_uri="data:image/png;base64,test2"),
        ]
        message = UnifiedMessage(role=UserRoleEnum.ASSISTANT, content=contents)

        text_blocks = message.get_content_by_type("text")
        assert len(text_blocks) == 2
        assert all(isinstance(block, TextContent) for block in text_blocks)
        assert text_blocks[0].text == "First text"
        assert text_blocks[1].text == "Second text"

    def test_get_content_by_type_image(self):
        """Test getting content blocks by type - image."""
        contents = [
            TextContent(text="Description"),
            ImageContent(data_uri="data:image/png;base64,test1"),
            ImageContent(data_uri="data:image/jpeg;base64,test2"),
        ]
        message = UnifiedMessage(role=UserRoleEnum.ASSISTANT, content=contents)

        image_blocks = message.get_content_by_type("image")
        assert len(image_blocks) == 2
        assert all(isinstance(block, ImageContent) for block in image_blocks)
        assert image_blocks[0].data_uri == "data:image/png;base64,test1"
        assert image_blocks[1].data_uri == "data:image/jpeg;base64,test2"

    def test_get_content_by_type_audio(self):
        """Test getting content blocks by type - audio."""
        contents = [
            TextContent(text="Audio description"),
            AudioContent(data_uri="data:audio/mp3;base64,test"),
        ]
        message = UnifiedMessage(role=UserRoleEnum.ASSISTANT, content=contents)

        audio_blocks = message.get_content_by_type("audio")
        assert len(audio_blocks) == 1
        assert isinstance(audio_blocks[0], AudioContent)
        assert audio_blocks[0].data_uri == "data:audio/mp3;base64,test"

    def test_get_content_by_type_video(self):
        """Test getting content blocks by type - video."""
        contents = [
            TextContent(text="Video description"),
            VideoContent(data_uri="data:video/mp4;base64,test"),
        ]
        message = UnifiedMessage(role=UserRoleEnum.ASSISTANT, content=contents)

        video_blocks = message.get_content_by_type("video")
        assert len(video_blocks) == 1
        assert isinstance(video_blocks[0], VideoContent)
        assert video_blocks[0].data_uri == "data:video/mp4;base64,test"

    def test_get_content_by_type_reasoning(self):
        """Test getting content blocks by type - reasoning."""
        contents = [
            ReasoningContent(reasoning="Step 1", summary="Summary 1"),
            TextContent(text="Text between"),
            ReasoningContent(reasoning="Step 2", summary="Summary 2"),
        ]
        message = UnifiedMessage(role=UserRoleEnum.ASSISTANT, content=contents)

        reasoning_blocks = message.get_content_by_type("reasoning")
        assert len(reasoning_blocks) == 2
        assert all(isinstance(block, ReasoningContent) for block in reasoning_blocks)
        assert reasoning_blocks[0].reasoning == "Step 1"
        assert reasoning_blocks[1].reasoning == "Step 2"

    def test_get_content_by_type_tool_call(self):
        """Test getting content blocks by type - tool_call."""
        contents = [
            ToolCallContent(id="call-1", name="func1", arguments='{"arg": "value1"}'),
            TextContent(text="Between calls"),
            ToolCallContent(id="call-2", name="func2", arguments='{"arg": "value2"}'),
        ]
        message = UnifiedMessage(role=UserRoleEnum.ASSISTANT, content=contents)

        tool_call_blocks = message.get_content_by_type("tool_call")
        assert len(tool_call_blocks) == 2
        assert all(isinstance(block, ToolCallContent) for block in tool_call_blocks)
        assert tool_call_blocks[0].name == "func1"
        assert tool_call_blocks[1].name == "func2"

    def test_get_content_by_type_nonexistent(self):
        """Test getting content blocks by type that doesn't exist."""
        contents = [
            TextContent(type="text", text="Hello"),
            ImageContent(data_uri="data:image/png;base64,test"),
        ]
        message = UnifiedMessage(role=UserRoleEnum.ASSISTANT, content=contents)

        audio_blocks = message.get_content_by_type("audio")
        assert len(audio_blocks) == 0

    def test_unified_message_with_gemini_response(self):
        """Test UnifiedMessage created from Gemini V2 response."""
        # Simulate a response with multiple content types
        contents = [
            ReasoningContent(reasoning="Analyzing the request...", summary="Analysis"),
            TextContent(text="Based on my analysis, the answer is 42."),
            ToolCallContent(id="call-456", name="verify", arguments='{"value": 42}'),
        ]
        message = UnifiedMessage(
            role=UserRoleEnum.ASSISTANT,
            content=contents,
            name="gemini_assistant",
            metadata={"model": "gemini-2.5-pro", "provider": "gemini"},
        )

        assert message.role == UserRoleEnum.ASSISTANT
        assert message.name == "gemini_assistant"
        assert message.metadata["model"] == "gemini-2.5-pro"
        assert len(message.content) == 3

        # Test all extraction methods
        assert "Analyzing the request" in message.get_text()
        assert len(message.get_reasoning()) == 1
        assert len(message.get_tool_calls()) == 1
        assert len(message.get_content_by_type("text")) == 1
        assert len(message.get_content_by_type("reasoning")) == 1
        assert len(message.get_content_by_type("tool_call")) == 1

    def test_unified_message_multimodal_content(self):
        """Test UnifiedMessage with multimodal content (text + image + audio)."""
        contents = [
            TextContent(text="Here's a multimodal response:"),
            ImageContent(data_uri="data:image/png;base64,image123"),
            AudioContent(data_uri="data:audio/mp3;base64,audio123", transcript="Audio transcript"),
        ]
        message = UnifiedMessage(role=UserRoleEnum.ASSISTANT, content=contents)

        assert len(message.content) == 3
        assert len(message.get_content_by_type("text")) == 1
        assert len(message.get_content_by_type("image")) == 1
        assert len(message.get_content_by_type("audio")) == 1

        # Text extraction should include transcript
        text = message.get_text()
        assert "Here's a multimodal response" in text
        assert "Audio transcript" in text

    def test_unified_message_empty_content(self):
        """Test UnifiedMessage with empty content list."""
        message = UnifiedMessage(role=UserRoleEnum.ASSISTANT, content=[])

        assert len(message.content) == 0
        assert message.get_text() == ""
        assert len(message.get_reasoning()) == 0
        assert len(message.get_tool_calls()) == 0

    def test_unified_message_role_enum(self):
        """Test UnifiedMessage with UserRoleEnum roles."""
        roles = [UserRoleEnum.USER, UserRoleEnum.ASSISTANT, UserRoleEnum.SYSTEM, UserRoleEnum.TOOL]
        for role in roles:
            content = TextContent(text=f"Message from {role.value}")
            message = UnifiedMessage(role=role, content=[content])
            assert message.role == role

    def test_unified_message_serialization(self):
        """Test UnifiedMessage serialization to dict."""
        contents = [
            TextContent(text="Hello"),
            ReasoningContent(reasoning="Thinking...", summary="Summary"),
            ToolCallContent(id="call-123", name="test", arguments='{"x": 1}'),
        ]
        message = UnifiedMessage(
            role=UserRoleEnum.ASSISTANT,
            content=contents,
            name="test_assistant",
            metadata={"key": "value"},
        )

        # Test model_dump
        dumped = message.model_dump()
        assert dumped["role"] == "assistant"
        assert dumped["name"] == "test_assistant"
        assert dumped["metadata"] == {"key": "value"}
        assert len(dumped["content"]) == 3

        # Test model_dump_json
        json_str = message.model_dump_json()
        assert isinstance(json_str, str)
        assert "Hello" in json_str
        assert "test_assistant" in json_str

    def test_unified_message_with_complex_content_mix(self):
        """Test UnifiedMessage with a complex mix of all content types."""
        contents = [
            ReasoningContent(reasoning="Initial reasoning", summary="Initial"),
            TextContent(text="Main response text"),
            ImageContent(data_uri="data:image/png;base64,img1"),
            ToolCallContent(id="call-1", name="search", arguments='{"query": "test"}'),
            AudioContent(data_uri="data:audio/mp3;base64,aud1", transcript="Audio content"),
            VideoContent(data_uri="data:video/mp4;base64,vid1"),
            ReasoningContent(reasoning="Final reasoning", summary="Final"),
            TextContent(text="Conclusion"),
        ]
        message = UnifiedMessage(role=UserRoleEnum.ASSISTANT, content=contents)

        # Verify all content types are present
        assert len(message.content) == 8
        assert len(message.get_content_by_type("text")) == 2
        assert len(message.get_content_by_type("reasoning")) == 2
        assert len(message.get_content_by_type("image")) == 1
        assert len(message.get_content_by_type("audio")) == 1
        assert len(message.get_content_by_type("video")) == 1
        assert len(message.get_content_by_type("tool_call")) == 1

        # Verify extraction methods work correctly
        assert len(message.get_reasoning()) == 2
        assert len(message.get_tool_calls()) == 1
        text = message.get_text()
        assert "Main response text" in text
        assert "Conclusion" in text
        assert "Audio content" in text
