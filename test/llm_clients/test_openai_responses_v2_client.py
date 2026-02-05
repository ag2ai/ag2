# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for OpenAIResponsesV2Client."""

from unittest.mock import Mock, patch

import pytest

from autogen.llm_clients.models import (
    CitationContent,
    GenericContent,
    ImageContent,
    ReasoningContent,
    TextContent,
    ToolCallContent,
    UnifiedResponse,
)
from autogen.llm_clients.openai_responses_v2 import (
    OpenAIResponsesV2Client,
    OpenAIResponsesV2LLMConfigEntry,
    calculate_image_cost,
    calculate_token_cost,
)

# =============================================================================
# Mock Classes for Testing
# =============================================================================


class MockResponsesAPIResponse:
    """Mock OpenAI Responses API response object."""

    def __init__(
        self,
        response_id="resp_test123",
        model="gpt-4.1",
        output=None,
        usage=None,
        created=1234567890,
        output_parsed=None,
    ):
        self.id = response_id
        self.model = model
        self.output = output or []
        self.usage = usage
        self.created = created
        self.output_parsed = output_parsed


class MockUsage:
    """Mock usage stats for Responses API."""

    def __init__(
        self,
        input_tokens=50,
        output_tokens=100,
        total_tokens=150,
        output_tokens_details=None,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens
        self.output_tokens_details = output_tokens_details or {}

    def model_dump(self):
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "output_tokens_details": self.output_tokens_details,
        }


class MockOutputItem:
    """Mock output item from Responses API."""

    def __init__(self, item_type, **kwargs):
        self._type = item_type
        self._data = kwargs

    def model_dump(self):
        return {"type": self._type, **self._data}


def create_mock_message_output(text="Hello, world!", annotations=None):
    """Create a mock message output item."""
    content = [{"type": "output_text", "text": text, "annotations": annotations or []}]
    return MockOutputItem("message", content=content)


def create_mock_reasoning_output(reasoning_text, summary=None):
    """Create a mock reasoning output item."""
    return MockOutputItem("reasoning", reasoning=reasoning_text, summary=summary)


def create_mock_function_call_output(call_id, name, arguments):
    """Create a mock function_call output item."""
    return MockOutputItem("function_call", call_id=call_id, name=name, arguments=arguments)


def create_mock_image_generation_output(result, size="1024x1024", quality="high"):
    """Create a mock image_generation_call output item."""
    return MockOutputItem("image_generation_call", result=result, size=size, quality=quality)


def create_mock_web_search_output(search_id, status="completed"):
    """Create a mock web_search_call output item."""
    return MockOutputItem("web_search_call", id=search_id, status=status)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_openai_client():
    """Create mock OpenAI client for Responses API."""
    with patch("autogen.llm_clients.openai_responses_v2.OpenAI") as mock_openai_class:
        mock_client_instance = Mock()
        mock_client_instance.responses = Mock()
        mock_openai_class.return_value = mock_client_instance
        yield mock_client_instance


@pytest.fixture
def client(mock_openai_client):
    """Create a client instance with mocked OpenAI."""
    return OpenAIResponsesV2Client(api_key="test-key")


# =============================================================================
# Test: Client Initialization
# =============================================================================


class TestOpenAIResponsesV2ClientCreation:
    """Test client initialization."""

    def test_create_client_with_api_key(self, mock_openai_client):
        """Test creating client with API key."""
        client = OpenAIResponsesV2Client(api_key="test-key")
        assert client is not None
        assert client.client is not None

    def test_create_client_with_base_url(self, mock_openai_client):
        """Test creating client with custom base URL."""
        client = OpenAIResponsesV2Client(api_key="test-key", base_url="https://custom.api.com")
        assert client is not None

    def test_create_client_with_response_format(self, mock_openai_client):
        """Test creating client with default response format."""
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("Pydantic not installed")

        class TestModel(BaseModel):
            name: str

        client = OpenAIResponsesV2Client(api_key="test-key", response_format=TestModel)
        assert client._default_response_format == TestModel

    def test_create_client_with_workspace_dir(self, mock_openai_client):
        """Test creating client with workspace directory for apply_patch."""
        client = OpenAIResponsesV2Client(api_key="test-key", workspace_dir="/custom/workspace")
        assert client._workspace_dir == "/custom/workspace"

    def test_client_has_required_methods(self, client):
        """Test that client has all ModelClient methods."""
        assert hasattr(client, "create")
        assert hasattr(client, "create_v1_compatible")
        assert hasattr(client, "cost")
        assert hasattr(client, "get_usage")
        assert hasattr(client, "message_retrieval")
        assert hasattr(client, "RESPONSE_USAGE_KEYS")
        assert hasattr(client, "reset_conversation")

    def test_initial_state(self, client):
        """Test initial client state."""
        assert client._previous_response_id is None
        assert client._image_costs == 0.0
        assert client._token_costs == 0.0
        assert client._total_prompt_tokens == 0
        assert client._total_completion_tokens == 0


# =============================================================================
# Test: LLMConfigEntry
# =============================================================================


class TestOpenAIResponsesV2LLMConfigEntry:
    """Test LLMConfigEntry for responses_v2 api_type."""

    def test_config_entry_api_type(self):
        """Test that config entry has correct api_type."""
        entry = OpenAIResponsesV2LLMConfigEntry(model="gpt-4.1")
        assert entry.api_type == "responses_v2"

    def test_config_entry_defaults(self):
        """Test config entry default values."""
        entry = OpenAIResponsesV2LLMConfigEntry(model="gpt-4.1")
        assert entry.model == "gpt-4.1"

    def test_create_client_raises_not_implemented(self):
        """Test that create_client raises NotImplementedError."""
        entry = OpenAIResponsesV2LLMConfigEntry(model="gpt-4.1")
        with pytest.raises(NotImplementedError):
            entry.create_client()


# =============================================================================
# Test: Stateful Conversation Management
# =============================================================================


class TestStatefulConversation:
    """Test stateful conversation management."""

    def test_get_previous_response_id_initial(self, client):
        """Test getting previous response ID when none set."""
        assert client._get_previous_response_id() is None

    def test_set_previous_response_id(self, client):
        """Test setting previous response ID."""
        client._set_previous_response_id("resp_12345")
        assert client._get_previous_response_id() == "resp_12345"

    def test_reset_conversation(self, client):
        """Test resetting conversation state."""
        client._set_previous_response_id("resp_12345")
        assert client._get_previous_response_id() == "resp_12345"

        client.reset_conversation()
        assert client._get_previous_response_id() is None

    def test_create_updates_state(self, client):
        """Test that create() updates conversation state."""
        # Setup mock response
        mock_response = MockResponsesAPIResponse(
            response_id="resp_new123",
            output=[create_mock_message_output("Hello!")],
            usage=MockUsage(),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        # Initial state
        assert client._get_previous_response_id() is None

        # Make request
        client.create({"model": "gpt-4.1", "messages": [{"role": "user", "content": "Hi"}]})

        # State should be updated
        assert client._get_previous_response_id() == "resp_new123"

    def test_create_uses_previous_response_id(self, client):
        """Test that create() uses previous response ID for stateful conversation."""
        # Set previous response ID
        client._set_previous_response_id("resp_previous")

        mock_response = MockResponsesAPIResponse(
            response_id="resp_new",
            output=[create_mock_message_output("Response")],
            usage=MockUsage(),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        # Make request
        client.create({"model": "gpt-4.1", "messages": [{"role": "user", "content": "Continue"}]})

        # Verify previous_response_id was passed
        call_kwargs = client.client.responses.create.call_args[1]
        assert call_kwargs.get("previous_response_id") == "resp_previous"


# =============================================================================
# Test: Basic Response Creation
# =============================================================================


class TestOpenAIResponsesV2ClientCreate:
    """Test create() method."""

    def test_create_simple_response(self, client):
        """Test creating a simple text response."""
        mock_response = MockResponsesAPIResponse(
            response_id="resp_test123",
            model="gpt-4.1",
            output=[create_mock_message_output("The answer is 42")],
            usage=MockUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        response = client.create({
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "What is 40+2?"}],
        })

        assert isinstance(response, UnifiedResponse)
        assert response.id == "resp_test123"
        assert response.model == "gpt-4.1"
        assert response.provider == "openai_responses"
        assert len(response.messages) == 1
        assert response.text == "The answer is 42"

    def test_create_response_with_reasoning(self, client):
        """Test creating response with reasoning blocks (o3 models)."""
        mock_response = MockResponsesAPIResponse(
            model="o3",
            output=[
                create_mock_reasoning_output(
                    "Step 1: Add 40 + 2\nStep 2: Result is 42",
                    summary="Calculated 40+2=42",
                ),
                create_mock_message_output("The answer is 42"),
            ],
            usage=MockUsage(),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        response = client.create({
            "model": "o3",
            "messages": [{"role": "user", "content": "What is 40+2?"}],
        })

        # Verify reasoning blocks are extracted
        assert len(response.reasoning) >= 1
        reasoning_block = response.reasoning[0]
        assert isinstance(reasoning_block, ReasoningContent)
        assert "Step 1" in reasoning_block.reasoning

    def test_create_response_with_tool_calls(self, client):
        """Test creating response with function tool calls."""
        mock_response = MockResponsesAPIResponse(
            output=[
                create_mock_function_call_output(
                    call_id="call_123",
                    name="get_weather",
                    arguments='{"city": "San Francisco"}',
                ),
            ],
            usage=MockUsage(),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        response = client.create({
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "Get weather in SF"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        })

        # Verify tool calls
        tool_calls = response.messages[0].get_tool_calls()
        assert len(tool_calls) == 1
        assert isinstance(tool_calls[0], ToolCallContent)
        assert tool_calls[0].id == "call_123"
        assert tool_calls[0].name == "get_weather"
        assert "San Francisco" in tool_calls[0].arguments

    def test_create_response_with_usage(self, client):
        """Test that usage information is properly extracted."""
        mock_response = MockResponsesAPIResponse(
            output=[create_mock_message_output("Test")],
            usage=MockUsage(input_tokens=100, output_tokens=200, total_tokens=300),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        response = client.create({"model": "gpt-4.1", "messages": []})

        assert response.usage["prompt_tokens"] == 100
        assert response.usage["completion_tokens"] == 200
        assert response.usage["total_tokens"] == 300

    def test_create_response_with_reasoning_tokens(self, client):
        """Test extracting reasoning tokens for o3 models."""
        mock_response = MockResponsesAPIResponse(
            model="o3",
            output=[create_mock_message_output("Answer")],
            usage=MockUsage(
                input_tokens=50,
                output_tokens=100,
                total_tokens=150,
                output_tokens_details={"reasoning_tokens": 80},
            ),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        response = client.create({"model": "o3", "messages": []})

        assert response.usage.get("reasoning_tokens") == 80


# =============================================================================
# Test: Built-in Tools
# =============================================================================


class TestBuiltInTools:
    """Test built-in tools support."""

    def test_create_with_web_search(self, client):
        """Test creating request with web_search built-in tool."""
        mock_response = MockResponsesAPIResponse(
            output=[
                create_mock_web_search_output("search_123", "completed"),
                create_mock_message_output(
                    "Latest AI news...",
                    annotations=[
                        {
                            "type": "url_citation",
                            "url": "https://example.com/ai-news",
                            "title": "AI News Article",
                            "start_index": 0,
                            "end_index": 10,
                        }
                    ],
                ),
            ],
            usage=MockUsage(),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        response = client.create({
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "Latest AI news?"}],
            "built_in_tools": ["web_search"],
        })

        # Verify tools were passed
        call_kwargs = client.client.responses.create.call_args[1]
        tools = call_kwargs.get("tools", [])
        assert any(t.get("type") == "web_search_preview" for t in tools)

        # Verify citations extracted
        citations = OpenAIResponsesV2Client.get_citations(response)
        assert len(citations) >= 1
        assert isinstance(citations[0], CitationContent)
        assert citations[0].url == "https://example.com/ai-news"

    def test_create_with_image_generation(self, client):
        """Test creating request with image_generation built-in tool."""
        # Set image output params
        client.set_image_output_params(quality="high", size="1024x1024", output_format="png")

        mock_response = MockResponsesAPIResponse(
            output=[
                create_mock_image_generation_output(
                    result="base64encodedimagedata",
                    size="1024x1024",
                    quality="high",
                ),
            ],
            usage=MockUsage(),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        response = client.create({
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "Generate a sunset image"}],
            "built_in_tools": ["image_generation"],
        })

        # Verify tools were passed
        call_kwargs = client.client.responses.create.call_args[1]
        tools = call_kwargs.get("tools", [])
        assert any(t.get("type") == "image_generation" for t in tools)

        # Verify images extracted
        images = OpenAIResponsesV2Client.get_generated_images(response)
        assert len(images) == 1
        assert isinstance(images[0], ImageContent)
        assert images[0].data_uri.startswith("data:image/png;base64,")

    def test_create_with_apply_patch(self, client):
        """Test creating request with apply_patch built-in tool."""
        mock_response = MockResponsesAPIResponse(
            output=[
                MockOutputItem(
                    "apply_patch_call",
                    call_id="patch_123",
                    status="success",
                    operation={"type": "create_file", "path": "test.py"},
                )
            ],
            usage=MockUsage(),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        response = client.create({
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "Create test.py"}],
            "built_in_tools": ["apply_patch"],
        })

        # Verify tools were passed
        call_kwargs = client.client.responses.create.call_args[1]
        tools = call_kwargs.get("tools", [])
        assert any(t.get("type") == "apply_patch" for t in tools)

    def test_get_web_search_calls(self, client):
        """Test extracting web search call metadata."""
        mock_response = MockResponsesAPIResponse(
            output=[
                create_mock_web_search_output("search_456", "completed"),
                create_mock_message_output("Search results"),
            ],
            usage=MockUsage(),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        response = client.create({
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "Search something"}],
            "built_in_tools": ["web_search"],
        })

        search_calls = OpenAIResponsesV2Client.get_web_search_calls(response)
        assert len(search_calls) >= 1
        assert isinstance(search_calls[0], GenericContent)
        assert search_calls[0].type == "web_search_call"


# =============================================================================
# Test: Image Output Configuration
# =============================================================================


class TestImageOutputConfiguration:
    """Test image output parameter configuration."""

    def test_set_image_output_params(self, client):
        """Test setting image output parameters."""
        client.set_image_output_params(
            quality="high",
            size="1024x1536",
            output_format="png",
            background="transparent",
        )

        assert client.image_output_params["quality"] == "high"
        assert client.image_output_params["size"] == "1024x1536"
        assert client.image_output_params["output_format"] == "png"
        assert client.image_output_params["background"] == "transparent"

    def test_set_image_output_params_partial(self, client):
        """Test setting only some image output parameters."""
        original_format = client.image_output_params["output_format"]
        client.set_image_output_params(quality="low")

        assert client.image_output_params["quality"] == "low"
        assert client.image_output_params["output_format"] == original_format


# =============================================================================
# Test: Web Search Configuration
# =============================================================================


class TestWebSearchConfiguration:
    """Test web search parameter configuration."""

    def test_set_web_search_params(self, client):
        """Test setting web search parameters."""
        client.set_web_search_params(
            user_location={"country": "US", "city": "San Francisco"},
            search_context_size="high",
        )

        assert client.web_search_params["user_location"]["country"] == "US"
        assert client.web_search_params["search_context_size"] == "high"


# =============================================================================
# Test: Structured Output
# =============================================================================


class TestStructuredOutput:
    """Test structured output with Pydantic models."""

    def test_structured_output_with_pydantic_model(self, client):
        """Test structured output using Pydantic model."""
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("Pydantic not installed")

        class Person(BaseModel):
            name: str
            age: int

        # Create mock parsed object
        parsed_obj = Person(name="Alice", age=30)
        mock_response = MockResponsesAPIResponse(
            output=[create_mock_message_output('{"name": "Alice", "age": 30}')],
            usage=MockUsage(),
            output_parsed=parsed_obj,
        )
        client.client.responses.parse = Mock(return_value=mock_response)

        response = client.create({
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "Generate a person profile"}],
            "response_format": Person,
        })

        # Verify parse() was called
        assert client.client.responses.parse.called

        # Verify parsed object extracted
        parsed = OpenAIResponsesV2Client.get_parsed_object(response)
        assert parsed is not None
        assert parsed.name == "Alice"
        assert parsed.age == 30

    def test_get_parsed_dict(self, client):
        """Test getting parsed object as dictionary."""
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("Pydantic not installed")

        class Person(BaseModel):
            name: str
            age: int

        parsed_obj = Person(name="Bob", age=25)
        mock_response = MockResponsesAPIResponse(
            output=[create_mock_message_output("{}")],
            usage=MockUsage(),
            output_parsed=parsed_obj,
        )
        client.client.responses.parse = Mock(return_value=mock_response)

        response = client.create({
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "Generate"}],
            "response_format": Person,
        })

        parsed_content = OpenAIResponsesV2Client.get_parsed_content(response)
        assert parsed_content is not None
        parsed_dict = parsed_content.parsed_dict
        assert parsed_dict is not None
        assert parsed_dict["name"] == "Bob"
        assert parsed_dict["age"] == 25


# =============================================================================
# Test: Cost Tracking
# =============================================================================


class TestCostTracking:
    """Test cost tracking functionality."""

    def test_token_cost_tracking(self, client):
        """Test that token costs are tracked."""
        mock_response = MockResponsesAPIResponse(
            model="gpt-4o-mini",
            output=[create_mock_message_output("Response")],
            usage=MockUsage(input_tokens=100, output_tokens=50, total_tokens=150),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        response = client.create({"model": "gpt-4o-mini", "messages": []})

        assert response.cost >= 0
        assert response.usage.get("token_cost", 0) >= 0

    def test_cumulative_token_tracking(self, client):
        """Test cumulative token tracking across requests."""
        mock_response = MockResponsesAPIResponse(
            model="gpt-4o-mini",
            output=[create_mock_message_output("Response")],
            usage=MockUsage(input_tokens=100, output_tokens=50, total_tokens=150),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        # Make multiple requests
        client.create({"model": "gpt-4o-mini", "messages": []})
        client.create({"model": "gpt-4o-mini", "messages": []})

        cumulative = client.get_cumulative_usage()
        assert cumulative["prompt_tokens"] == 200
        assert cumulative["completion_tokens"] == 100
        assert cumulative["total_tokens"] == 300

    def test_image_cost_tracking(self, client):
        """Test image generation cost tracking."""
        mock_response = MockResponsesAPIResponse(
            output=[
                create_mock_image_generation_output(
                    result="base64data",
                    size="1024x1024",
                    quality="high",
                ),
            ],
            usage=MockUsage(),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        response = client.create({
            "model": "gpt-4.1",
            "messages": [],
            "built_in_tools": ["image_generation"],
        })

        assert client._image_costs > 0
        assert client.get_image_costs() > 0

    def test_total_cost_includes_images(self, client):
        """Test that total cost includes both tokens and images."""
        mock_response = MockResponsesAPIResponse(
            model="gpt-4o-mini",
            output=[
                create_mock_message_output("Here's your image"),
                create_mock_image_generation_output("base64data", "1024x1024", "high"),
            ],
            usage=MockUsage(input_tokens=50, output_tokens=25, total_tokens=75),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        response = client.create({"model": "gpt-4o-mini", "messages": [], "built_in_tools": ["image_generation"]})

        total_cost = client.get_total_costs()
        token_cost = client.get_token_costs()
        image_cost = client.get_image_costs()

        assert total_cost == token_cost + image_cost

    def test_reset_all_costs(self, client):
        """Test resetting all cost tracking."""
        mock_response = MockResponsesAPIResponse(
            output=[create_mock_image_generation_output("base64data", "1024x1024", "high")],
            usage=MockUsage(input_tokens=100, output_tokens=50, total_tokens=150),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        # Accumulate costs
        client.create({"model": "gpt-4.1", "messages": [], "built_in_tools": ["image_generation"]})
        assert client.get_total_costs() > 0

        # Reset
        client.reset_all_costs()

        assert client.get_total_costs() == 0
        assert client.get_token_costs() == 0
        assert client.get_image_costs() == 0
        assert client._total_prompt_tokens == 0
        assert client._total_completion_tokens == 0

    def test_set_custom_price(self, client):
        """Test setting custom pricing."""
        client.set_custom_price(input_price_per_1k=0.001, output_price_per_1k=0.002)
        assert client._custom_price == (0.001, 0.002)


# =============================================================================
# Test: Cost Calculation Functions
# =============================================================================


class TestCostCalculationFunctions:
    """Test standalone cost calculation functions."""

    def test_calculate_image_cost_valid(self):
        """Test image cost calculation with valid params."""
        cost, error = calculate_image_cost("gpt-image-1", "1024x1024", "high")
        assert error is None
        assert cost == 0.167

    def test_calculate_image_cost_invalid_model(self):
        """Test image cost calculation with invalid model."""
        cost, error = calculate_image_cost("invalid-model", "1024x1024", "high")
        assert cost == 0.0
        assert "Invalid model" in error

    def test_calculate_image_cost_invalid_size(self):
        """Test image cost calculation with invalid size."""
        cost, error = calculate_image_cost("gpt-image-1", "256x256", "high")
        assert cost == 0.0
        assert "Invalid size" in error

    def test_calculate_token_cost_known_model(self):
        """Test token cost calculation for known model."""
        cost = calculate_token_cost("gpt-4o-mini", 1000, 500)
        assert cost > 0

    def test_calculate_token_cost_unknown_model(self):
        """Test token cost calculation for unknown model."""
        cost = calculate_token_cost("unknown-model", 1000, 500)
        assert cost == 0.0

    def test_calculate_token_cost_custom_price(self):
        """Test token cost calculation with custom pricing."""
        cost = calculate_token_cost(
            "custom-model",
            prompt_tokens=1000,
            completion_tokens=500,
            custom_price=(0.01, 0.02),
        )
        # (1000 * 0.01 + 500 * 0.02) / 1000 = 0.02
        expected = (1000 * 0.01 + 500 * 0.02) / 1000
        assert cost == expected


# =============================================================================
# Test: Static Helper Methods
# =============================================================================


class TestStaticHelperMethods:
    """Test static helper methods."""

    def test_get_citations(self, client):
        """Test extracting citations from response."""
        mock_response = MockResponsesAPIResponse(
            output=[
                create_mock_message_output(
                    "News article content",
                    annotations=[
                        {
                            "type": "url_citation",
                            "url": "https://example1.com",
                            "title": "Article 1",
                            "start_index": 0,
                            "end_index": 10,
                        },
                        {
                            "type": "url_citation",
                            "url": "https://example2.com",
                            "title": "Article 2",
                            "start_index": 10,
                            "end_index": 20,
                        },
                    ],
                )
            ],
            usage=MockUsage(),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        response = client.create({"model": "gpt-4.1", "messages": [], "built_in_tools": ["web_search"]})
        citations = OpenAIResponsesV2Client.get_citations(response)

        assert len(citations) == 2
        assert all(isinstance(c, CitationContent) for c in citations)
        assert citations[0].url == "https://example1.com"
        assert citations[1].url == "https://example2.com"

    def test_get_generated_images(self, client):
        """Test extracting generated images from response."""
        mock_response = MockResponsesAPIResponse(
            output=[
                create_mock_image_generation_output("image1data", "1024x1024", "high"),
                create_mock_image_generation_output("image2data", "1024x1024", "medium"),
            ],
            usage=MockUsage(),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        response = client.create({"model": "gpt-4.1", "messages": [], "built_in_tools": ["image_generation"]})
        images = OpenAIResponsesV2Client.get_generated_images(response)

        assert len(images) == 2
        assert all(isinstance(img, ImageContent) for img in images)
        assert all(img.data_uri is not None for img in images)

    def test_get_usage_static(self, client):
        """Test static get_usage method."""
        mock_response = MockResponsesAPIResponse(
            model="gpt-4.1",
            output=[create_mock_message_output("Test")],
            usage=MockUsage(input_tokens=100, output_tokens=50, total_tokens=150),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        response = client.create({"model": "gpt-4.1", "messages": []})
        usage = OpenAIResponsesV2Client.get_usage(response)

        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150
        assert "cost" in usage
        assert "model" in usage

    def test_create_multimodal_message(self):
        """Test creating multimodal message with images."""
        message = OpenAIResponsesV2Client.create_multimodal_message(
            text="What's in these images?",
            images=["https://example.com/image1.jpg", "https://example.com/image2.jpg"],
            role="user",
        )

        assert message["role"] == "user"
        assert isinstance(message["content"], list)
        assert len(message["content"]) == 3  # 1 text + 2 images

        # Check text block (Responses API uses "input_text" type)
        text_blocks = [c for c in message["content"] if c.get("type") == "input_text"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "What's in these images?"

        # Check image blocks (Responses API uses "input_image" type)
        image_blocks = [c for c in message["content"] if c.get("type") == "input_image"]
        assert len(image_blocks) == 2


# =============================================================================
# Test: V1 Compatibility
# =============================================================================


class TestV1Compatibility:
    """Test V1 backward compatibility."""

    def test_create_v1_compatible_format(self, client):
        """Test backward compatible response format."""
        mock_response = MockResponsesAPIResponse(
            output=[create_mock_message_output("Test response")],
            usage=MockUsage(),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        response = client.create_v1_compatible({"model": "gpt-4.1", "messages": []})

        # Verify ChatCompletion-like structure
        assert hasattr(response, "id")
        assert hasattr(response, "model")
        assert hasattr(response, "choices")
        assert hasattr(response, "usage")
        assert hasattr(response, "cost")
        assert response.object == "chat.completion"

    def test_v1_compatible_content_access(self, client):
        """Test accessing content from V1 compatible response."""
        mock_response = MockResponsesAPIResponse(
            output=[create_mock_message_output("Hello world!")],
            usage=MockUsage(),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        response = client.create_v1_compatible({"model": "gpt-4.1", "messages": []})

        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None
        assert "Hello world" in response.choices[0].message.content

    def test_v1_compatible_with_tool_calls(self, client):
        """Test V1 compatible response with tool calls."""
        mock_response = MockResponsesAPIResponse(
            output=[create_mock_function_call_output("call_123", "get_weather", '{"city": "NYC"}')],
            usage=MockUsage(),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        response = client.create_v1_compatible({
            "model": "gpt-4.1",
            "messages": [],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        })

        assert response.choices[0].message.tool_calls is not None
        assert len(response.choices[0].message.tool_calls) == 1
        assert response.choices[0].message.tool_calls[0].function.name == "get_weather"
        assert response.choices[0].finish_reason == "tool_calls"

    def test_is_v1_compatible(self, client):
        """Test is_v1_compatible method."""
        assert client.is_v1_compatible() is True


# =============================================================================
# Test: Message Retrieval
# =============================================================================


class TestMessageRetrieval:
    """Test message_retrieval() method."""

    def test_message_retrieval_simple_text(self, client):
        """Test retrieving text from simple response."""
        mock_response = MockResponsesAPIResponse(
            output=[create_mock_message_output("Hello world")],
            usage=MockUsage(),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        response = client.create({"model": "gpt-4.1", "messages": []})
        messages = client.message_retrieval(response)

        assert len(messages) == 1
        assert messages[0] == "Hello world"

    def test_message_retrieval_with_tool_calls(self, client):
        """Test retrieving messages with tool calls returns dict format."""
        mock_response = MockResponsesAPIResponse(
            output=[
                create_mock_message_output("Let me check the weather"),
                create_mock_function_call_output("call_123", "get_weather", '{"city": "NYC"}'),
            ],
            usage=MockUsage(),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        response = client.create({"model": "gpt-4.1", "messages": []})
        messages = client.message_retrieval(response)

        # Should return dict format when tool calls present
        assert len(messages) >= 1
        assert isinstance(messages[0], dict)
        assert "tool_calls" in messages[0]

    def test_message_retrieval_with_images(self, client):
        """Test retrieving messages with images returns dict format."""
        mock_response = MockResponsesAPIResponse(
            output=[
                create_mock_message_output("Here's your image"),
                create_mock_image_generation_output("base64data", "1024x1024", "high"),
            ],
            usage=MockUsage(),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        response = client.create({"model": "gpt-4.1", "messages": [], "built_in_tools": ["image_generation"]})
        messages = client.message_retrieval(response)

        assert len(messages) >= 1
        assert isinstance(messages[0], dict)


# =============================================================================
# Test: Response Transformation
# =============================================================================


class TestResponseTransformation:
    """Test response transformation edge cases."""

    def test_empty_response_output(self, client):
        """Test handling empty response output."""
        mock_response = MockResponsesAPIResponse(
            output=[],
            usage=MockUsage(),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        response = client.create({"model": "gpt-4.1", "messages": []})

        # Should still have a message, even if empty
        assert len(response.messages) == 1

    def test_unknown_output_type(self, client):
        """Test handling unknown output item types."""
        mock_response = MockResponsesAPIResponse(
            output=[MockOutputItem("unknown_new_type", data="some data")],
            usage=MockUsage(),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        response = client.create({"model": "gpt-4.1", "messages": []})

        # Unknown types should be preserved as GenericContent
        generic_blocks = [b for msg in response.messages for b in msg.content if isinstance(b, GenericContent)]
        assert len(generic_blocks) >= 1
        assert generic_blocks[0].type == "unknown_new_type"

    def test_multiple_output_items(self, client):
        """Test handling multiple output items of different types."""
        mock_response = MockResponsesAPIResponse(
            output=[
                create_mock_reasoning_output("Thinking..."),
                create_mock_message_output("Final answer"),
                create_mock_function_call_output("call_1", "tool", "{}"),
            ],
            usage=MockUsage(),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        response = client.create({"model": "gpt-4.1", "messages": []})

        # All content types should be captured
        content = response.messages[0].content
        assert any(isinstance(b, ReasoningContent) for b in content)
        assert any(isinstance(b, TextContent) for b in content)
        assert any(isinstance(b, ToolCallContent) for b in content)


# =============================================================================
# Test: Protocol Compliance
# =============================================================================


class TestProtocolCompliance:
    """Test ModelClient protocol compliance."""

    def test_response_usage_keys(self, client):
        """Test RESPONSE_USAGE_KEYS is defined."""
        assert hasattr(client, "RESPONSE_USAGE_KEYS")
        assert "prompt_tokens" in client.RESPONSE_USAGE_KEYS
        assert "completion_tokens" in client.RESPONSE_USAGE_KEYS
        assert "total_tokens" in client.RESPONSE_USAGE_KEYS
        assert "cost" in client.RESPONSE_USAGE_KEYS
        assert "model" in client.RESPONSE_USAGE_KEYS

    def test_cost_method(self, client):
        """Test cost() method signature."""
        mock_response = MockResponsesAPIResponse(
            output=[create_mock_message_output("Test")],
            usage=MockUsage(),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        response = client.create({"model": "gpt-4.1", "messages": []})
        cost = client.cost(response)

        assert isinstance(cost, float)
        assert cost >= 0


# =============================================================================
# Test: Error Handling
# =============================================================================


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_api_error_propagation(self, client):
        """Test that API errors are propagated."""
        client.client.responses.create = Mock(side_effect=Exception("API Error"))

        with pytest.raises(Exception, match="API Error"):
            client.create({"model": "gpt-4.1", "messages": []})

    def test_parse_error_fallback(self, client):
        """Test fallback when parse() fails with unexpected error."""
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("Pydantic not installed")

        class TestModel(BaseModel):
            field: str

        # Make parse() fail with TypeError for text_format
        def side_effect(**kwargs):
            if "text_format" in kwargs:
                raise TypeError("text_format is an unexpected keyword")
            return MockResponsesAPIResponse(
                output=[create_mock_message_output("Fallback response")],
                usage=MockUsage(),
            )

        client.client.responses.parse = Mock(side_effect=side_effect)
        client.client.responses.create = Mock(
            return_value=MockResponsesAPIResponse(
                output=[create_mock_message_output("Fallback response")],
                usage=MockUsage(),
            )
        )

        # Should warn and fallback to create()
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            response = client.create({
                "model": "gpt-4.1",
                "messages": [],
                "response_format": TestModel,
            })
            # Check warning was issued
            assert len(w) >= 1
            assert "response_format" in str(w[0].message).lower()


# =============================================================================
# Test: Integration Scenarios
# =============================================================================


class TestIntegrationScenarios:
    """Integration tests for complete workflows."""

    def test_full_workflow_with_reasoning(self, client):
        """Test complete workflow with reasoning blocks."""
        mock_response = MockResponsesAPIResponse(
            model="o3",
            output=[
                create_mock_reasoning_output(
                    "Step 1: Analyze the problem\nStep 2: Calculate\nStep 3: Verify",
                    summary="Mathematical analysis complete",
                ),
                create_mock_message_output("The answer is 42"),
            ],
            usage=MockUsage(
                input_tokens=100,
                output_tokens=200,
                total_tokens=300,
                output_tokens_details={"reasoning_tokens": 150},
            ),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        response = client.create({
            "model": "o3",
            "messages": [{"role": "user", "content": "What is the meaning of life?"}],
        })

        # Verify response structure
        assert isinstance(response, UnifiedResponse)
        assert response.model == "o3"
        assert response.provider == "openai_responses"

        # Check reasoning
        assert len(response.reasoning) >= 1
        assert "Step 1" in response.reasoning[0].reasoning

        # Check text
        assert "42" in response.text

        # Check usage including reasoning tokens
        usage = client.get_usage(response)
        assert usage["reasoning_tokens"] == 150

        # Check cost
        assert response.cost >= 0

    def test_multi_turn_conversation_workflow(self, client):
        """Test multi-turn conversation with state management."""
        # First turn
        mock_response1 = MockResponsesAPIResponse(
            response_id="resp_turn1",
            output=[create_mock_message_output("I'll remember your name is Alice.")],
            usage=MockUsage(),
        )
        client.client.responses.create = Mock(return_value=mock_response1)

        response1 = client.create({
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "My name is Alice"}],
        })

        # Verify state updated
        assert client._get_previous_response_id() == "resp_turn1"

        # Second turn
        mock_response2 = MockResponsesAPIResponse(
            response_id="resp_turn2",
            output=[create_mock_message_output("Your name is Alice!")],
            usage=MockUsage(),
        )
        client.client.responses.create = Mock(return_value=mock_response2)

        response2 = client.create({
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "What's my name?"}],
        })

        # Verify previous_response_id was used
        call_kwargs = client.client.responses.create.call_args[1]
        assert call_kwargs.get("previous_response_id") == "resp_turn1"

        # Verify state updated again
        assert client._get_previous_response_id() == "resp_turn2"

    def test_web_search_with_citations_workflow(self, client):
        """Test web search workflow with citation extraction."""
        mock_response = MockResponsesAPIResponse(
            output=[
                create_mock_web_search_output("search_abc", "completed"),
                create_mock_message_output(
                    "Based on recent research, AI is advancing rapidly.",
                    annotations=[
                        {
                            "type": "url_citation",
                            "url": "https://research.example.com/ai-2024",
                            "title": "AI Research 2024",
                            "start_index": 0,
                            "end_index": 20,
                        },
                        {
                            "type": "url_citation",
                            "url": "https://news.example.com/ai-breakthrough",
                            "title": "AI Breakthrough",
                            "start_index": 21,
                            "end_index": 40,
                        },
                    ],
                ),
            ],
            usage=MockUsage(),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        response = client.create({
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "What's happening in AI?"}],
            "built_in_tools": ["web_search"],
        })

        # Extract citations
        citations = OpenAIResponsesV2Client.get_citations(response)
        assert len(citations) == 2
        assert citations[0].title == "AI Research 2024"
        assert citations[1].title == "AI Breakthrough"

        # Extract search calls
        search_calls = OpenAIResponsesV2Client.get_web_search_calls(response)
        assert len(search_calls) >= 1


# =============================================================================
# Test: Apply Patch Operations
# =============================================================================


class TestApplyPatchOperation:
    """Test _apply_patch_operation method."""

    def test_apply_patch_invalid_operation_type(self, client):
        """Test that invalid operation type returns failed status."""
        result = client._apply_patch_operation(
            operation={"type": "invalid_op", "path": "test.py"},
            call_id="call_123",
        )

        assert result.call_id == "call_123"
        assert result.status == "failed"
        assert "Invalid operation type" in result.output
        assert "invalid_op" in result.output

    def test_apply_patch_create_file_sync(self, client):
        """Test synchronous create_file operation."""
        with patch("autogen.tools.experimental.apply_patch.apply_patch_tool.WorkspaceEditor") as MockEditor:
            mock_editor = Mock()
            mock_editor.create_file.return_value = {"status": "success", "output": "File created"}
            MockEditor.return_value = mock_editor

            result = client._apply_patch_operation(
                operation={"type": "create_file", "path": "test.py", "diff": "print('hello')"},
                call_id="call_create",
                async_patches=False,
            )

            assert result.call_id == "call_create"
            assert result.status == "success"
            assert result.output == "File created"
            mock_editor.create_file.assert_called_once()

    def test_apply_patch_update_file_sync(self, client):
        """Test synchronous update_file operation."""
        with patch("autogen.tools.experimental.apply_patch.apply_patch_tool.WorkspaceEditor") as MockEditor:
            mock_editor = Mock()
            mock_editor.update_file.return_value = {"status": "success", "output": "File updated"}
            MockEditor.return_value = mock_editor

            result = client._apply_patch_operation(
                operation={"type": "update_file", "path": "test.py", "diff": "..."},
                call_id="call_update",
                async_patches=False,
            )

            assert result.call_id == "call_update"
            assert result.status == "success"
            assert result.output == "File updated"
            mock_editor.update_file.assert_called_once()

    def test_apply_patch_delete_file_sync(self, client):
        """Test synchronous delete_file operation."""
        with patch("autogen.tools.experimental.apply_patch.apply_patch_tool.WorkspaceEditor") as MockEditor:
            mock_editor = Mock()
            mock_editor.delete_file.return_value = {"status": "success", "output": "File deleted"}
            MockEditor.return_value = mock_editor

            result = client._apply_patch_operation(
                operation={"type": "delete_file", "path": "test.py"},
                call_id="call_delete",
                async_patches=False,
            )

            assert result.call_id == "call_delete"
            assert result.status == "success"
            assert result.output == "File deleted"
            mock_editor.delete_file.assert_called_once()

    def test_apply_patch_exception_handling(self, client):
        """Test exception handling in apply_patch_operation."""
        with patch("autogen.tools.experimental.apply_patch.apply_patch_tool.WorkspaceEditor") as MockEditor:
            mock_editor = Mock()
            mock_editor.create_file.side_effect = Exception("Disk full")
            MockEditor.return_value = mock_editor

            result = client._apply_patch_operation(
                operation={"type": "create_file", "path": "test.py", "diff": "..."},
                call_id="call_error",
                async_patches=False,
            )

            assert result.call_id == "call_error"
            assert result.status == "failed"
            assert "Error applying patch" in result.output
            assert "Disk full" in result.output

    def test_apply_patch_uses_instance_defaults(self, mock_openai_client):
        """Test that apply_patch uses instance workspace_dir and allowed_paths."""
        client = OpenAIResponsesV2Client(
            api_key="test-key",
            workspace_dir="/custom/workspace",
            allowed_paths=["*.py", "*.txt"],
        )

        with patch("autogen.tools.experimental.apply_patch.apply_patch_tool.WorkspaceEditor") as MockEditor:
            mock_editor = Mock()
            mock_editor.create_file.return_value = {"status": "success", "output": "Done"}
            MockEditor.return_value = mock_editor

            client._apply_patch_operation(
                operation={"type": "create_file", "path": "test.py", "diff": "..."},
                call_id="call_123",
            )

            # Verify WorkspaceEditor was created with instance defaults
            MockEditor.assert_called_once_with(
                workspace_dir="/custom/workspace",
                allowed_paths=["*.py", "*.txt"],
            )

    def test_apply_patch_overrides_instance_defaults(self, client):
        """Test that explicit params override instance defaults."""
        with patch("autogen.tools.experimental.apply_patch.apply_patch_tool.WorkspaceEditor") as MockEditor:
            mock_editor = Mock()
            mock_editor.create_file.return_value = {"status": "success", "output": "Done"}
            MockEditor.return_value = mock_editor

            client._apply_patch_operation(
                operation={"type": "create_file", "path": "test.py", "diff": "..."},
                call_id="call_123",
                workspace_dir="/override/workspace",
                allowed_paths=["**/*.md"],
            )

            MockEditor.assert_called_once_with(
                workspace_dir="/override/workspace",
                allowed_paths=["**/*.md"],
            )

    def test_apply_patch_async_no_running_loop(self, client):
        """Test async execution when no event loop is running."""
        with patch("autogen.tools.experimental.apply_patch.apply_patch_tool.WorkspaceEditor") as MockEditor:
            mock_editor = Mock()

            # Create an async mock

            async def async_create_file(op):
                return {"status": "success", "output": "Async created"}

            mock_editor.a_create_file = Mock(side_effect=lambda op: async_create_file(op))
            MockEditor.return_value = mock_editor

            with patch("asyncio.get_running_loop", side_effect=RuntimeError("No running loop")):
                with patch("asyncio.run") as mock_run:
                    mock_run.return_value = {"status": "success", "output": "Async created"}

                    result = client._apply_patch_operation(
                        operation={"type": "create_file", "path": "test.py", "diff": "..."},
                        call_id="call_async",
                        async_patches=True,
                    )

                    assert result.status == "success"
                    mock_run.assert_called_once()


class TestApplyPatchCallOutput:
    """Test ApplyPatchCallOutput dataclass."""

    def test_apply_patch_call_output_creation(self):
        """Test creating ApplyPatchCallOutput."""
        from autogen.llm_clients.openai_responses_v2 import ApplyPatchCallOutput

        output = ApplyPatchCallOutput(
            call_id="call_123",
            status="success",
            output="Operation completed",
        )

        assert output.call_id == "call_123"
        assert output.status == "success"
        assert output.output == "Operation completed"
        assert output.type == "apply_patch_call_output"

    def test_apply_patch_call_output_to_dict(self):
        """Test converting ApplyPatchCallOutput to dict."""
        from autogen.llm_clients.openai_responses_v2 import ApplyPatchCallOutput

        output = ApplyPatchCallOutput(
            call_id="call_456",
            status="failed",
            output="Permission denied",
        )

        result = output.to_dict()

        assert result == {
            "call_id": "call_456",
            "status": "failed",
            "output": "Permission denied",
            "type": "apply_patch_call_output",
        }


# =============================================================================
# Test: Shell Tool Operations
# =============================================================================


class TestShellToolDataclasses:
    """Test shell tool dataclasses."""

    def test_shell_call_outcome_creation(self):
        """Test creating ShellCallOutcome."""
        from autogen.llm_clients.openai_responses_v2 import ShellCallOutcome

        outcome = ShellCallOutcome(type="exit", exit_code=0)

        assert outcome.type == "exit"
        assert outcome.exit_code == 0

    def test_shell_call_outcome_timeout(self):
        """Test ShellCallOutcome with timeout."""
        from autogen.llm_clients.openai_responses_v2 import ShellCallOutcome

        outcome = ShellCallOutcome(type="timeout", exit_code=None)

        assert outcome.type == "timeout"
        assert outcome.exit_code is None

    def test_shell_call_outcome_to_dict(self):
        """Test converting ShellCallOutcome to dict."""
        from autogen.llm_clients.openai_responses_v2 import ShellCallOutcome

        outcome = ShellCallOutcome(type="exit", exit_code=1)
        result = outcome.to_dict()

        assert result == {"type": "exit", "exit_code": 1}

    def test_shell_command_output_creation(self):
        """Test creating ShellCommandOutput."""
        from autogen.llm_clients.openai_responses_v2 import (
            ShellCallOutcome,
            ShellCommandOutput,
        )

        outcome = ShellCallOutcome(type="exit", exit_code=0)
        output = ShellCommandOutput(
            stdout="file1.txt\nfile2.txt",
            stderr="",
            outcome=outcome,
        )

        assert output.stdout == "file1.txt\nfile2.txt"
        assert output.stderr == ""
        assert output.outcome.exit_code == 0

    def test_shell_command_output_to_dict(self):
        """Test converting ShellCommandOutput to dict."""
        from autogen.llm_clients.openai_responses_v2 import (
            ShellCallOutcome,
            ShellCommandOutput,
        )

        outcome = ShellCallOutcome(type="exit", exit_code=0)
        output = ShellCommandOutput(stdout="hello", stderr="", outcome=outcome)
        result = output.to_dict()

        assert result["stdout"] == "hello"
        assert result["stderr"] == ""
        assert result["outcome"]["type"] == "exit"
        assert result["outcome"]["exit_code"] == 0

    def test_shell_call_output_creation(self):
        """Test creating ShellCallOutput."""
        from autogen.llm_clients.openai_responses_v2 import (
            ShellCallOutcome,
            ShellCallOutput,
            ShellCommandOutput,
        )

        outcome = ShellCallOutcome(type="exit", exit_code=0)
        cmd_output = ShellCommandOutput(stdout="output", stderr="", outcome=outcome)
        shell_output = ShellCallOutput(
            call_id="call_shell_123",
            output=[cmd_output],
        )

        assert shell_output.call_id == "call_shell_123"
        assert shell_output.type == "shell_call_output"
        assert len(shell_output.output) == 1

    def test_shell_call_output_to_dict(self):
        """Test converting ShellCallOutput to dict."""
        from autogen.llm_clients.openai_responses_v2 import (
            ShellCallOutcome,
            ShellCallOutput,
            ShellCommandOutput,
        )

        outcome = ShellCallOutcome(type="exit", exit_code=0)
        cmd_output = ShellCommandOutput(stdout="hello world", stderr="", outcome=outcome)
        shell_output = ShellCallOutput(
            call_id="call_456",
            max_output_length=1000,
            output=[cmd_output],
        )

        result = shell_output.to_dict()

        assert result["call_id"] == "call_456"
        assert result["type"] == "shell_call_output"
        assert result["max_output_length"] == 1000
        assert len(result["output"]) == 1
        assert result["output"][0]["stdout"] == "hello world"


class TestShellToolOperation:
    """Test shell tool operation methods."""

    def test_extract_shell_calls_from_messages(self, client):
        """Test extracting shell_call items from messages."""
        messages = [
            {
                "role": "user",
                "content": "List files",
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "shell_call", "call_id": "shell_123", "action": {"commands": ["ls -la"]}},
                    {"type": "text", "text": "Running command..."},
                ],
            },
        ]

        result = client._extract_shell_calls(messages)

        assert "shell_123" in result
        assert result["shell_123"]["type"] == "shell_call"
        assert result["shell_123"]["action"]["commands"] == ["ls -la"]

    def test_extract_shell_calls_from_tool_calls(self, client):
        """Test extracting shell_call items from tool_calls field."""
        messages = [
            {
                "role": "assistant",
                "content": "Let me run that command.",
                "tool_calls": [
                    {"type": "shell_call", "call_id": "shell_456", "action": {"commands": ["pwd"]}},
                ],
            },
        ]

        result = client._extract_shell_calls(messages)

        assert "shell_456" in result
        assert result["shell_456"]["action"]["commands"] == ["pwd"]

    def test_extract_shell_calls_empty(self, client):
        """Test extracting shell_call items when none present."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = client._extract_shell_calls(messages)

        assert result == {}

    def test_execute_shell_operation_no_commands(self, client):
        """Test shell operation with no commands."""
        result = client._execute_shell_operation(
            action={},  # No commands
            call_id="call_empty",
        )

        assert result.call_id == "call_empty"
        assert len(result.output) == 1
        assert "No commands provided" in result.output[0].stderr
        assert result.output[0].outcome.exit_code == 1

    def test_execute_shell_calls_not_in_tools(self, client):
        """Test that shell calls are not executed when shell not in built_in_tools."""
        calls_dict = {"shell_123": {"action": {"commands": ["ls"]}}}

        result = client._execute_shell_calls(
            calls_dict=calls_dict,
            built_in_tools=["web_search"],  # shell not included
            workspace_dir="/tmp",
            allowed_paths=["**"],
        )

        assert result == []

    def test_execute_shell_operation_with_mock(self, client):
        """Test shell operation with mocked ShellExecutor."""
        with patch("autogen.tools.experimental.shell.shell_tool.ShellExecutor") as MockExecutor:
            mock_executor = Mock()
            mock_executor.run_commands.return_value = [
                {"stdout": "file1.txt", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}
            ]
            MockExecutor.return_value = mock_executor

            # Reset executor to force re-creation
            client._shell_executor = None

            result = client._execute_shell_operation(
                action={"commands": ["ls"], "timeout_ms": 5000},
                call_id="call_ls",
            )

            assert result.call_id == "call_ls"
            mock_executor.run_commands.assert_called_once_with(["ls"], timeout_ms=5000)

    def test_execute_shell_operation_exception_handling(self, client):
        """Test shell operation exception handling."""
        with patch("autogen.tools.experimental.shell.shell_tool.ShellExecutor") as MockExecutor:
            mock_executor = Mock()
            mock_executor.run_commands.side_effect = Exception("Command failed")
            MockExecutor.return_value = mock_executor

            # Reset executor
            client._shell_executor = None

            result = client._execute_shell_operation(
                action={"commands": ["bad_command"]},
                call_id="call_error",
            )

            assert result.call_id == "call_error"
            assert result.output[0].outcome.exit_code == 1
            assert "Error executing shell commands" in result.output[0].stderr


class TestShellToolConfiguration:
    """Test shell tool configuration."""

    def test_set_shell_params(self, client):
        """Test setting shell parameters."""
        client.set_shell_params(
            allowed_commands=["ls", "cat", "grep"],
            denied_commands=["rm", "sudo"],
            enable_command_filtering=True,
            dangerous_patterns=[("rm -rf", "Dangerous recursive delete")],
        )

        assert client._shell_allowed_commands == ["ls", "cat", "grep"]
        assert client._shell_denied_commands == ["rm", "sudo"]
        assert client._shell_enable_command_filtering is True
        assert client._shell_dangerous_patterns == [("rm -rf", "Dangerous recursive delete")]
        # Executor should be reset
        assert client._shell_executor is None

    def test_set_shell_params_partial(self, client):
        """Test setting only some shell parameters."""
        original_filtering = client._shell_enable_command_filtering

        client.set_shell_params(allowed_commands=["echo"])

        assert client._shell_allowed_commands == ["echo"]
        # Other params should be unchanged
        assert client._shell_enable_command_filtering == original_filtering

    def test_initial_shell_config(self, mock_openai_client):
        """Test initial shell configuration in client."""
        client = OpenAIResponsesV2Client(api_key="test-key")

        assert client._shell_allowed_commands is None
        assert client._shell_denied_commands is None
        assert client._shell_enable_command_filtering is True
        assert client._shell_dangerous_patterns is None
        assert client._shell_executor is None


class TestShellToolInCreate:
    """Test shell tool integration in create() method."""

    def test_create_with_shell_tool(self, client):
        """Test creating request with shell built-in tool."""
        mock_response = MockResponsesAPIResponse(
            output=[
                MockOutputItem(
                    "shell_call",
                    call_id="shell_123",
                    action={"commands": ["ls -la"]},
                    status="completed",
                ),
                create_mock_message_output("Here are the files..."),
            ],
            usage=MockUsage(),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        client.create({
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "List files"}],
            "built_in_tools": ["shell"],
        })

        # Verify tools were passed
        call_kwargs = client.client.responses.create.call_args[1]
        tools = call_kwargs.get("tools", [])
        assert any(t.get("type") == "shell" for t in tools)

    def test_get_shell_calls_static_method(self, client):
        """Test get_shell_calls static method."""
        mock_response = MockResponsesAPIResponse(
            output=[
                MockOutputItem(
                    "shell_call",
                    call_id="shell_abc",
                    action={"commands": ["pwd"]},
                    status="completed",
                ),
            ],
            usage=MockUsage(),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        response = client.create({
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "Show directory"}],
            "built_in_tools": ["shell"],
        })

        shell_calls = OpenAIResponsesV2Client.get_shell_calls(response)

        assert len(shell_calls) >= 1
        assert shell_calls[0].type == "shell_call"

    def test_create_with_shell_config_params(self, client):
        """Test create with shell configuration parameters."""
        mock_response = MockResponsesAPIResponse(
            output=[create_mock_message_output("Done")],
            usage=MockUsage(),
        )
        client.client.responses.create = Mock(return_value=mock_response)

        client.create({
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "Run command"}],
            "built_in_tools": ["shell"],
            "allowed_commands": ["ls", "cat"],
            "denied_commands": ["rm"],
            "enable_command_filtering": True,
        })

        # Verify shell config params were removed from API call
        call_kwargs = client.client.responses.create.call_args[1]
        assert "allowed_commands" not in call_kwargs
        assert "denied_commands" not in call_kwargs
        assert "enable_command_filtering" not in call_kwargs
