# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for BedrockV2Client with real API calls.

These tests require:
- AWS credentials configured (via environment variables, IAM role, or AWS credentials file)
- AWS_REGION environment variable set
- Bedrock model access
- pytest markers: @pytest.mark.integration

Run with:
    pytest test/llm_clients/test_bedrock_v2_integration.py -m integration
"""

import os
from pathlib import Path

import pytest
from pydantic import BaseModel

from autogen import ConversableAgent, GroupChat, GroupChatManager, LLMConfig
from autogen.import_utils import run_for_optional_imports
from autogen.llm_clients.bedrock_v2 import BedrockV2Client
from autogen.llm_clients.models import TextContent, UnifiedResponse, UserRoleEnum


@pytest.fixture(scope="class")
def bedrock_v2_config():
    """Create Bedrock V2 LLM config from environment."""
    try:
        import dotenv

        env_file = Path(__file__).parent.parent.parent / ".env"
        if env_file.exists():
            dotenv.load_dotenv(env_file)
    except ImportError:
        pass

    aws_region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if not aws_region:
        pytest.skip("AWS_REGION environment variable not set")

    model = os.getenv("BEDROCK_MODEL", "qwen.qwen3-coder-480b-a35b-v1:0")

    return {
        "api_type": "bedrock_v2",
        "model": model,
        "aws_region": aws_region,
        "aws_access_key": os.getenv("AWS_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "aws_profile_name": os.getenv("AWS_PROFILE"),
    }


@pytest.fixture(scope="class")
def bedrock_v2_client(bedrock_v2_config):
    """Create BedrockV2Client instance."""
    return BedrockV2Client(
        aws_region=bedrock_v2_config["aws_region"],
        aws_access_key=bedrock_v2_config["aws_access_key"],
        aws_secret_key=bedrock_v2_config["aws_secret_key"],
        aws_profile_name=bedrock_v2_config["aws_profile_name"],
    )


@pytest.mark.integration
@run_for_optional_imports(["boto3", "botocore"], "bedrock")
class TestBedrockV2ClientBasicUsage:
    """Test basic Bedrock V2 client usage."""

    def test_direct_client_usage(self, bedrock_v2_client, bedrock_v2_config):
        """Test direct client usage with UnifiedResponse."""
        response = bedrock_v2_client.create({
            "model": bedrock_v2_config["model"],
            "messages": [{"role": "user", "content": "Say 'Hello' in one word."}],
        })

        # Verify UnifiedResponse structure
        assert isinstance(response, UnifiedResponse), f"Expected UnifiedResponse, got {type(response)}"
        assert response.provider == "bedrock"
        assert response.model == bedrock_v2_config["model"]
        assert response.id is not None
        assert len(response.id) > 0
        assert response.status == "completed"
        assert response.finish_reason in ["stop", "length", "tool_calls", "content_filter"]

        # Verify messages structure
        assert len(response.messages) == 1, f"Expected 1 message, got {len(response.messages)}"
        assert response.messages[0].role == UserRoleEnum.ASSISTANT
        assert len(response.messages[0].content) > 0

        # Verify text content
        assert len(response.text) > 0, "Response text should not be empty"
        assert "hello" in response.text.lower(), f"Expected 'hello' in response, got: {response.text}"

        # Verify usage tracking
        assert response.usage is not None
        assert response.usage["prompt_tokens"] > 0
        assert response.usage["completion_tokens"] > 0
        assert response.usage["total_tokens"] == response.usage["prompt_tokens"] + response.usage["completion_tokens"]
        assert response.usage["total_tokens"] > 0

        # Verify cost calculation
        assert response.cost is not None
        assert response.cost >= 0

    def test_content_blocks_access(self, bedrock_v2_client, bedrock_v2_config):
        """Test accessing content blocks from response."""
        response = bedrock_v2_client.create({
            "model": bedrock_v2_config["model"],
            "messages": [{"role": "user", "content": "List 3 benefits of cloud computing."}],
        })

        # Verify response structure
        assert len(response.messages) == 1
        message = response.messages[0]
        assert len(message.content) > 0

        # Verify text blocks
        text_blocks = response.get_content_by_type("text")
        assert len(text_blocks) > 0, "Should have at least one text block"
        assert all(isinstance(block, TextContent) for block in text_blocks)
        assert all(len(block.text) > 0 for block in text_blocks)

        # Verify text property aggregates correctly
        assert len(response.text) > 0
        assert len(response.text) >= sum(len(block.text) for block in text_blocks)

        # Verify content blocks are accessible via message
        message_text_blocks = [b for b in message.content if b.type == "text"]
        assert len(message_text_blocks) > 0

    def test_usage_and_cost_tracking(self, bedrock_v2_client, bedrock_v2_config):
        """Test usage and cost tracking."""
        response = bedrock_v2_client.create({
            "model": bedrock_v2_config["model"],
            "messages": [{"role": "user", "content": "Count from 1 to 3."}],
        })

        # Verify usage structure
        usage = bedrock_v2_client.get_usage(response)
        assert isinstance(usage, dict)

        # Verify all required keys present
        for key in bedrock_v2_client.RESPONSE_USAGE_KEYS:
            assert key in usage, f"Missing required usage key: {key}"

        # Verify usage values
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        assert usage["model"] == bedrock_v2_config["model"]
        assert isinstance(usage["cost"], (int, float))
        assert usage["cost"] >= 0

        # Verify cost matches response.cost
        assert usage["cost"] == response.cost

        # Verify cost calculation method
        calculated_cost = bedrock_v2_client.cost(response)
        assert calculated_cost == response.cost
        assert calculated_cost >= 0


@pytest.mark.integration
@run_for_optional_imports(["boto3", "botocore"], "bedrock")
class TestBedrockV2ClientStructuredOutputs:
    """Test structured outputs with Bedrock V2."""

    def test_pydantic_structured_output(self, bedrock_v2_config):
        """Test structured output with Pydantic model."""

        class Answer(BaseModel):
            answer: str
            confidence: float

        llm_config = LLMConfig(
            config_list=[
                {
                    **bedrock_v2_config,
                    "response_format": Answer,
                }
            ],
        )

        agent = ConversableAgent(
            name="test_agent",
            llm_config=llm_config,
            max_consecutive_auto_reply=1,
            human_input_mode="NEVER",
        )

        result = agent.run(message="What is 2+2? Answer with confidence 0.95.", max_turns=1).process()

        # Verify result structure
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

        # Verify structured output was used (should contain answer and confidence)
        result_lower = result.lower()
        assert "4" in result_lower or "answer" in result_lower or "confidence" in result_lower
        assert any(char.isdigit() for char in result), "Result should contain numeric answer"

    def test_structured_output_with_agent(self, bedrock_v2_config):
        """Test agent with structured outputs."""

        class Step(BaseModel):
            explanation: str
            output: str

        class ProblemSolution(BaseModel):
            problem: str
            steps: list[Step]
            final_answer: str

        llm_config = LLMConfig(
            config_list=[
                {
                    **bedrock_v2_config,
                    "response_format": ProblemSolution,
                }
            ],
        )

        agent = ConversableAgent(
            name="math_agent",
            llm_config=llm_config,
            system_message="Solve problems step by step.",
            max_consecutive_auto_reply=1,
            human_input_mode="NEVER",
        )

        result = agent.run(message="Solve: 3x + 7 = 22", max_turns=1).process()

        # Verify result
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

        # Verify structured output format (should contain problem, steps, final_answer)
        result_lower = result.lower()
        assert "22" in result or "x" in result_lower or "5" in result or "solve" in result_lower


@pytest.mark.integration
@run_for_optional_imports(["boto3", "botocore"], "bedrock")
class TestBedrockV2ClientV1Compatibility:
    """Test V1 vs V2 client compatibility."""

    def test_v1_v2_comparison(self, bedrock_v2_config):
        """Test V1 and V2 clients work with same interface."""
        llm_config_v2 = LLMConfig(config_list=[bedrock_v2_config])
        llm_config_v1 = LLMConfig(
            config_list=[
                {
                    **bedrock_v2_config,
                    "api_type": "bedrock",
                }
            ],
        )

        agent_v2 = ConversableAgent(
            name="agent_v2",
            llm_config=llm_config_v2,
            max_consecutive_auto_reply=1,
            human_input_mode="NEVER",
        )

        agent_v1 = ConversableAgent(
            name="agent_v1",
            llm_config=llm_config_v1,
            max_consecutive_auto_reply=1,
            human_input_mode="NEVER",
        )

        question = "What is 5+5?"
        result_v2 = agent_v2.run(message=question, max_turns=1).process()
        result_v1 = agent_v1.run(message=question, max_turns=1).process()

        # Verify both results are valid
        assert result_v2 is not None
        assert result_v1 is not None
        assert isinstance(result_v2, str)
        assert isinstance(result_v1, str)
        assert len(result_v2) > 0
        assert len(result_v1) > 0

        # Verify both contain correct answer
        result_v2_lower = result_v2.lower()
        result_v1_lower = result_v1.lower()
        assert "10" in result_v2_lower or "10" in result_v1_lower, (
            f"Expected '10' in at least one result. V2: {result_v2[:100]}, V1: {result_v1[:100]}"
        )

    def test_v1_compatible_format(self, bedrock_v2_client, bedrock_v2_config):
        """Test create_v1_compatible returns correct format."""
        v1_response = bedrock_v2_client.create_v1_compatible({
            "model": bedrock_v2_config["model"],
            "messages": [{"role": "user", "content": "Say 'test'"}],
        })

        # Verify v1 format structure
        assert isinstance(v1_response, dict)
        assert "id" in v1_response
        assert "model" in v1_response
        assert "created" in v1_response
        assert "object" in v1_response
        assert "choices" in v1_response
        assert "usage" in v1_response
        assert "cost" in v1_response

        # Verify object type
        assert v1_response["object"] == "chat.completion"

        # Verify choices structure
        assert isinstance(v1_response["choices"], list)
        assert len(v1_response["choices"]) > 0
        choice = v1_response["choices"][0]
        assert "index" in choice
        assert "message" in choice
        assert "finish_reason" in choice

        # Verify message structure
        message = choice["message"]
        assert "role" in message
        assert "content" in message
        assert message["role"] == "assistant"
        assert isinstance(message["content"], str)
        assert len(message["content"]) > 0

        # Verify usage structure
        assert isinstance(v1_response["usage"], dict)
        assert "prompt_tokens" in v1_response["usage"]
        assert "completion_tokens" in v1_response["usage"]
        assert "total_tokens" in v1_response["usage"]
        assert v1_response["usage"]["total_tokens"] > 0

        # Verify cost
        assert isinstance(v1_response["cost"], (int, float))
        assert v1_response["cost"] >= 0


@pytest.mark.integration
@run_for_optional_imports(["boto3", "botocore"], "bedrock")
class TestBedrockV2ClientGroupChat:
    """Test group chat with Bedrock V2."""

    def test_group_chat_mixed_v1_v2(self, bedrock_v2_config):
        """Test group chat with mixed V1/V2 clients."""
        planner = ConversableAgent(
            name="planner",
            llm_config=LLMConfig(config_list=[bedrock_v2_config]),
            system_message="Create plans. Say DONE when finished.",
            max_consecutive_auto_reply=2,
            human_input_mode="NEVER",
        )

        reviewer = ConversableAgent(
            name="reviewer",
            llm_config=LLMConfig(
                config_list=[
                    {
                        **bedrock_v2_config,
                        "api_type": "bedrock",
                    }
                ]
            ),
            system_message="Review plans. Say DONE when finished.",
            max_consecutive_auto_reply=2,
            human_input_mode="NEVER",
        )

        groupchat = GroupChat(
            agents=[planner, reviewer],
            messages=[],
            speaker_selection_method="auto",
        )

        manager = GroupChatManager(
            name="manager",
            groupchat=groupchat,
            llm_config=LLMConfig(config_list=[bedrock_v2_config]),
            is_termination_msg=lambda x: "DONE" in (x.get("content", "") or "").upper(),
        )

        result = planner.initiate_chat(
            recipient=manager,
            message="Create a plan for organizing a small event.",
            max_turns=3,
        )

        # Verify result structure
        assert result is not None
        assert hasattr(result, "chat_history")
        assert hasattr(result, "cost")
        assert hasattr(result, "summary")

        # Verify chat history
        assert isinstance(result.chat_history, list)
        assert len(result.chat_history) > 0, "Chat history should not be empty"

        # Verify messages have required structure
        for msg in result.chat_history:
            assert isinstance(msg, dict)
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ["user", "assistant", "system"]
            assert isinstance(msg["content"], str)

        # Verify cost tracking
        assert isinstance(result.cost, (int, float))
        assert result.cost >= 0

    def test_group_chat_with_structured_outputs(self, bedrock_v2_config):
        """Test group chat with structured outputs."""

        class TaskDetails(BaseModel):
            task_type: str
            description: str

        class RoutingDecision(BaseModel):
            selected_agent: str
            routing_reason: str
            task_details: TaskDetails

        orchestrator_config = LLMConfig(
            config_list=[
                {
                    **bedrock_v2_config,
                    "response_format": RoutingDecision,
                }
            ],
        )

        regular_config = LLMConfig(config_list=[bedrock_v2_config])

        orchestrator = ConversableAgent(
            name="orchestrator",
            llm_config=orchestrator_config,
            system_message="Route tasks to appropriate agents. Say DONE when finished.",
            max_consecutive_auto_reply=2,
            human_input_mode="NEVER",
        )

        worker = ConversableAgent(
            name="worker",
            llm_config=regular_config,
            system_message="Complete assigned tasks. Say DONE when finished.",
            max_consecutive_auto_reply=2,
            human_input_mode="NEVER",
        )

        groupchat = GroupChat(
            agents=[orchestrator, worker],
            messages=[],
            speaker_selection_method="auto",
        )

        manager = GroupChatManager(
            name="manager",
            groupchat=groupchat,
            llm_config=orchestrator_config,
            is_termination_msg=lambda x: "DONE" in (x.get("content", "") or "").upper(),
        )

        result = orchestrator.initiate_chat(
            recipient=manager,
            message="Help me organize a task.",
            max_turns=3,
        )

        # Verify result structure
        assert result is not None
        assert hasattr(result, "chat_history")
        assert isinstance(result.chat_history, list)
        assert len(result.chat_history) > 0

        # Verify messages structure
        for msg in result.chat_history:
            assert isinstance(msg, dict)
            assert "role" in msg
            assert "content" in msg


# @pytest.mark.integration
@run_for_optional_imports(["boto3", "botocore"], "bedrock")
class TestBedrockV2ClientMessageRetrieval:
    """Test message retrieval methods."""

    def test_message_retrieval(self, bedrock_v2_client, bedrock_v2_config):
        """Test message retrieval returns correct format."""
        response = bedrock_v2_client.create({
            "model": bedrock_v2_config["model"],
            "messages": [{"role": "user", "content": "Say 'integration test'"}],
        })

        messages = bedrock_v2_client.message_retrieval(response)

        # Verify return type and structure
        assert isinstance(messages, list)
        assert len(messages) > 0

        # Verify first message is string (simple text response)
        assert isinstance(messages[0], str)
        assert len(messages[0]) > 0
        assert "integration" in messages[0].lower() or "test" in messages[0].lower()

    def test_message_retrieval_with_tool_calls(self, bedrock_v2_client, bedrock_v2_config):
        """Test message retrieval with tool calls."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]

        response = bedrock_v2_client.create({
            "model": bedrock_v2_config["model"],
            "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}],
            "tools": tools,
        })

        messages = bedrock_v2_client.message_retrieval(response)

        # Should return dict format when tool calls present
        assert isinstance(messages, list)
        assert len(messages) > 0

        # Check if tool calls are present
        if response.finish_reason == "tool_calls":
            assert isinstance(messages[0], dict)
            assert "tool_calls" in messages[0]
            assert isinstance(messages[0]["tool_calls"], list)
            assert len(messages[0]["tool_calls"]) > 0
