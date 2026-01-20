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

from autogen import ConversableAgent, GroupChat, GroupChatManager, LLMConfig, UserProxyAgent
from autogen.import_utils import run_for_optional_imports
from autogen.llm_clients.bedrock_v2 import BedrockV2Client
from autogen.llm_clients.models import UnifiedResponse


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

    def test_direct_client_usage(self, bedrock_v2_client):
        """Test direct client usage with UnifiedResponse."""
        response = bedrock_v2_client.create({
            "model": bedrock_v2_client._aws_region or "us-east-1",
            "messages": [{"role": "user", "content": "Say 'Hello' in one word."}],
        })

        assert isinstance(response, UnifiedResponse)
        assert response.provider == "bedrock"
        assert len(response.messages) > 0
        assert "hello" in response.text.lower()
        assert response.usage["prompt_tokens"] > 0
        assert response.usage["completion_tokens"] > 0

    def test_content_blocks_access(self, bedrock_v2_client, bedrock_v2_config):
        """Test accessing content blocks from response."""
        response = bedrock_v2_client.create({
            "model": bedrock_v2_config["model"],
            "messages": [{"role": "user", "content": "List 3 benefits of cloud computing."}],
        })

        assert len(response.messages) > 0
        text_blocks = response.get_content_by_type("text")
        assert len(text_blocks) > 0
        assert len(response.text) > 0

    def test_usage_and_cost_tracking(self, bedrock_v2_client, bedrock_v2_config):
        """Test usage and cost tracking."""
        response = bedrock_v2_client.create({
            "model": bedrock_v2_config["model"],
            "messages": [{"role": "user", "content": "Count from 1 to 3."}],
        })

        usage = bedrock_v2_client.get_usage(response)
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] > 0
        assert usage["model"] == bedrock_v2_config["model"]
        assert response.cost >= 0


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
        assert result is not None
        assert "4" in result.lower() or "answer" in result.lower()

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
        assert result is not None


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

        assert result_v2 is not None
        assert result_v1 is not None
        assert "10" in result_v2.lower() or "10" in result_v1.lower()

    def test_v1_compatible_format(self, bedrock_v2_client, bedrock_v2_config):
        """Test create_v1_compatible returns correct format."""
        v1_response = bedrock_v2_client.create_v1_compatible({
            "model": bedrock_v2_config["model"],
            "messages": [{"role": "user", "content": "Say 'test'"}],
        })

        assert isinstance(v1_response, dict)
        assert "choices" in v1_response
        assert "usage" in v1_response
        assert v1_response["object"] == "chat.completion"


@pytest.mark.integration
@run_for_optional_imports(["boto3", "botocore"], "bedrock")
class TestBedrockV2ClientGroupChat:
    """Test group chat with Bedrock V2."""

    def test_group_chat_mixed_v1_v2(self, bedrock_v2_config):
        """Test group chat with mixed V1/V2 clients."""
        planner = ConversableAgent(
            name="planner",
            llm_config=LLMConfig(config_list=[bedrock_v2_config]),
            system_message="Create plans.",
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
            system_message="Review plans.",
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

        assert result is not None
        assert len(result.chat_history) > 0

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
            system_message="Route tasks to appropriate agents.",
            max_consecutive_auto_reply=2,
            human_input_mode="NEVER",
        )

        worker = ConversableAgent(
            name="worker",
            llm_config=regular_config,
            system_message="Complete assigned tasks.",
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

        assert result is not None


@pytest.mark.integration
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
        assert len(messages) > 0
        assert isinstance(messages[0], str)
        assert len(messages[0]) > 0
