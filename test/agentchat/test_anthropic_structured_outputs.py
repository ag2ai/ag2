# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

"""
E2E Integration tests for Anthropic native structured outputs.

Tests verify that:
1. Claude Sonnet 4.5+ uses native structured outputs (beta API)
2. Older Claude models fallback to JSON Mode
3. Both modes produce valid, schema-compliant responses
4. Works with two-agent chat, groupchat, and AutoPattern
"""

import pytest
from pydantic import BaseModel, ValidationError

import autogen
from autogen.agentchat.groupchat import GroupChat, GroupChatManager
from autogen.import_utils import run_for_optional_imports

try:
    from autogen.agentchat.group.multi_agent_chat import initiate_group_chat
    from autogen.agentchat.group.patterns import DefaultPattern

    HAS_GROUP_PATTERNS = True
except ImportError:
    HAS_GROUP_PATTERNS = False


# Pydantic models for structured outputs
class Step(BaseModel):
    """A single step in mathematical reasoning."""

    explanation: str
    output: str


class MathReasoning(BaseModel):
    """Structured output for mathematical problem solving."""

    steps: list[Step]
    final_answer: str

    def format(self) -> str:
        """Format the response for display."""
        steps_output = "\n".join(
            f"Step {i + 1}: {step.explanation}\n  Output: {step.output}" for i, step in enumerate(self.steps)
        )
        return f"{steps_output}\n\nFinal Answer: {self.final_answer}"


class AnalysisResult(BaseModel):
    """Structured output for data analysis."""

    summary: str
    key_findings: list[str]
    recommendation: str


class AgentResponse(BaseModel):
    """Generic structured agent response."""

    agent_name: str
    response_type: str
    content: str
    confidence: float


# Fixture for Sonnet 4.5 (native structured outputs)
@pytest.fixture
def config_list_sonnet_4_5_structured(credentials_anthropic_claude_sonnet):
    """Config for Claude Sonnet 4.5 with structured outputs."""
    config_list = []
    for config in credentials_anthropic_claude_sonnet.config_list:
        new_config = config.copy()
        new_config["response_format"] = MathReasoning
        config_list.append(new_config)
    return config_list


# Fixture for older Claude model (JSON Mode fallback)
@pytest.fixture
def config_list_haiku_structured():
    """Config for Claude Haiku with structured outputs (JSON Mode)."""
    import os

    return [
        {
            "model": "claude-3-haiku-20240307",
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "api_type": "anthropic",
            "response_format": MathReasoning,
        }
    ]


# ==============================================================================
# E2E Test 1: Two-Agent Chat with Structured Outputs
# ==============================================================================


@pytest.mark.anthropic
@pytest.mark.aux_neg_flag
@run_for_optional_imports(["anthropic"], "anthropic")
def test_two_agent_chat_native_structured_output(config_list_sonnet_4_5_structured):
    """Test two-agent chat with Sonnet 4.5 using native structured outputs."""

    llm_config = {
        "config_list": config_list_sonnet_4_5_structured,
    }

    user_proxy = autogen.UserProxyAgent(
        name="User",
        system_message="A human user asking math questions.",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    math_assistant = autogen.AssistantAgent(
        name="MathAssistant",
        system_message="You are a math tutor. Solve problems step by step.",
        llm_config=llm_config,
    )

    # Initiate chat with math problem
    chat_result = user_proxy.initiate_chat(
        math_assistant,
        message="Solve the equation: 3x + 7 = 22",
        max_turns=1,
        summary_method="last_msg",
    )

    # Verify response is valid structured output
    last_message_content = chat_result.chat_history[-1]["content"]

    try:
        result = MathReasoning.model_validate_json(last_message_content)

        # Validate structure
        assert len(result.steps) > 0, "Should have at least one step"
        assert result.final_answer, "Should have a final answer"

        # Validate each step has required fields
        for step in result.steps:
            assert step.explanation, "Each step should have an explanation"
            assert step.output, "Each step should have output"

        # Verify the answer makes sense (x should be 5)
        assert "5" in result.final_answer or "x = 5" in result.final_answer.lower()

    except ValidationError as e:
        raise AssertionError(f"Response did not match MathReasoning schema: {e}")


@pytest.mark.anthropic
@pytest.mark.aux_neg_flag
@run_for_optional_imports(["anthropic"], "anthropic")
def test_two_agent_chat_json_mode_fallback(config_list_haiku_structured):
    """Test two-agent chat with older Claude model using JSON Mode fallback."""

    llm_config = {
        "config_list": config_list_haiku_structured,
    }

    user_proxy = autogen.UserProxyAgent(
        name="User",
        system_message="A human user asking math questions.",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    math_assistant = autogen.AssistantAgent(
        name="MathAssistant",
        system_message="You are a math tutor. Solve problems step by step.",
        llm_config=llm_config,
    )

    # Initiate chat
    chat_result = user_proxy.initiate_chat(
        math_assistant,
        message="Solve: 2x - 5 = 15",
        max_turns=1,
        summary_method="last_msg",
    )

    # Verify JSON Mode still produces valid structured output
    last_message_content = chat_result.chat_history[-1]["content"]

    try:
        result = MathReasoning.model_validate_json(last_message_content)
        assert len(result.steps) > 0, "Should have steps even with JSON Mode"
        assert result.final_answer, "Should have final answer"

        # Verify answer (x should be 10)
        assert "10" in result.final_answer or "x = 10" in result.final_answer.lower()

    except ValidationError as e:
        raise AssertionError(f"JSON Mode fallback failed to produce valid output: {e}")


# ==============================================================================
# E2E Test 2: GroupChat with Structured Outputs
# ==============================================================================


@pytest.mark.anthropic
@pytest.mark.aux_neg_flag
@run_for_optional_imports(["anthropic"], "anthropic")
def test_groupchat_structured_output(config_list_sonnet_4_5_structured):
    """Test GroupChat with multiple agents using structured outputs."""

    # Modify config for different response formats per agent
    config_analysis = []
    for config in config_list_sonnet_4_5_structured:
        new_config = config.copy()
        new_config["response_format"] = AnalysisResult
        config_analysis.append(new_config)

    config_math = []
    for config in config_list_sonnet_4_5_structured:
        new_config = config.copy()
        new_config["response_format"] = MathReasoning
        config_math.append(new_config)

    user_proxy = autogen.UserProxyAgent(
        name="User",
        system_message="A human admin initiating tasks.",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    data_analyst = autogen.AssistantAgent(
        name="DataAnalyst",
        system_message="You analyze data and provide insights in structured format.",
        llm_config={"config_list": config_analysis},
    )

    math_expert = autogen.AssistantAgent(
        name="MathExpert",
        system_message="You solve mathematical problems with step-by-step reasoning.",
        llm_config={"config_list": config_math},
    )

    # Create groupchat
    groupchat = GroupChat(
        agents=[user_proxy, data_analyst, math_expert],
        messages=[],
        max_round=3,
        speaker_selection_method="round_robin",
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config={"config_list": config_list_sonnet_4_5_structured},
    )

    # Start groupchat with a task
    chat_result = user_proxy.initiate_chat(
        manager,
        message="Analyze this dataset: [10, 20, 30, 40, 50] and explain the mean calculation.",
        max_turns=2,
    )

    # Verify that at least one agent produced structured output
    found_valid_structure = False

    for message in chat_result.chat_history:
        if message["role"] == "assistant":
            content = message["content"]

            # Try to validate as AnalysisResult
            try:
                result = AnalysisResult.model_validate_json(content)
                assert result.summary, "AnalysisResult should have summary"
                assert len(result.key_findings) > 0, "Should have key findings"
                assert result.recommendation, "Should have recommendation"
                found_valid_structure = True
                break
            except (ValidationError, ValueError):
                pass

            # Try to validate as MathReasoning
            try:
                result = MathReasoning.model_validate_json(content)
                assert len(result.steps) > 0, "MathReasoning should have steps"
                assert result.final_answer, "Should have final answer"
                found_valid_structure = True
                break
            except (ValidationError, ValueError):
                pass

    assert found_valid_structure, "At least one agent should produce valid structured output"


# ==============================================================================
# E2E Test 3: GroupChat with DefaultPattern and Structured Outputs
# ==============================================================================


@pytest.mark.anthropic
@pytest.mark.aux_neg_flag
@pytest.mark.skipif(not HAS_GROUP_PATTERNS, reason="Requires group patterns")
@run_for_optional_imports(["anthropic"], "anthropic")
def test_groupchat_defaultpattern_structured_output(config_list_sonnet_4_5_structured):
    """Test GroupChat with DefaultPattern using structured outputs."""

    # Create agents with structured response formats
    analyst = autogen.AssistantAgent(
        name="Analyst",
        system_message="You analyze problems and provide structured insights.",
        llm_config={
            "config_list": [
                {
                    **config_list_sonnet_4_5_structured[0],
                    "response_format": AnalysisResult,
                }
            ],
        },
    )

    solver = autogen.AssistantAgent(
        name="Solver",
        system_message="You solve problems with step-by-step reasoning.",
        llm_config={
            "config_list": [
                {
                    **config_list_sonnet_4_5_structured[0],
                    "response_format": MathReasoning,
                }
            ],
        },
    )

    # Create DefaultPattern for orchestration
    pattern = DefaultPattern(
        initial_agent=analyst,
        agents=[analyst, solver],
    )

    # Initiate group chat
    messages = [
        {
            "role": "user",
            "content": "First analyze this problem: What is 15% of 200? Then solve it step by step.",
        }
    ]

    chat_result, context_variables, last_agent = initiate_group_chat(
        pattern=pattern,
        messages=messages,
        max_rounds=3,
    )

    # Verify structured outputs from different agents
    found_analysis = False
    found_math = False

    for message in chat_result.chat_history:
        if message.get("role") == "assistant":
            content = message.get("content", "")

            # Check for AnalysisResult structure
            try:
                result = AnalysisResult.model_validate_json(content)
                found_analysis = True
            except (ValidationError, ValueError):
                pass

            # Check for MathReasoning structure
            try:
                result = MathReasoning.model_validate_json(content)
                found_math = True
                # Verify the answer (15% of 200 = 30)
                assert "30" in result.final_answer
            except (ValidationError, ValueError):
                pass

    # At least one type of structured output should be found
    assert found_analysis or found_math, "DefaultPattern should produce structured outputs"


# ==============================================================================
# E2E Test 4: Verify Format Method Integration
# ==============================================================================


@pytest.mark.anthropic
@pytest.mark.aux_neg_flag
@run_for_optional_imports(["anthropic"], "anthropic")
def test_structured_output_with_format_method(config_list_sonnet_4_5_structured):
    """Test that custom format() method is called correctly."""

    llm_config = {
        "config_list": config_list_sonnet_4_5_structured,
    }

    user_proxy = autogen.UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    assistant = autogen.AssistantAgent(
        name="Assistant",
        llm_config=llm_config,
    )

    chat_result = user_proxy.initiate_chat(
        assistant,
        message="Calculate: 5 + 3 * 2",
        max_turns=1,
        summary_method="last_msg",
    )

    last_message = chat_result.chat_history[-1]["content"]

    # Parse the structured output
    result = MathReasoning.model_validate_json(last_message)

    # Call the format method
    formatted = result.format()

    # Verify format method produces expected output
    assert "Step" in formatted, "Formatted output should contain steps"
    assert "Final Answer:" in formatted, "Formatted output should contain final answer"
    assert result.final_answer in formatted, "Formatted output should include the answer"


# ==============================================================================
# E2E Test 5: Mixed Models in GroupChat (Native + JSON Mode)
# ==============================================================================


@pytest.mark.anthropic
@pytest.mark.aux_neg_flag
@run_for_optional_imports(["anthropic"], "anthropic")
def test_groupchat_mixed_models(config_list_sonnet_4_5_structured, config_list_haiku_structured):
    """Test GroupChat with mixed Claude models (Sonnet 4.5 + Haiku)."""

    user_proxy = autogen.UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    # Sonnet 4.5 agent (native structured outputs)
    sonnet_agent = autogen.AssistantAgent(
        name="SonnetExpert",
        system_message="Advanced math expert using latest Claude.",
        llm_config={"config_list": config_list_sonnet_4_5_structured},
    )

    # Haiku agent (JSON Mode fallback)
    haiku_agent = autogen.AssistantAgent(
        name="HaikuHelper",
        system_message="Quick math helper.",
        llm_config={"config_list": config_list_haiku_structured},
    )

    groupchat = GroupChat(
        agents=[user_proxy, sonnet_agent, haiku_agent],
        messages=[],
        max_round=3,
        speaker_selection_method="round_robin",
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config={"config_list": config_list_sonnet_4_5_structured},
    )

    chat_result = user_proxy.initiate_chat(
        manager,
        message="Calculate: (10 + 5) * 2",
        max_turns=2,
    )

    # Both agents should produce valid structured outputs (via different methods)
    valid_outputs = 0

    for message in chat_result.chat_history:
        if message["role"] == "assistant":
            try:
                result = MathReasoning.model_validate_json(message["content"])
                assert result.steps and result.final_answer
                valid_outputs += 1
            except (ValidationError, ValueError):
                pass

    assert valid_outputs >= 1, "At least one agent should produce valid structured output"


# ==============================================================================
# E2E Test 6: Error Handling and Fallback
# ==============================================================================


@pytest.mark.anthropic
@pytest.mark.aux_neg_flag
@run_for_optional_imports(["anthropic"], "anthropic")
def test_structured_output_error_handling(config_list_sonnet_4_5_structured):
    """Test error handling when structured output fails."""

    class ComplexModel(BaseModel):
        """A more complex model to potentially trigger errors."""

        nested_data: dict[str, list[dict[str, str]]]
        numbers: list[float]
        summary: str

    config_complex = []
    for config in config_list_sonnet_4_5_structured:
        new_config = config.copy()
        new_config["response_format"] = ComplexModel
        config_complex.append(new_config)

    llm_config = {
        "config_list": config_complex,
    }

    user_proxy = autogen.UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    assistant = autogen.AssistantAgent(
        name="Assistant",
        llm_config=llm_config,
    )

    # Should not crash even with complex schema
    try:
        chat_result = user_proxy.initiate_chat(
            assistant,
            message="Provide complex nested data structure.",
            max_turns=1,
            summary_method="last_msg",
        )

        # If it succeeds, verify structure
        last_content = chat_result.chat_history[-1]["content"]
        ComplexModel.model_validate_json(last_content)

    except Exception as e:
        # Should have graceful error handling, not crash
        pytest.fail(f"Should handle complex schemas gracefully: {e}")
