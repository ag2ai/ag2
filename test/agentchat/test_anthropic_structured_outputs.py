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

    # Verify response is formatted output (FormatterProtocol applied in message_retrieval)
    last_message_content = chat_result.chat_history[-1]["content"]

    # Content should be formatted text, not JSON (FormatterProtocol.format() was applied)
    assert isinstance(last_message_content, str), "Content should be a string"
    assert "Step" in last_message_content, "Should contain formatted steps"
    assert "Final Answer:" in last_message_content, "Should contain 'Final Answer:'"
    assert "x = 5" in last_message_content or "5" in last_message_content, "Should contain the answer"


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

    # Verify JSON Mode produces formatted output (FormatterProtocol applied in message_retrieval)
    last_message_content = chat_result.chat_history[-1]["content"]

    # Content should be formatted text, not JSON (FormatterProtocol.format() was applied)
    assert isinstance(last_message_content, str), "Content should be a string"
    assert "Step" in last_message_content, "Should contain formatted steps"
    assert "Final Answer:" in last_message_content, "Should contain 'Final Answer:'"
    assert "10" in last_message_content or "x = 10" in last_message_content.lower(), "Should contain the answer (x=10)"


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
        # In GroupChat, agent responses have role="user" when sent to manager
        # Skip the initial message from user_proxy (role="assistant", name="User")
        if message.get("name") in ["DataAnalyst", "MathExpert"]:
            content = message["content"]

            # Try to validate as AnalysisResult (remains JSON - no format() method)
            try:
                result = AnalysisResult.model_validate_json(content)
                assert result.summary, "AnalysisResult should have summary"
                assert len(result.key_findings) > 0, "Should have key findings"
                assert result.recommendation, "Should have recommendation"
                found_valid_structure = True
                break
            except (ValidationError, ValueError):
                pass

            # Check for formatted MathReasoning (has format() method - will be formatted text)
            if isinstance(content, str) and "Step" in content and "Final Answer:" in content:
                found_valid_structure = True
                break

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
        # Check messages from Analyst and Solver agents
        if message.get("name") in ["Analyst", "Solver"]:
            content = message.get("content", "")

            # Check for AnalysisResult (remains JSON - no format() method)
            try:
                AnalysisResult.model_validate_json(content)
                found_analysis = True
            except (ValidationError, ValueError):
                pass

            # Check for formatted MathReasoning (has format() method - will be formatted text)
            if isinstance(content, str) and "Step" in content and "Final Answer:" in content:
                found_math = True
                # Verify the answer (15% of 200 = 30)
                assert "30" in content

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

    # Verify that FormatterProtocol.format() was already applied in message_retrieval()
    # Content should be formatted text, not JSON
    assert isinstance(last_message, str), "Content should be a string"
    assert "Step" in last_message, "Formatted output should contain steps"
    assert "Final Answer:" in last_message, "Formatted output should contain final answer"
    # Verify the calculation result (5 + 3 * 2 = 11)
    assert "11" in last_message, "Formatted output should include the answer"


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

    # Both agents should produce valid formatted outputs (via different methods)
    valid_outputs = 0

    for message in chat_result.chat_history:
        # Check messages from SonnetExpert and HaikuHelper agents
        if message.get("name") in ["SonnetExpert", "HaikuHelper"]:
            content = message["content"]
            # Check for formatted MathReasoning (contains Step and Final Answer)
            if isinstance(content, str) and "Step" in content and "Final Answer:" in content:
                valid_outputs += 1

    assert valid_outputs >= 1, "At least one agent should produce valid formatted output"


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


# ==============================================================================
# E2E Test 6: Strict Tool Use
# ==============================================================================


@pytest.mark.anthropic
@pytest.mark.aux_neg_flag
@run_for_optional_imports(["anthropic"], "anthropic")
def test_strict_tool_use(credentials_anthropic_claude_sonnet):
    """Test that strict: true is preserved and enables schema validation.

    This test verifies that when tools are marked with strict=True,
    Anthropic's constrained decoding ensures tool inputs strictly follow
    the defined schema with guaranteed type safety.
    """

    # Define a tool with strict mode enabled
    def get_weather(location: str, unit: str = "celsius") -> str:
        """Get the weather for a location.

        Args:
            location: The city and state, e.g. San Francisco, CA
            unit: Temperature unit (celsius or fahrenheit)
        """
        return f"Weather in {location}: 22 {unit}"

    llm_config = {
        "config_list": credentials_anthropic_claude_sonnet.config_list,
        "functions": [
            {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "strict": True,  # Enable strict mode for guaranteed schema compliance
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit",
                        },
                    },
                    "required": ["location"],
                },
            }
        ],
    }

    assistant = autogen.AssistantAgent(
        name="WeatherAssistant",
        system_message="You help users get weather information. Use the get_weather function.",
        llm_config=llm_config,
    )

    assistant.register_function({"get_weather": get_weather})

    user_proxy = autogen.UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        code_execution_config=False,
    )

    # Initiate chat that should trigger tool use
    chat_result = user_proxy.initiate_chat(
        assistant,
        message="What's the weather in Boston, MA?",
        max_turns=2,
    )

    # Verify tool was called
    found_tool_call = False
    for message in chat_result.chat_history:
        if message.get("tool_calls"):
            found_tool_call = True
            # Verify tool call has correct structure
            tool_call = message["tool_calls"][0]
            assert tool_call["function"]["name"] == "get_weather"
            # With strict mode, inputs should be properly typed
            import json

            args = json.loads(tool_call["function"]["arguments"])
            assert isinstance(args["location"], str)
            if "unit" in args:
                assert args["unit"] in ["celsius", "fahrenheit"]
            break

    assert found_tool_call, "Strict tool use should trigger tool call"


@pytest.mark.anthropic
@pytest.mark.aux_neg_flag
@run_for_optional_imports(["anthropic"], "anthropic")
def test_combined_json_output_and_strict_tools(credentials_anthropic_claude_sonnet):
    """Test using both JSON outputs and strict tools together.

    Anthropic's structured outputs support using both modes simultaneously:
    - response_format for structured JSON responses
    - strict: true for validated tool inputs
    """

    # Define a calculator tool with strict mode
    def calculate(operation: str, a: float, b: float) -> float:
        """Perform a calculation.

        Args:
            operation: The operation to perform (add, subtract, multiply, divide)
            a: First number
            b: Second number
        """
        if operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            return a / b if b != 0 else 0
        return 0

    # Result model for structured output
    class CalculationResult(BaseModel):
        """Structured output for calculation results."""

        problem: str
        steps: list[str]
        result: float
        verification: str

    llm_config = {
        "config_list": [
            {
                **credentials_anthropic_claude_sonnet.config_list[0],
                "response_format": CalculationResult,  # Structured JSON output
            }
        ],
        "functions": [
            {
                "name": "calculate",
                "description": "Perform arithmetic calculation",
                "strict": True,  # Strict tool validation
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["operation", "a", "b"],
                },
            }
        ],
    }

    assistant = autogen.AssistantAgent(
        name="MathAssistant",
        system_message="You solve math problems using tools and provide structured results.",
        llm_config=llm_config,
    )

    assistant.register_function({"calculate": calculate})

    user_proxy = autogen.UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=2,
        code_execution_config=False,
    )

    # Initiate chat requiring both tool use and structured output
    chat_result = user_proxy.initiate_chat(
        assistant,
        message="Calculate (15 + 7) * 3 and explain your steps",
        max_turns=3,
    )

    # Verify both features work together
    found_tool_call = False
    found_structured_output = False

    for message in chat_result.chat_history:
        # Check for tool calls
        if message.get("tool_calls"):
            found_tool_call = True
            tool_call = message["tool_calls"][0]
            assert tool_call["function"]["name"] == "calculate"
            # Verify strict typing
            import json

            args = json.loads(tool_call["function"]["arguments"])
            assert isinstance(args["a"], (int, float))
            assert isinstance(args["b"], (int, float))
            assert args["operation"] in ["add", "subtract", "multiply", "divide"]

        # Check for structured output
        if message.get("role") == "assistant" and message.get("content"):
            try:
                result = CalculationResult.model_validate_json(message["content"])
                found_structured_output = True
                assert result.problem
                assert len(result.steps) > 0
                assert isinstance(result.result, (int, float))
                assert result.verification
            except (ValidationError, ValueError):
                pass

    # Both features should work together
    assert found_tool_call, "Should use strict tools for calculations"
    assert found_structured_output, "Should produce structured JSON output"
