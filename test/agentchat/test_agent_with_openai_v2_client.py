# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for OpenAIResponsesClient V2 with AG2 agents.

These tests verify that the ModelClientV2 architecture works seamlessly with
AG2's agent system, including AssistantAgent, UserProxyAgent, multi-turn conversations,
and group chat scenarios.

The V2 client returns rich UnifiedResponse objects with typed content blocks while
maintaining full compatibility with existing agent infrastructure via duck typing.

Run with:
    bash scripts/test-core-llm.sh test/agentchat/test_agent_with_openai_v2_client.py
"""

import os
from typing import Any

import pytest

from autogen import AssistantAgent, ConversableAgent, UserProxyAgent
from autogen.agentchat.group.multi_agent_chat import initiate_group_chat
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.agentchat.groupchat import GroupChat, GroupChatManager
from autogen.code_utils import content_str
from autogen.import_utils import run_for_optional_imports
from test.credentials import Credentials


def _assert_v2_response_structure(chat_result: Any) -> None:
    """Verify that chat result has expected structure."""
    assert chat_result is not None, "Chat result should not be None"
    assert hasattr(chat_result, "chat_history"), "Should have chat_history"
    assert hasattr(chat_result, "cost"), "Should have cost tracking"
    assert hasattr(chat_result, "summary"), "Should have summary"
    assert len(chat_result.chat_history) >= 2, "Should have at least 2 messages"


def _create_test_v2_config(credentials: Credentials) -> dict[str, Any]:
    """Create V2 client config from credentials."""
    # Extract the base config and add api_type
    base_config = credentials.llm_config._model.config_list[0]

    return {
        "config_list": [
            {
                "api_type": "openai_v2",  # Use V2 client
                "model": getattr(base_config, "model", "gpt-4o-mini"),
                "api_key": getattr(base_config, "api_key", os.getenv("OPENAI_API_KEY")),
            }
        ],
        "temperature": 0.3,
    }


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_simple_chat(credentials_gpt_4o_mini: Credentials) -> None:
    """Test basic chat using V2 client with real API."""
    llm_config = _create_test_v2_config(credentials_gpt_4o_mini)

    assistant = AssistantAgent(
        name="assistant",
        llm_config=llm_config,
        system_message="You are a helpful assistant. Be concise.",
    )

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False
    )

    chat_result = user_proxy.initiate_chat(
        assistant, message="What is 2 + 2? Answer with just the number.", max_turns=1
    )

    _assert_v2_response_structure(chat_result)
    assert "4" in chat_result.summary
    # Verify cost tracking
    assert "usage_including_cached_inference" in chat_result.cost
    assert len(chat_result.cost["usage_including_cached_inference"]) > 0


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_with_vision_multimodal(credentials_gpt_4o_mini: Credentials) -> None:
    """Test V2 client with vision/multimodal content using formal image input format."""
    llm_config = _create_test_v2_config(credentials_gpt_4o_mini)

    vision_assistant = AssistantAgent(
        name="vision_bot",
        llm_config=llm_config,
        system_message="You are an AI assistant with vision capabilities. Analyze images accurately.",
    )

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False
    )

    # Use formal multimodal content format
    image_url = "https://upload.wikimedia.org/wikipedia/commons/3/3b/BlkStdSchnauzer2.jpg"
    multimodal_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What animal is in this image? Answer in one word."},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    }

    chat_result = user_proxy.initiate_chat(vision_assistant, message=multimodal_message, max_turns=1)

    _assert_v2_response_structure(chat_result)
    summary_lower = chat_result.summary.lower()
    assert "dog" in summary_lower or "schnauzer" in summary_lower
    # Verify cost tracking for vision
    assert "usage_including_cached_inference" in chat_result.cost
    assert len(chat_result.cost["usage_including_cached_inference"]) > 0

    # Verify multimodal content is preserved in history
    first_msg = chat_result.chat_history[0]
    assert isinstance(first_msg["content"], list), "First message should be multimodal"


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_multi_turn_conversation(credentials_gpt_4o_mini: Credentials) -> None:
    """Test multi-turn conversation maintains context with V2 client."""
    llm_config = _create_test_v2_config(credentials_gpt_4o_mini)

    assistant = AssistantAgent(name="assistant", llm_config=llm_config, system_message="You are helpful. Be brief.")

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False
    )

    # First turn
    chat_result = user_proxy.initiate_chat(
        assistant, message="My favorite color is blue.", max_turns=1, clear_history=True
    )
    _assert_v2_response_structure(chat_result)

    # Second turn - should remember context
    response = user_proxy.send(message="What is my favorite color?", recipient=assistant, request_reply=True)

    assert "blue" in str(response).lower()


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_with_system_message(credentials_gpt_4o_mini: Credentials) -> None:
    """Test V2 client respects system message configuration."""
    llm_config = _create_test_v2_config(credentials_gpt_4o_mini)

    assistant = AssistantAgent(
        name="math_tutor",
        llm_config=llm_config,
        system_message="You are a math tutor. Always show your work step by step.",
    )

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False
    )

    chat_result = user_proxy.initiate_chat(assistant, message="What is 15 + 27?", max_turns=1)

    _assert_v2_response_structure(chat_result)
    assert "42" in chat_result.summary


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_cost_tracking(credentials_gpt_4o_mini: Credentials) -> None:
    """Test that V2 client provides accurate cost tracking."""
    llm_config = _create_test_v2_config(credentials_gpt_4o_mini)

    assistant = AssistantAgent(name="assistant", llm_config=llm_config)

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False
    )

    chat_result = user_proxy.initiate_chat(assistant, message="Count from 1 to 5.", max_turns=1)

    # V2 client should provide accurate cost
    assert "usage_including_cached_inference" in chat_result.cost
    assert len(chat_result.cost["usage_including_cached_inference"]) > 0


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_group_chat(credentials_gpt_4o_mini: Credentials) -> None:
    """Test V2 client works in group chat scenarios."""
    llm_config = _create_test_v2_config(credentials_gpt_4o_mini)

    # Create specialized agents with V2 client
    analyst = ConversableAgent(
        name="analyst",
        llm_config=llm_config,
        human_input_mode="NEVER",
        system_message="You analyze data. Keep responses very brief.",
    )

    reviewer = ConversableAgent(
        name="reviewer",
        llm_config=llm_config,
        human_input_mode="NEVER",
        system_message="You review analysis. Keep responses very brief.",
    )

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False
    )

    # Create group chat
    groupchat = GroupChat(
        agents=[user_proxy, analyst, reviewer], messages=[], max_round=3, speaker_selection_method="round_robin"
    )

    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    chat_result = user_proxy.initiate_chat(
        manager, message="Team, analyze the number 42 and provide brief feedback.", max_turns=2
    )

    _assert_v2_response_structure(chat_result)

    # Verify agents participated
    participant_names = {msg.get("name") for msg in chat_result.chat_history if msg.get("name")}
    assert len(participant_names.intersection({"analyst", "reviewer"})) >= 1


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_run_interface(credentials_gpt_4o_mini: Credentials) -> None:
    """Test V2 client with ConversableAgent::run() interface."""
    llm_config = _create_test_v2_config(credentials_gpt_4o_mini)

    assistant = ConversableAgent(
        name="runner",
        llm_config=llm_config,
        human_input_mode="NEVER",
        system_message="You are helpful. Keep responses brief.",
    )

    # Test run interface
    run_response = assistant.run(
        message="Say exactly: 'Run interface works'", user_input=False, max_turns=1, clear_history=True
    )

    # Verify run response object
    assert run_response is not None
    assert hasattr(run_response, "messages")
    assert hasattr(run_response, "process")

    # Process the response
    run_response.process()

    # Verify messages
    messages_list = list(run_response.messages)
    assert len(messages_list) >= 2


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_content_str_compatibility(credentials_gpt_4o_mini: Credentials) -> None:
    """Test that V2 client responses work with content_str utility."""
    llm_config = _create_test_v2_config(credentials_gpt_4o_mini)

    assistant = ConversableAgent(name="assistant", llm_config=llm_config, human_input_mode="NEVER")

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False
    )

    chat_result = user_proxy.initiate_chat(assistant, message="Hello, how are you?", max_turns=1)

    _assert_v2_response_structure(chat_result)

    # Verify all messages can be processed by content_str
    for msg in chat_result.chat_history:
        content = msg["content"]
        try:
            content_string = content_str(content)
            assert isinstance(content_string, str)
        except Exception as e:
            pytest.fail(f"content_str failed on V2 client response: {e}")


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_vs_standard_comparison(credentials_gpt_4o_mini: Credentials) -> None:
    """Compare V2 client with standard client - both should work."""
    # Standard client config
    standard_config = {
        "config_list": [
            {
                "model": "gpt-4o-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
            }
        ],
        "temperature": 0,
    }

    standard_assistant = AssistantAgent(name="standard", llm_config=standard_config, system_message="Be concise.")

    # V2 client config
    v2_config = _create_test_v2_config(credentials_gpt_4o_mini)
    v2_assistant = AssistantAgent(name="v2_bot", llm_config=v2_config, system_message="Be concise.")

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False
    )

    prompt = "What is the capital of France? Answer in one word."

    # Test standard
    result_standard = user_proxy.initiate_chat(standard_assistant, message=prompt, max_turns=1, clear_history=True)

    # Test V2
    result_v2 = user_proxy.initiate_chat(v2_assistant, message=prompt, max_turns=1, clear_history=True)

    # Both should contain "Paris"
    assert "paris" in result_standard.summary.lower()
    assert "paris" in result_v2.summary.lower()

    # Both should have cost tracking
    assert "usage_including_cached_inference" in result_standard.cost
    assert len(result_standard.cost["usage_including_cached_inference"]) > 0
    assert "usage_including_cached_inference" in result_v2.cost
    assert len(result_v2.cost["usage_including_cached_inference"]) > 0


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_error_handling_invalid_model(credentials_gpt_4o_mini: Credentials) -> None:
    """Test V2 client error handling with invalid model."""
    llm_config = _create_test_v2_config(credentials_gpt_4o_mini)
    # Override with invalid model for error testing
    llm_config["config_list"][0]["model"] = "invalid-model-xyz-12345"

    assistant = AssistantAgent(name="error_bot", llm_config=llm_config)
    user_proxy = UserProxyAgent(name="user", human_input_mode="NEVER", max_consecutive_auto_reply=0)

    with pytest.raises(Exception):  # OpenAI will raise error for invalid model
        user_proxy.initiate_chat(assistant, message="Hello", max_turns=1)


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_sequential_chats(credentials_gpt_4o_mini: Credentials) -> None:
    """Test V2 client with sequential chats and carryover."""
    llm_config = _create_test_v2_config(credentials_gpt_4o_mini)

    user_proxy = UserProxyAgent(
        name="manager", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False
    )

    analyst = ConversableAgent(
        name="analyst", llm_config=llm_config, human_input_mode="NEVER", system_message="Analyze briefly."
    )

    reviewer = ConversableAgent(
        name="reviewer", llm_config=llm_config, human_input_mode="NEVER", system_message="Review briefly."
    )

    # Sequential chat sequence
    chat_sequence = [
        {"recipient": analyst, "message": "Analyze the number 42.", "max_turns": 1, "summary_method": "last_msg"},
        {"recipient": reviewer, "message": "Review the analysis.", "max_turns": 1},
    ]

    chat_results = user_proxy.initiate_chats(chat_sequence)

    # Verify sequential execution
    assert len(chat_results) == 2
    assert all(result.chat_history for result in chat_results)

    # Verify carryover context
    second_chat = chat_results[1]
    second_first_msg = second_chat.chat_history[0]
    content_str_rep = str(second_first_msg.get("content", ""))

    # Should have carryover context
    assert len(content_str_rep) >= len("Review the analysis.")


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_backwards_compatibility(credentials_gpt_4o_mini: Credentials) -> None:
    """Test V2 client maintains backwards compatibility with string/dict messages."""
    llm_config = _create_test_v2_config(credentials_gpt_4o_mini)

    assistant = ConversableAgent(name="compat_bot", llm_config=llm_config, human_input_mode="NEVER")

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False
    )

    # Test 1: String message
    result1 = user_proxy.initiate_chat(assistant, message="Hello, this is a string message.", max_turns=1)
    assert result1 is not None
    assert len(result1.chat_history) >= 2

    # Test 2: Dict message
    result2 = user_proxy.initiate_chat(
        assistant,
        message={"role": "user", "content": "This is a dict message."},
        max_turns=1,
        clear_history=True,
    )
    assert result2 is not None
    assert len(result2.chat_history) >= 2


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_multimodal_with_multiple_images(credentials_gpt_4o_mini: Credentials) -> None:
    """Test V2 client with multiple images in one request."""
    llm_config = _create_test_v2_config(credentials_gpt_4o_mini)

    vision_assistant = AssistantAgent(name="vision_bot", llm_config=llm_config)

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False
    )

    # Two dog images
    image_url_1 = "https://upload.wikimedia.org/wikipedia/commons/3/3b/BlkStdSchnauzer2.jpg"
    image_url_2 = "https://upload.wikimedia.org/wikipedia/commons/2/2d/Golde33443.jpg"

    multimodal_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Compare these two dogs briefly."},
            {"type": "image_url", "image_url": {"url": image_url_1}},
            {"type": "image_url", "image_url": {"url": image_url_2}},
        ],
    }

    chat_result = user_proxy.initiate_chat(vision_assistant, message=multimodal_message, max_turns=1)

    _assert_v2_response_structure(chat_result)
    # Verify cost tracking for multiple images
    assert "usage_including_cached_inference" in chat_result.cost
    assert len(chat_result.cost["usage_including_cached_inference"]) > 0


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_with_group_pattern(credentials_gpt_4o_mini: Credentials) -> None:
    """Test V2 client with DefaultPattern group orchestration."""
    llm_config = _create_test_v2_config(credentials_gpt_4o_mini)

    # Create specialized agents with V2 client
    analyst = ConversableAgent(
        name="DataAnalyst",
        llm_config=llm_config,
        human_input_mode="NEVER",
        system_message="You analyze data. Be brief and focused.",
    )

    reviewer = ConversableAgent(
        name="QualityReviewer",
        llm_config=llm_config,
        human_input_mode="NEVER",
        system_message="You review analysis quality. Be concise.",
    )

    # Create pattern-based group chat
    pattern = DefaultPattern(
        initial_agent=analyst,
        agents=[analyst, reviewer],
    )

    # Initiate group chat using pattern
    chat_result, context_variables, last_agent = initiate_group_chat(
        pattern=pattern,
        messages="Analyze the number 42 briefly, then have the reviewer comment.",
        max_rounds=3,
    )

    # Verify pattern-based group chat works with V2 client
    _assert_v2_response_structure(chat_result)
    assert len(chat_result.chat_history) >= 2
    assert "usage_including_cached_inference" in chat_result.cost
    assert len(chat_result.cost["usage_including_cached_inference"]) > 0

    # Verify agents participated
    participant_names = {msg.get("name") for msg in chat_result.chat_history if msg.get("name")}
    assert len(participant_names.intersection({"DataAnalyst", "QualityReviewer"})) >= 1

    # Verify context variables and last agent
    assert context_variables is not None
    assert last_agent is not None


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_pattern_with_vision(credentials_gpt_4o_mini: Credentials) -> None:
    """Test V2 client with DefaultPattern and vision/multimodal content."""
    llm_config = _create_test_v2_config(credentials_gpt_4o_mini)

    # Create vision-capable agents
    image_describer = ConversableAgent(
        name="ImageDescriber",
        llm_config=llm_config,
        human_input_mode="NEVER",
        system_message="You describe images concisely.",
    )

    detail_analyst = ConversableAgent(
        name="DetailAnalyst",
        llm_config=llm_config,
        human_input_mode="NEVER",
        system_message="You analyze image details. Be brief.",
    )

    # Create pattern with vision agents
    pattern = DefaultPattern(
        initial_agent=image_describer,
        agents=[image_describer, detail_analyst],
    )

    # Multimodal message with image
    image_url = "https://upload.wikimedia.org/wikipedia/commons/3/3b/BlkStdSchnauzer2.jpg"
    multimodal_message = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Team, analyze this image and identify the animal breed."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]

    # Initiate group chat with image
    chat_result, context_variables, last_agent = initiate_group_chat(
        pattern=pattern,
        messages=multimodal_message,
        max_rounds=3,
    )

    # Verify pattern works with multimodal V2 responses
    _assert_v2_response_structure(chat_result)
    summary_lower = chat_result.summary.lower()
    assert "dog" in summary_lower or "schnauzer" in summary_lower

    # Verify cost tracking
    assert "usage_including_cached_inference" in chat_result.cost
    assert len(chat_result.cost["usage_including_cached_inference"]) > 0

    # Verify multimodal content preserved
    first_msg = chat_result.chat_history[0]
    assert isinstance(first_msg["content"], list), "First message should be multimodal"

    # Verify context and last agent
    assert context_variables is not None
    assert last_agent is not None
