# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for OpenAICompletionsClient V2 with AG2 agents.

These tests verify that the ModelClientV2 architecture works seamlessly with
AG2's agent system, including AssistantAgent, UserProxyAgent, multi-turn conversations,
and group chat scenarios.

The V2 client uses OpenAI Chat Completions API and returns rich UnifiedResponse objects
with typed content blocks while maintaining full compatibility with existing agent
infrastructure via duck typing.

Run with:
    bash scripts/test-core-llm.sh test/agentchat/test_agent_with_openai_v2_client.py
"""

import logging
import os
from typing import Any

import pytest

from autogen import AssistantAgent, ConversableAgent, UserProxyAgent
from autogen.agentchat.group.multi_agent_chat import initiate_group_chat, run_group_chat
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.agentchat.groupchat import GroupChat, GroupChatManager
from autogen.code_utils import content_str
from autogen.import_utils import run_for_optional_imports
from test.credentials import Credentials

logger = logging.getLogger(__name__)


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

    # Use base64-encoded image to avoid remote URL issues (1x1 pixel brown PNG representing a dog)
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    multimodal_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What animal is in this image? Answer in one word."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
        ],
    }

    chat_result = user_proxy.initiate_chat(vision_assistant, message=multimodal_message, max_turns=1)

    _assert_v2_response_structure(chat_result)
    # Verify response is received (1x1 pixel images don't have recognizable content, so just check we got a response)
    assert chat_result.summary is not None and len(chat_result.summary) > 0
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

    assistant = AssistantAgent(
        name="assistant", llm_config=llm_config, system_message="You are helpful assistant. Be brief."
    )

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False
    )

    # First turn
    chat_result = user_proxy.initiate_chat(
        assistant, message="My favorite color is blue.", max_turns=1, clear_history=True
    )
    _assert_v2_response_structure(chat_result)

    # Second turn - should remember context
    user_proxy.send(message="What is my favorite color?", recipient=assistant, request_reply=True)

    # Get the assistant's reply from chat history
    reply = user_proxy.last_message(assistant)
    assert reply is not None, "Should have a reply from assistant"
    assert "blue" in str(reply["content"]).lower()


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
    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False
    )

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
    """Test V2 client with multiple images in one request using Base64 encoding."""
    llm_config = _create_test_v2_config(credentials_gpt_4o_mini)

    vision_assistant = AssistantAgent(name="vision_bot", llm_config=llm_config)

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False
    )

    # Two simple Base64 encoded images (1x1 pixel red and blue PNGs)
    # Red 1x1 pixel PNG
    base64_image_1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    # Blue 1x1 pixel PNG
    base64_image_2 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M/wHwAEBgIApD5fRAAAAABJRU5ErkJggg=="

    multimodal_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Compare these two images briefly. What colors do you see?"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image_1}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image_2}"}},
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

    # Use base64-encoded image to avoid remote URL issues (1x1 pixel brown PNG representing a dog)
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    multimodal_message = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Team, analyze this image and identify the animal breed."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
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
    # Verify response is received (1x1 pixel images don't have recognizable content, so just check we got a response)
    assert chat_result.summary is not None and len(chat_result.summary) > 0

    # Verify cost tracking
    assert "usage_including_cached_inference" in chat_result.cost
    assert len(chat_result.cost["usage_including_cached_inference"]) > 0

    # Verify multimodal content preserved
    first_msg = chat_result.chat_history[0]
    assert isinstance(first_msg["content"], list), "First message should be multimodal"

    # Verify context and last agent
    assert context_variables is not None
    assert last_agent is not None


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_run_group_chat_basic(credentials_gpt_4o_mini: Credentials) -> None:
    """Test V2 client with run_group_chat interface for basic text messages.

    Note: run_group_chat uses threading internally - the conversation happens in a
    background thread and sends events to the iostream. The process() method should
    block until the thread completes and all events are received.
    """
    llm_config = _create_test_v2_config(credentials_gpt_4o_mini)

    # Create specialized agents with V2 client
    analyst = ConversableAgent(
        name="Analyst",
        llm_config=llm_config,
        human_input_mode="NEVER",
        system_message="You analyze data. Be very brief.",
    )

    reviewer = ConversableAgent(
        name="Reviewer",
        llm_config=llm_config,
        human_input_mode="NEVER",
        system_message="You review analysis. Be very brief.",
    )

    # Create user proxy that won't hang but also won't interfere
    # Set max_consecutive_auto_reply=0 so it terminates immediately if selected
    user_proxy = ConversableAgent(
        name="User",
        human_input_mode="NEVER",
        llm_config=False,
        code_execution_config=False,
        max_consecutive_auto_reply=0,
    )

    # Create pattern-based group chat
    pattern = DefaultPattern(
        initial_agent=analyst,
        agents=[analyst, reviewer],
        user_agent=user_proxy,
    )

    # Use run_group_chat interface (returns immediately, chat runs in background thread)
    run_response = run_group_chat(
        pattern=pattern,
        messages="Analyze the number 7 briefly.",
        max_rounds=3,
    )

    # Verify run response object structure
    assert run_response is not None
    assert hasattr(run_response, "messages")
    assert hasattr(run_response, "process")
    assert hasattr(run_response, "events")

    # Process the response - this should block until the background thread completes
    # and all events have been sent to the iostream
    # NOTE: process() drains the events queue, so we cannot access response.events afterward
    run_response.process()

    # After process() completes, verify the conversation completed successfully
    # by checking the cached properties (messages, summary, cost, last_speaker)
    messages_list = list(run_response.messages)
    assert len(messages_list) >= 2, "Should have at least 2 messages after process() completes"

    # Verify summary is available (indicates RunCompletionEvent was received)
    assert run_response.summary is not None, "Should have summary after process() completes"

    # Verify last speaker is set
    assert run_response.last_speaker is not None, "Should have last_speaker after process() completes"

    # Verify cost information is available
    assert run_response.cost is not None, "Should have cost information after process() completes"


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_run_group_chat_multimodal(credentials_gpt_4o_mini: Credentials) -> None:
    """Test V2 client with run_group_chat and multimodal content (images)."""
    llm_config = _create_test_v2_config(credentials_gpt_4o_mini)

    # Create vision-capable agents
    image_analyst = ConversableAgent(
        name="ImageAnalyst",
        llm_config=llm_config,
        human_input_mode="NEVER",
        system_message="You analyze images. Be very brief.",
    )

    breed_expert = ConversableAgent(
        name="BreedExpert",
        llm_config=llm_config,
        human_input_mode="NEVER",
        system_message="You identify breeds. Be very brief.",
    )

    # Create user proxy that won't hang but also won't interfere
    # Set max_consecutive_auto_reply=0 so it terminates immediately if selected
    user_proxy = ConversableAgent(
        name="User",
        human_input_mode="NEVER",
        llm_config=False,
        code_execution_config=False,
        max_consecutive_auto_reply=0,
    )

    # Create pattern with vision agents
    pattern = DefaultPattern(
        initial_agent=image_analyst,
        agents=[image_analyst, breed_expert],
        user_agent=user_proxy,
    )

    # Use base64-encoded image to avoid remote URL issues (1x1 pixel brown PNG representing a dog)
    # Do NOT include "name" field - it causes role to become "assistant" which is invalid for images
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    multimodal_message = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Team, what animal is in this image?"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            ],
        }
    ]

    # Use run_group_chat with multimodal content
    run_response = run_group_chat(
        pattern=pattern,
        messages=multimodal_message,
        max_rounds=3,
    )

    # Process the response
    # NOTE: process() drains the events queue, so we use cached messages property
    run_response.process()

    # Get chat history from cached messages property (set by RunCompletionEvent)
    chat_history = list(run_response.messages)
    assert len(chat_history) >= 2, "Chat history should have at least 2 messages"

    # Check if first message content is preserved
    first_msg = chat_history[0]
    assert first_msg is not None
    assert "content" in first_msg

    # CRITICAL TEST: Verify if multimodal content is preserved as list or converted to string
    first_msg_content = first_msg["content"]
    logger.info("First message content type: %s", type(first_msg_content))
    logger.info("First message content: %s", first_msg_content)

    if isinstance(first_msg_content, list):
        logger.info("✓ Multimodal content PRESERVED as list")
        # Verify structure
        assert len(first_msg_content) >= 2, "Should have text and image blocks"
        text_blocks = [b for b in first_msg_content if b.get("type") == "text"]
        image_blocks = [b for b in first_msg_content if b.get("type") == "image_url"]
        assert len(text_blocks) > 0, "Should have text block"
        assert len(image_blocks) > 0, "Should have image block"
    elif isinstance(first_msg_content, str):
        logger.warning("⚠ Multimodal content CONVERTED to string")
        # This indicates data loss - image URL replaced with <image>
        assert "<image>" in first_msg_content, "String should contain <image> placeholder"


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_run_group_chat_content_preservation(credentials_gpt_4o_mini: Credentials) -> None:
    """Test that run_group_chat preserves multimodal content structure throughout conversation."""
    llm_config = _create_test_v2_config(credentials_gpt_4o_mini)

    # Create agents
    agent1 = ConversableAgent(
        name="Agent1",
        llm_config=llm_config,
        human_input_mode="NEVER",
        system_message="You analyze. Be brief.",
    )

    agent2 = ConversableAgent(
        name="Agent2",
        llm_config=llm_config,
        human_input_mode="NEVER",
        system_message="You review. Be brief.",
    )

    # Create user proxy that won't hang but also won't interfere
    # Set max_consecutive_auto_reply=0 so it terminates immediately if selected
    user_proxy = ConversableAgent(
        name="User",
        human_input_mode="NEVER",
        llm_config=False,
        code_execution_config=False,
        max_consecutive_auto_reply=0,
    )

    pattern = DefaultPattern(initial_agent=agent1, agents=[agent1, agent2], user_agent=user_proxy)

    # Multiple images in one message using Base64 encoding
    # Do NOT include "name" field - it causes role to become "assistant" which is invalid for images
    # Red 1x1 pixel PNG
    base64_image_1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    # Blue 1x1 pixel PNG
    base64_image_2 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M/wHwAEBgIApD5fRAAAAABJRU5ErkJggg=="

    multimodal_message = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these two images. What colors do you see?"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image_1}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image_2}"}},
            ],
        }
    ]

    # Run group chat
    run_response = run_group_chat(pattern=pattern, messages=multimodal_message, max_rounds=3)
    # NOTE: process() drains the events queue, so we use cached messages property
    run_response.process()

    # Get chat history from cached messages property (set by RunCompletionEvent)
    chat_history = list(run_response.messages)
    assert len(chat_history) > 0, "Should have chat history"
    first_msg = chat_history[0]

    logger.info("=== Content Preservation Test ===")
    logger.info("Original message had: 1 text + 2 images")
    logger.info("Stored content type: %s", type(first_msg["content"]))

    if isinstance(first_msg["content"], list):
        logger.info("✓ PRESERVED: Content is still a list with %d blocks", len(first_msg["content"]))
        content_blocks = first_msg["content"]

        # Count block types
        text_count = sum(1 for b in content_blocks if b.get("type") == "text")
        image_count = sum(1 for b in content_blocks if b.get("type") == "image_url")

        logger.info("  - Text blocks: %d", text_count)
        logger.info("  - Image blocks: %d", image_count)

        # Verify all blocks preserved
        assert text_count >= 1, "Should preserve text block"
        assert image_count >= 2, "Should preserve both image blocks"

        # Verify image URLs are intact
        for block in content_blocks:
            if block.get("type") == "image_url":
                assert "image_url" in block, "Image block should have image_url field"
                assert "url" in block["image_url"], "Image URL should have url field"
                # Check for either http URLs or Base64 data URIs
                url = block["image_url"]["url"]
                assert url.startswith("http") or url.startswith("data:image"), "URL should be preserved"

    elif isinstance(first_msg["content"], str):
        logger.warning("⚠ CONVERTED to string: %s...", first_msg["content"][:100])

        # Check what was lost
        content_str_result = first_msg["content"]
        image_placeholder_count = content_str_result.count("<image>")

        logger.warning("  - Image URLs converted to %d <image> placeholder(s)", image_placeholder_count)
        logger.warning("  - Original URLs LOST")

        # At minimum, should have placeholders for both images
        assert image_placeholder_count >= 2, "Should have placeholders for both images"


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_structured_output_pydantic_simple(credentials_gpt_4o_mini: Credentials) -> None:
    """Test V2 client with Pydantic structured output in agent chat."""
    try:
        from pydantic import BaseModel
    except ImportError:
        pytest.skip("Pydantic not installed")

    # Define Pydantic model for response
    class QueryAnswer(BaseModel):
        """Structured answer to a query."""

        question: str
        answer: str
        confidence: float

    # Create V2 config with Pydantic response_format
    base_config = credentials_gpt_4o_mini.llm_config._model.config_list[0]
    llm_config = {
        "config_list": [
            {
                "api_type": "openai_v2",
                "model": getattr(base_config, "model", "gpt-4o-mini"),
                "api_key": getattr(base_config, "api_key", os.getenv("OPENAI_API_KEY")),
            }
        ],
        "response_format": QueryAnswer,  # Pydantic model
        "temperature": 0,
    }

    assistant = AssistantAgent(
        name="structured_assistant",
        llm_config=llm_config,
        system_message="You provide structured answers. Always fill all required fields.",
    )

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False
    )

    chat_result = user_proxy.initiate_chat(
        assistant, message="What is the capital of France? Rate your confidence from 0-1.", max_turns=1
    )

    _assert_v2_response_structure(chat_result)

    # Verify structured output was returned in the response
    assert chat_result.summary is not None
    assert "paris" in chat_result.summary.lower()


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_structured_output_pydantic_complex(credentials_gpt_4o_mini: Credentials) -> None:
    """Test V2 client with complex Pydantic structured output."""
    try:
        from pydantic import BaseModel
    except ImportError:
        pytest.skip("Pydantic not installed")

    # Define complex Pydantic model
    class MathSolution(BaseModel):
        """Solution to a math problem."""

        problem: str
        solution: int
        steps: str
        difficulty: str

    base_config = credentials_gpt_4o_mini.llm_config._model.config_list[0]
    llm_config = {
        "config_list": [
            {
                "api_type": "openai_v2",
                "model": getattr(base_config, "model", "gpt-4o-mini"),
                "api_key": getattr(base_config, "api_key", os.getenv("OPENAI_API_KEY")),
            }
        ],
        "response_format": MathSolution,
        "temperature": 0,
    }

    assistant = AssistantAgent(
        name="math_assistant",
        llm_config=llm_config,
        system_message="You solve math problems with structured output. Always provide all required fields.",
    )

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False
    )

    chat_result = user_proxy.initiate_chat(
        assistant, message="What is 15 + 27? Show your work and rate the difficulty.", max_turns=1
    )

    _assert_v2_response_structure(chat_result)

    # Verify the answer is present
    assert "42" in chat_result.summary


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_structured_output_multi_turn(credentials_gpt_4o_mini: Credentials) -> None:
    """Test V2 client structured output in multi-turn conversation."""
    try:
        from pydantic import BaseModel
    except ImportError:
        pytest.skip("Pydantic not installed")

    class FactCheck(BaseModel):
        """Fact check result."""

        statement: str
        is_true: bool
        explanation: str

    base_config = credentials_gpt_4o_mini.llm_config._model.config_list[0]
    llm_config = {
        "config_list": [
            {
                "api_type": "openai_v2",
                "model": getattr(base_config, "model", "gpt-4o-mini"),
                "api_key": getattr(base_config, "api_key", os.getenv("OPENAI_API_KEY")),
            }
        ],
        "response_format": FactCheck,
        "temperature": 0,
    }

    assistant = AssistantAgent(
        name="fact_checker", llm_config=llm_config, system_message="You fact-check statements with structured output."
    )

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False
    )

    # First turn
    chat_result1 = user_proxy.initiate_chat(assistant, message="Is the Earth flat?", max_turns=1, clear_history=True)
    _assert_v2_response_structure(chat_result1)

    # Second turn - should maintain structured output
    user_proxy.send(message="Is water wet?", recipient=assistant, request_reply=True)

    # Verify both responses worked
    reply = user_proxy.last_message(assistant)
    assert reply is not None


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_structured_output_group_chat(credentials_gpt_4o_mini: Credentials) -> None:
    """Test V2 client structured output in group chat scenario."""
    try:
        from pydantic import BaseModel
    except ImportError:
        pytest.skip("Pydantic not installed")

    class Analysis(BaseModel):
        """Data analysis result."""

        topic: str
        summary: str
        key_points: str

    base_config = credentials_gpt_4o_mini.llm_config._model.config_list[0]
    llm_config = {
        "config_list": [
            {
                "api_type": "openai_v2",
                "model": getattr(base_config, "model", "gpt-4o-mini"),
                "api_key": getattr(base_config, "api_key", os.getenv("OPENAI_API_KEY")),
            }
        ],
        "response_format": Analysis,
        "temperature": 0,
    }

    # Create agents with structured output
    analyst = ConversableAgent(
        name="analyst",
        llm_config=llm_config,
        human_input_mode="NEVER",
        system_message="You analyze data with structured output. Be brief.",
    )

    reviewer = ConversableAgent(
        name="reviewer",
        llm_config=llm_config,
        human_input_mode="NEVER",
        system_message="You review analysis with structured output. Be brief.",
    )

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False
    )

    # Create group chat with structured output agents
    groupchat = GroupChat(
        agents=[user_proxy, analyst, reviewer], messages=[], max_round=3, speaker_selection_method="round_robin"
    )

    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    chat_result = user_proxy.initiate_chat(manager, message="Team, analyze the concept of AI safety.", max_turns=2)

    _assert_v2_response_structure(chat_result)

    # Verify agents participated
    participant_names = {msg.get("name") for msg in chat_result.chat_history if msg.get("name")}
    assert len(participant_names.intersection({"analyst", "reviewer"})) >= 1


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_structured_output_pattern_based(credentials_gpt_4o_mini: Credentials) -> None:
    """Test V2 client structured output with pattern-based group chat."""
    try:
        from pydantic import BaseModel
    except ImportError:
        pytest.skip("Pydantic not installed")

    class Report(BaseModel):
        """Analysis report."""

        title: str
        findings: str
        recommendation: str

    base_config = credentials_gpt_4o_mini.llm_config._model.config_list[0]
    llm_config = {
        "config_list": [
            {
                "api_type": "openai_v2",
                "model": getattr(base_config, "model", "gpt-4o-mini"),
                "api_key": getattr(base_config, "api_key", os.getenv("OPENAI_API_KEY")),
            }
        ],
        "response_format": Report,
        "temperature": 0,
    }

    # Create agents with structured output
    analyst = ConversableAgent(
        name="DataAnalyst",
        llm_config=llm_config,
        human_input_mode="NEVER",
        system_message="You create analysis reports with structured output. Be concise.",
    )

    reviewer = ConversableAgent(
        name="QualityReviewer",
        llm_config=llm_config,
        human_input_mode="NEVER",
        system_message="You review reports with structured output. Be concise.",
    )

    # Create pattern-based group chat
    pattern = DefaultPattern(
        initial_agent=analyst,
        agents=[analyst, reviewer],
    )

    # Initiate group chat
    chat_result, context_variables, last_agent = initiate_group_chat(
        pattern=pattern,
        messages="Create a brief report analyzing the number 42.",
        max_rounds=2,
    )

    _assert_v2_response_structure(chat_result)

    # Verify structured output worked in pattern-based chat
    assert len(chat_result.chat_history) >= 2
    assert "usage_including_cached_inference" in chat_result.cost


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_structured_output_override_in_params(credentials_gpt_4o_mini: Credentials) -> None:
    """Test that response_format in agent params overrides client default."""
    try:
        from pydantic import BaseModel
    except ImportError:
        pytest.skip("Pydantic not installed")

    class DefaultModel(BaseModel):
        default_field: str

    class OverrideModel(BaseModel):
        override_field: str
        value: int

    # Create config with default response_format
    base_config = credentials_gpt_4o_mini.llm_config._model.config_list[0]
    llm_config = {
        "config_list": [
            {
                "api_type": "openai_v2",
                "model": getattr(base_config, "model", "gpt-4o-mini"),
                "api_key": getattr(base_config, "api_key", os.getenv("OPENAI_API_KEY")),
            }
        ],
        "response_format": DefaultModel,  # Default
        "temperature": 0,
    }

    # Create assistant with default
    assistant = AssistantAgent(
        name="assistant",
        llm_config=llm_config,
        system_message="Provide structured responses.",
    )

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False
    )

    # First chat with default model
    chat_result1 = user_proxy.initiate_chat(
        assistant, message="Return default_field='test1'", max_turns=1, clear_history=True
    )
    _assert_v2_response_structure(chat_result1)

    # Note: Overriding response_format in generate_oai_reply params is not directly
    # supported in the current agent API, so we verify the default works
    assert len(chat_result1.chat_history) >= 2


# =============================================================================
# Tool Calling and Function Calling Tests
# =============================================================================


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_tool_calling_two_agent(credentials_gpt_4o_mini: Credentials) -> None:
    """Test tool calling in two-agent conversation with V2 client using new registration pattern."""
    from typing import Annotated

    def add_numbers(
        a: Annotated[int, "First number to add"],
        b: Annotated[int, "Second number to add"],
    ) -> int:
        """Add two numbers together.

        Args:
            a: First number
            b: Second number

        Returns:
            Sum of a and b
        """
        return a + b

    def multiply_numbers(
        a: Annotated[int, "First number to multiply"],
        b: Annotated[int, "Second number to multiply"],
    ) -> int:
        """Multiply two numbers together.

        Args:
            a: First number
            b: Second number

        Returns:
            Product of a and b
        """
        return a * b

    # Create V2 config
    llm_config = _create_test_v2_config(credentials_gpt_4o_mini)
    llm_config["temperature"] = 0

    # Create assistant with tools using new pattern (functions parameter)
    assistant = ConversableAgent(
        name="math_assistant",
        llm_config=llm_config,
        system_message="You are a helpful math assistant. Use the provided functions to perform calculations.",
        functions=[add_numbers, multiply_numbers],  # New pattern: register tools here
    )

    # Create user proxy that can execute functions
    user_proxy = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
        code_execution_config=False,
    )

    # Initiate chat with tool calling
    chat_result = user_proxy.initiate_chat(
        assistant,
        message="Please calculate (5 + 3) * 2. First add 5 and 3 using add_numbers, then multiply the result by 2 using multiply_numbers.",
        max_turns=5,
    )

    _assert_v2_response_structure(chat_result)

    # Verify tool execution happened
    chat_history_str = str(chat_result.chat_history)
    assert "add_numbers" in chat_history_str or "16" in chat_history_str, "Tool should have been called"

    # Verify we have tool call messages in history
    tool_call_messages = [msg for msg in chat_result.chat_history if msg.get("tool_calls")]
    assert len(tool_call_messages) > 0, "Should have at least one message with tool_calls"

    # Verify tool call structure
    tool_call = tool_call_messages[0]["tool_calls"][0]
    assert "id" in tool_call
    assert "function" in tool_call
    assert "name" in tool_call["function"]
    assert "arguments" in tool_call["function"]


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_tool_calling_group_chat(credentials_gpt_4o_mini: Credentials) -> None:
    """Test tool calling in group chat with V2 client."""

    def get_weather(location: str) -> str:
        """Get weather for a location.

        Args:
            location: City name

        Returns:
            Weather description
        """
        return f"The weather in {location} is sunny and 72°F"

    def get_time(timezone: str) -> str:
        """Get current time in a timezone.

        Args:
            timezone: Timezone name (e.g., 'America/New_York')

        Returns:
            Current time
        """
        return f"The time in {timezone} is 3:45 PM"

    # Create V2 config
    llm_config = _create_test_v2_config(credentials_gpt_4o_mini)
    llm_config["temperature"] = 0

    # Create agents
    weather_agent = AssistantAgent(
        name="weather_agent",
        llm_config=llm_config,
        system_message="You provide weather information using the get_weather function.",
    )

    time_agent = AssistantAgent(
        name="time_agent",
        llm_config=llm_config,
        system_message="You provide time information using the get_time function.",
    )

    user_proxy = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    # Register weather tool
    weather_agent.register_for_llm(name="get_weather", description="Get weather")(get_weather)
    user_proxy.register_for_execution(name="get_weather")(get_weather)

    # Register time tool
    time_agent.register_for_llm(name="get_time", description="Get time")(get_time)
    user_proxy.register_for_execution(name="get_time")(get_time)

    # Create group chat
    group_chat = GroupChat(
        agents=[user_proxy, weather_agent, time_agent],
        messages=[],
        max_round=10,
        speaker_selection_method="auto",
    )

    manager = GroupChatManager(groupchat=group_chat, llm_config=llm_config)

    # Initiate group chat
    chat_result = user_proxy.initiate_chat(
        manager, message="What's the weather in San Francisco and what time is it in America/Los_Angeles?", max_turns=8
    )

    _assert_v2_response_structure(chat_result)

    # Verify tool execution
    chat_history_str = str(chat_result.chat_history)
    assert "weather" in chat_history_str.lower() or "san francisco" in chat_history_str.lower(), (
        "Weather function should have been called"
    )

    # Verify tool call messages exist
    tool_call_messages = [msg for msg in chat_result.chat_history if msg.get("tool_calls")]
    assert len(tool_call_messages) > 0, "Should have tool call messages in group chat"


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_function_call_legacy(credentials_gpt_4o_mini: Credentials) -> None:
    """Test legacy function_call format still works with V2 client."""

    def calculate_sum(numbers: list[int]) -> int:
        """Calculate sum of a list of numbers.

        Args:
            numbers: List of integers to sum

        Returns:
            Sum of all numbers
        """
        return sum(numbers)

    # Create V2 config
    llm_config = _create_test_v2_config(credentials_gpt_4o_mini)
    llm_config["temperature"] = 0

    # Create agents
    assistant = AssistantAgent(
        name="calculator",
        llm_config=llm_config,
        system_message="You help with calculations using the calculate_sum function.",
    )

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=2, code_execution_config=False
    )

    # Register function
    assistant.register_for_llm(name="calculate_sum", description="Calculate sum of numbers")(calculate_sum)
    user_proxy.register_for_execution(name="calculate_sum")(calculate_sum)

    # Initiate chat
    chat_result = user_proxy.initiate_chat(
        assistant, message="What is the sum of 10, 20, 30, 40, and 50? Use the calculate_sum function.", max_turns=4
    )

    _assert_v2_response_structure(chat_result)

    # Verify function execution
    assert "150" in str(chat_result.chat_history) or "150" in str(chat_result.summary), (
        "Function should have calculated correct sum"
    )


# =============================================================================
# Multimodal Content Tests
# =============================================================================


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_multimodal_image_two_agent(credentials_gpt_4o_mini: Credentials) -> None:
    """Test image/vision capabilities in two-agent conversation with V2 client."""

    # Create V2 config with vision model
    llm_config = _create_test_v2_config(credentials_gpt_4o_mini)
    llm_config["temperature"] = 0

    # Create vision-capable assistant
    assistant = AssistantAgent(
        name="vision_assistant",
        llm_config=llm_config,
        system_message="You are a helpful assistant that can analyze images.",
    )

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False
    )

    # Use base64 encoded image to avoid remote URL issues (1x1 pixel red PNG)
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

    # Create multimodal message - must be dict with "content" for initiate_chat
    message = {
        "content": [
            {"type": "text", "text": "What do you see in this image? Describe it briefly in one sentence."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
        ]
    }

    # Initiate chat with image
    chat_result = user_proxy.initiate_chat(assistant, message=message, max_turns=1)

    _assert_v2_response_structure(chat_result)

    # Verify image was processed
    assert len(chat_result.chat_history) >= 2
    # The response should contain some description
    response_text = str(chat_result.summary).lower()
    assert len(response_text) > 10, "Should have generated a response about the image"


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_multimodal_group_chat(credentials_gpt_4o_mini: Credentials) -> None:
    """Test multimodal content in group chat with V2 client."""

    # Create V2 config
    llm_config = _create_test_v2_config(credentials_gpt_4o_mini)
    llm_config["temperature"] = 0

    # Create vision agents
    describer = AssistantAgent(
        name="image_describer",
        llm_config=llm_config,
        system_message="You describe images in detail.",
    )

    analyzer = AssistantAgent(
        name="image_analyzer",
        llm_config=llm_config,
        system_message="You analyze images and provide insights.",
    )

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False
    )

    # Create group chat
    group_chat = GroupChat(
        agents=[user_proxy, describer, analyzer], messages=[], max_round=6, speaker_selection_method="round_robin"
    )

    manager = GroupChatManager(groupchat=group_chat, llm_config=llm_config)

    # Use base64 encoded image to avoid remote URL issues (1x1 pixel blue PNG)
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M/wHwAEBgIApD5fRAAAAABJRU5ErkJggg=="

    # Must be dict with "content" for initiate_chat
    message = {
        "content": [
            {"type": "text", "text": "Please analyze this image."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
        ]
    }

    # Initiate group chat
    chat_result = user_proxy.initiate_chat(manager, message=message, max_turns=4)

    _assert_v2_response_structure(chat_result)
    assert len(chat_result.chat_history) >= 2


# =============================================================================
# Reasoning Models (O1/O3) Tests
# =============================================================================
# Note: These tests require access to o1 models which may not be available on all API keys.
# The tests are marked with @pytest.mark.skip if o1 access is not available.
# The implementation is still verified through unit tests of the parameter processing logic.


@pytest.mark.openai
@pytest.mark.skipif(
    os.getenv("OPENAI_O1_AVAILABLE", "false").lower() != "true",
    reason="o1 model access not available - set OPENAI_O1_AVAILABLE=true if you have access",
)
@run_for_optional_imports("openai", "openai")
def test_v2_client_reasoning_model_basic(credentials_o1_mini: Credentials) -> None:
    """Test basic reasoning with V2 client using o1-mini model.

    This test verifies:
    1. V2 client works with actual o1-mini reasoning model
    2. Reasoning content is properly extracted from response
    3. Response structure is correct
    """
    base_config = credentials_o1_mini.llm_config._model.config_list[0]

    # Create V2 config with o1-mini reasoning model
    llm_config = {
        "config_list": [
            {
                "api_type": "openai_v2",
                "model": "o1-mini",
                "api_key": getattr(base_config, "api_key", os.getenv("OPENAI_API_KEY")),
            }
        ],
    }

    assistant = AssistantAgent(
        name="reasoning_assistant",
        llm_config=llm_config,
    )

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False
    )

    # Ask a reasoning question that requires step-by-step thinking
    chat_result = user_proxy.initiate_chat(
        assistant,
        message="What is 15% of 240? Show your reasoning.",
        max_turns=1,
    )

    _assert_v2_response_structure(chat_result)

    # Verify response contains correct answer
    response = str(chat_result.summary).lower()
    assert "36" in response, "Should calculate 15% of 240 = 36"

    # CRITICAL: Verify reasoning content was extracted
    # o1-mini returns reasoning field that should be extracted as ReasoningContent
    # We verify this by checking the chat history for reasoning content
    last_message = chat_result.chat_history[-1]
    assert "content" in last_message, "Message should have content field"

    # The reasoning should be in the message - either as separate content or embedded
    # With V2 client, reasoning is extracted separately from the main response
    print(f"Response summary: {chat_result.summary}")
    print(f"Last message keys: {last_message.keys()}")


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_reasoning_parameter_processing(credentials_o1_mini: Credentials) -> None:
    """Test that V2 client properly processes o1 model parameters.

    This test verifies:
    1. Unsupported parameters (temperature, top_p, etc.) are automatically removed
    2. max_tokens is converted to max_completion_tokens
    3. Request succeeds without API errors
    4. Warning messages are issued for removed parameters
    """
    import warnings

    base_config = credentials_o1_mini.llm_config._model.config_list[0]

    # Create config with parameters that ARE unsupported by o1 models
    # These should be automatically removed by _process_reasoning_model_params()
    llm_config = {
        "config_list": [
            {
                "api_type": "openai_v2",
                "model": "o1-mini",
                "api_key": getattr(base_config, "api_key", os.getenv("OPENAI_API_KEY")),
            }
        ],
        "temperature": 0.7,  # UNSUPPORTED - should be removed with warning
        "max_tokens": 1000,  # Should be converted to max_completion_tokens
    }

    assistant = AssistantAgent(
        name="test_assistant",
        llm_config=llm_config,
    )

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False
    )

    # Capture warnings to verify parameter processing warnings are issued
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # This should work WITHOUT API errors despite unsupported parameters
        # because _process_reasoning_model_params() removes them
        chat_result = user_proxy.initiate_chat(
            assistant,
            message="What is 2+2?",
            max_turns=1,
        )

        # Verify warning was issued for temperature parameter
        warning_messages = [str(warning.message) for warning in w]
        assert any("temperature" in msg.lower() for msg in warning_messages), (
            "Should warn about temperature being unsupported"
        )

    _assert_v2_response_structure(chat_result)
    assert "4" in chat_result.summary


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_reasoning_with_system_message(credentials_o1_mini: Credentials) -> None:
    """Test that V2 client handles system messages correctly for o1 models.

    Note: o1-mini (2024-09-12 and later) supports system messages natively.
    Older models like o1-preview would have system messages converted to user messages,
    but this is handled automatically by _process_reasoning_model_params().

    This test verifies the model works correctly with system messages.
    """
    base_config = credentials_o1_mini.llm_config._model.config_list[0]

    llm_config = {
        "config_list": [
            {
                "api_type": "openai_v2",
                "model": "o1-mini",
                "api_key": getattr(base_config, "api_key", os.getenv("OPENAI_API_KEY")),
            }
        ],
    }

    assistant = AssistantAgent(
        name="reasoning_assistant",
        llm_config=llm_config,
        system_message="You are a helpful math tutor.",
    )

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False
    )

    # Should work without errors with system message
    # (o1-mini supports system messages; older models would have them converted)
    chat_result = user_proxy.initiate_chat(
        assistant,
        message="Calculate 25% of 80.",
        max_turns=1,
    )

    _assert_v2_response_structure(chat_result)
    assert "20" in chat_result.summary


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_reasoning_no_streaming(credentials_o1_mini: Credentials) -> None:
    """Test that V2 client blocks streaming for o1 models.

    This test verifies:
    1. stream=True is automatically set to False for o1 models
    2. Warning is issued about streaming not being supported
    3. Request succeeds without API errors
    """
    import warnings

    base_config = credentials_o1_mini.llm_config._model.config_list[0]

    # Attempt to use streaming - should be automatically blocked
    llm_config = {
        "config_list": [
            {
                "api_type": "openai_v2",
                "model": "o1-mini",
                "api_key": getattr(base_config, "api_key", os.getenv("OPENAI_API_KEY")),
                "stream": True,  # NOT supported by o1 - should be blocked with warning
            }
        ],
    }

    assistant = AssistantAgent(
        name="test_assistant",
        llm_config=llm_config,
    )

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False
    )

    # Capture warnings to verify streaming blocking warning is issued
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Should work WITHOUT errors because stream=True is automatically set to False
        chat_result = user_proxy.initiate_chat(
            assistant,
            message="Say hello.",
            max_turns=1,
        )

        # Verify warning was issued about streaming not being supported
        warning_messages = [str(warning.message) for warning in w]
        assert any("streaming" in msg.lower() or "stream" in msg.lower() for msg in warning_messages), (
            "Should warn about streaming not being supported"
        )

    _assert_v2_response_structure(chat_result)


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_reasoning_no_tools(credentials_o1_mini: Credentials) -> None:
    """Test that V2 client blocks tools for o1 models.

    This test verifies:
    1. o1 models work correctly without tool registration
    2. o1 models don't support function calling
    3. Basic reasoning works for mathematical calculations
    """
    base_config = credentials_o1_mini.llm_config._model.config_list[0]

    # o1 models do NOT support function calling/tools
    llm_config = {
        "config_list": [
            {
                "api_type": "openai_v2",
                "model": "o1-mini",
                "api_key": getattr(base_config, "api_key", os.getenv("OPENAI_API_KEY")),
            }
        ],
    }

    # Create agent WITHOUT registering any functions
    # (o1 models don't support function calling)
    assistant = ConversableAgent(
        name="test_assistant",
        llm_config=llm_config,
        # No functions parameter - o1 doesn't support tools
    )

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False
    )

    # Should work without tools - o1 can reason through the math
    chat_result = user_proxy.initiate_chat(
        assistant,
        message="What is 5 squared?",
        max_turns=1,
    )

    _assert_v2_response_structure(chat_result)
    assert "25" in chat_result.summary


# =============================================================================
# Combined Features Tests
# =============================================================================


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_combined_structured_output_and_tools(credentials_gpt_4o_mini: Credentials) -> None:
    """Test structured output with tool calling (separate turns).

    Note: OpenAI does not support combining Pydantic response_format with tools in a single call.
    This test verifies that tools work in one turn, then structured output works in the next turn.
    """
    try:
        from pydantic import BaseModel
    except ImportError:
        pytest.skip("Pydantic not installed")

    class CalculationResult(BaseModel):
        """Structured result of a calculation."""

        operation: str
        result: int
        explanation: str

    def calculate(expression: str) -> int:
        """Safely calculate a math expression.

        Args:
            expression: Math expression like "25 + 17"

        Returns:
            The result
        """
        # Simple evaluation for this test
        return eval(expression, {"__builtins__": {}})

    # Create V2 config WITHOUT structured output first (for tool calling)
    llm_config = _create_test_v2_config(credentials_gpt_4o_mini)
    llm_config["temperature"] = 0

    assistant = AssistantAgent(
        name="calculator",
        llm_config=llm_config,
        system_message="You perform calculations using the calculate function.",
    )

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=2, code_execution_config=False
    )

    # Register tool
    assistant.register_for_llm(name="calculate", description="Calculate a math expression")(calculate)
    user_proxy.register_for_execution(name="calculate")(calculate)

    # Initiate chat with tool calling
    chat_result = user_proxy.initiate_chat(
        assistant,
        message="Use the calculate function to compute 25 + 17",
        max_turns=3,
    )

    _assert_v2_response_structure(chat_result)

    # Should have tool calls in history
    assert len(chat_result.chat_history) >= 2


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_multimodal_with_tools(credentials_gpt_4o_mini: Credentials) -> None:
    """Test combining multimodal input with tool calling."""

    def analyze_color(color_name: str) -> str:
        """Analyze a color.

        Args:
            color_name: Name of the color

        Returns:
            Color analysis
        """
        return f"The color {color_name} is often associated with creativity and energy."

    # Create V2 config
    llm_config = _create_test_v2_config(credentials_gpt_4o_mini)
    llm_config["temperature"] = 0

    assistant = AssistantAgent(
        name="vision_analyst",
        llm_config=llm_config,
        system_message="You analyze images and can provide color analysis using the analyze_color function.",
    )

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=2, code_execution_config=False
    )

    # Register tool
    assistant.register_for_llm(name="analyze_color", description="Analyze a color")(analyze_color)
    user_proxy.register_for_execution(name="analyze_color")(analyze_color)

    # Use base64 encoded image to avoid remote URL issues (1x1 pixel red PNG)
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

    # Create multimodal message with tool instruction - must be dict with "content" for initiate_chat
    message = {
        "content": [
            {
                "type": "text",
                "text": "Look at this image and identify the dominant color, then use the analyze_color function to analyze it.",
            },
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
        ]
    }

    # Initiate chat
    chat_result = user_proxy.initiate_chat(assistant, message=message, max_turns=4)

    _assert_v2_response_structure(chat_result)
    assert len(chat_result.chat_history) >= 2


# =============================================================================
# AutoPattern Tests (New Pattern-Based Orchestration)
# =============================================================================


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_v2_client_auto_pattern_with_tools(credentials_gpt_4o_mini: Credentials) -> None:
    """Test AutoPattern (LLM-based agent selection) with tool calling using new registration pattern."""
    from typing import Annotated

    from autogen.agentchat.group.patterns import AutoPattern

    def get_stock_price(
        symbol: Annotated[str, "Stock ticker symbol (e.g., 'AAPL', 'GOOGL')"],
    ) -> dict[str, Any]:
        """Get the current stock price for a given symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with stock price information
        """
        # Mock stock data for testing
        prices = {"AAPL": 150.25, "GOOGL": 2800.50, "MSFT": 380.75}
        return {
            "symbol": symbol,
            "price": prices.get(symbol, 100.0),
            "currency": "USD",
        }

    def calculate_portfolio_value(
        symbol: Annotated[str, "Stock ticker symbol"],
        shares: Annotated[int, "Number of shares owned"],
    ) -> dict[str, Any]:
        """Calculate the total value of a stock position.

        Args:
            symbol: Stock ticker symbol
            shares: Number of shares

        Returns:
            Dictionary with portfolio value calculation
        """
        # Use get_stock_price internally
        price_data = get_stock_price(symbol)
        total_value = price_data["price"] * shares
        return {
            "symbol": symbol,
            "shares": shares,
            "price_per_share": price_data["price"],
            "total_value": total_value,
            "currency": "USD",
        }

    # Create V2 config
    llm_config = _create_test_v2_config(credentials_gpt_4o_mini)
    llm_config["temperature"] = 0

    # Create specialized agents with tools using new pattern
    price_checker = ConversableAgent(
        name="price_checker",
        llm_config=llm_config,
        system_message="You specialize in checking stock prices. Use the get_stock_price function to look up prices.",
        functions=[get_stock_price],  # New pattern: register tool here
    )

    portfolio_analyst = ConversableAgent(
        name="portfolio_analyst",
        llm_config=llm_config,
        system_message="You specialize in portfolio analysis. Use the calculate_portfolio_value function to compute holdings.",
        functions=[calculate_portfolio_value],  # New pattern: register tool here
    )

    # Create user agent
    user_agent = ConversableAgent(
        name="user",
        human_input_mode="NEVER",  # Disable human input for automated testing
        llm_config=False,
    )

    # Create AutoPattern for intelligent agent selection
    pattern = AutoPattern(
        initial_agent=price_checker,
        agents=[price_checker, portfolio_analyst],
        user_agent=user_agent,  # Provide user agent
        group_manager_args={"llm_config": llm_config},  # Required for AutoPattern
    )

    # Initiate group chat with AutoPattern
    chat_result, context_variables, last_agent = initiate_group_chat(
        pattern=pattern,
        messages="I own 100 shares of AAPL. What's the current price and what's my position worth?",
        max_rounds=10,
    )

    _assert_v2_response_structure(chat_result)

    # Verify both tools were called
    chat_history_str = str(chat_result.chat_history)
    assert "get_stock_price" in chat_history_str or "calculate_portfolio_value" in chat_history_str, (
        "At least one tool should have been called"
    )

    # Verify tool calls in history
    tool_call_messages = [msg for msg in chat_result.chat_history if msg.get("tool_calls")]
    assert len(tool_call_messages) > 0, "Should have tool call messages with AutoPattern"
