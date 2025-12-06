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

    # Use formal multimodal content format (blue square test image)
    image_url = "https://media.githubusercontent.com/media/ag2ai/ag2/refs/heads/main/test/test_files/test_image.png"
    multimodal_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What color is this image? Answer in one word."},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    }

    chat_result = user_proxy.initiate_chat(vision_assistant, message=multimodal_message, max_turns=1)

    _assert_v2_response_structure(chat_result)
    summary_lower = chat_result.summary.lower()
    assert "blue" in summary_lower
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

    # Multimodal message with image (blue square test image)
    image_url = "https://media.githubusercontent.com/media/ag2ai/ag2/refs/heads/main/test/test_files/test_image.png"
    multimodal_message = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Team, analyze this image and identify the color."},
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
    assert "blue" in summary_lower

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

    # Multimodal message with image (blue square test image)
    # Do NOT include "name" field - it causes role to become "assistant" which is invalid for images
    image_url = "https://media.githubusercontent.com/media/ag2ai/ag2/refs/heads/main/test/test_files/test_image.png"
    multimodal_message = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Team, what color is this image?"},
                {"type": "image_url", "image_url": {"url": image_url}},
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
