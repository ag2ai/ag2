# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

from typing import Any

import pytest

from autogen import ConversableAgent, UserProxyAgent
from autogen.agentchat.groupchat import GroupChat, GroupChatManager
from autogen.code_utils import content_str
from autogen.import_utils import run_for_optional_imports
from test.credentials import Credentials


def _assert_multimodal_content_handling(message: dict[str, Any]) -> None:
    """Verify that multimodal content is properly handled."""
    content = message.get("content")

    # Content should be either string or list
    assert content is not None, "Message content should not be None"

    if isinstance(content, list):
        # Verify list structure for multimodal content
        assert len(content) > 0, "Multimodal content list should not be empty"
        for item in content:
            assert isinstance(item, dict), "Each multimodal content item should be a dict"
            assert "type" in item, "Each multimodal item should have a 'type' field"

            if item["type"] == "text":
                assert "text" in item, "Text items should have 'text' field"
                assert isinstance(item["text"], str), "Text field should be string"
            elif item["type"] in ["image_url", "input_image"]:
                assert "image_url" in item, "Image items should have 'image_url' field"
                assert isinstance(item["image_url"], (dict, str)), "image_url should be dict or string"
                if isinstance(item["image_url"], dict):
                    assert "url" in item["image_url"], "image_url dict should have 'url' field"
    else:
        # String content should be valid
        assert isinstance(content, str), "Non-list content should be string"
        assert len(content) > 0, "String content should not be empty"


def _create_test_multimodal_content() -> list[dict[str, Any]]:
    """Create test multimodal content for Chat Completion API integration tests."""
    # Using the format compatible with OpenAI Chat Completion API
    return [
        {"type": "text", "text": "Analyze this data visualization:"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://media.githubusercontent.com/media/ag2ai/ag2/refs/heads/main/test/test_files/test_image.png"
            },
        },
    ]


def _create_test_multimodal_content_responses_api() -> list[dict[str, Any]]:
    """Create test multimodal content for Responses API integration tests."""
    # Using the format compatible with OpenAI Responses API
    return [
        {"type": "text", "text": "Analyze this data visualization:"},
        {
            "type": "input_image",
            "image_url": "https://media.githubusercontent.com/media/ag2ai/ag2/refs/heads/main/test/test_files/test_image.png",
        },
    ]


def _verify_content_str_processing(content: Any) -> None:
    """Verify content_str can process the content properly."""
    if isinstance(content, list):
        # Test that content_str can handle multimodal content
        result = content_str(content)
        assert isinstance(result, str), "content_str should return string"
        assert len(result) > 0, "content_str result should not be empty"

        # Should contain text parts
        text_parts = [item for item in content if item.get("type") == "text"]
        for text_item in text_parts:
            if text_item.get("text"):
                assert text_item["text"] in result, "Text content should be preserved"

        # Should contain image placeholders for image content
        image_parts = [item for item in content if item.get("type") in ["image_url", "input_image"]]
        if image_parts:
            assert "<image>" in result, "Image content should be converted to <image> placeholder"


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_conversable_agent_multimodal_message_handling(credentials_responses_gpt_4o_mini: Credentials) -> None:
    """Test ConversableAgent can handle multimodal content in real conversations."""

    # Create agents with actual LLM configuration
    assistant = ConversableAgent(
        name="multimodal_assistant",
        llm_config=credentials_responses_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        system_message="You are helpful assistant that can process text and images.",
    )

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", code_execution_config=False, max_consecutive_auto_reply=1
    )

    # Create multimodal message
    multimodal_message = _create_test_multimodal_content_responses_api()

    # Test that the message can be processed
    chat_result = user_proxy.initiate_chat(assistant, message={"content": multimodal_message}, max_turns=1)

    # Verify chat completed successfully
    assert chat_result is not None, "Chat result should not be None"
    assert len(chat_result.chat_history) >= 2, "Should have at least user message and assistant reply"

    # Verify first message contains multimodal content
    first_message = chat_result.chat_history[0]
    _assert_multimodal_content_handling(first_message)

    # Verify assistant's response
    assistant_message = None
    for msg in chat_result.chat_history:
        if msg.get("name") == "multimodal_assistant":
            assistant_message = msg
            break

    assert assistant_message is not None, "Assistant should have responded"
    _assert_multimodal_content_handling(assistant_message)

    # Test content_str processing on both messages
    _verify_content_str_processing(first_message["content"])
    _verify_content_str_processing(assistant_message["content"])


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_two_agent_multimodal_conversation(credentials_gpt_4o_mini: Credentials) -> None:
    """Test two-agent conversation with multimodal content exchange."""

    # Create two agents with different roles
    analyst = ConversableAgent(
        name="data_analyst",
        llm_config=credentials_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        system_message="You are a data analyst who processes visualizations and provides insights.",
        max_consecutive_auto_reply=1,
    )

    # Use a UserProxyAgent to start the conversation with multimodal content
    # This avoids the OpenAI restriction about images only in 'user' role messages
    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", code_execution_config=False, max_consecutive_auto_reply=0
    )

    # Start conversation with multimodal content from user
    multimodal_content = [
        {
            "type": "text",
            "text": "Data analyst, please review this dashboard design and provide feedback to the UI designer:",
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "https://media.githubusercontent.com/media/ag2ai/ag2/refs/heads/main/test/test_files/test_image.png"
            },
        },
    ]

    chat_result = user_proxy.initiate_chat(analyst, message={"content": multimodal_content}, max_turns=2)

    # Verify conversation worked
    assert chat_result is not None, "Chat should complete successfully"
    assert len(chat_result.chat_history) >= 2, "Should have multiple messages"

    # Verify multimodal content was preserved in first message
    first_msg = chat_result.chat_history[0]
    assert isinstance(first_msg["content"], list), "First message should contain multimodal content"
    _assert_multimodal_content_handling(first_msg)

    # Verify analyst participated (designer may not respond in 2 turns)
    participant_names = {msg.get("name") for msg in chat_result.chat_history if msg.get("name")}
    assert "data_analyst" in participant_names, "Analyst should participate"

    # Verify all messages can be processed by content_str
    for msg in chat_result.chat_history:
        _verify_content_str_processing(msg["content"])


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_group_chat_multimodal_content(credentials_gpt_4o_mini: Credentials) -> None:
    """Test group chat with multimodal content sharing."""

    # Create specialized agents
    agents = []

    analyst = ConversableAgent(
        name="analyst",
        llm_config=credentials_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        system_message="You analyze data and charts. Keep responses concise.",
    )
    agents.append(analyst)

    designer = ConversableAgent(
        name="designer",
        llm_config=credentials_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        system_message="You review visual designs. Keep responses concise.",
    )
    agents.append(designer)

    # Create user proxy
    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", code_execution_config=False, max_consecutive_auto_reply=0
    )

    # Create group chat
    groupchat = GroupChat(
        agents=[user_proxy] + agents, messages=[], max_round=3, speaker_selection_method="round_robin"
    )

    manager = GroupChatManager(
        groupchat=groupchat, llm_config=credentials_gpt_4o_mini.llm_config, human_input_mode="NEVER"
    )

    # Start group conversation with multimodal content
    multimodal_message = [
        {"type": "text", "text": "Team, please review this product interface design:"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://media.githubusercontent.com/media/ag2ai/ag2/refs/heads/main/test/test_files/test_image.png"
            },
        },
        {"type": "text", "text": " Please provide your expert feedback."},
    ]

    chat_result = user_proxy.initiate_chat(manager, message={"content": multimodal_message}, max_turns=2)

    # Verify group chat completed
    assert chat_result is not None, "Group chat should complete"
    assert len(chat_result.chat_history) >= 2, "Should have multiple messages"

    # Verify initial multimodal message was preserved
    first_msg = chat_result.chat_history[0]
    _assert_multimodal_content_handling(first_msg)
    assert isinstance(first_msg["content"], list), "First message should be multimodal"

    # Verify that experts participated
    participant_names = {msg.get("name") for msg in chat_result.chat_history if msg.get("name")}
    expert_agents = {"analyst", "designer"}
    participating_experts = participant_names.intersection(expert_agents)
    assert len(participating_experts) >= 1, f"At least one expert should participate. Found: {participant_names}"

    # Verify all messages in history can be processed
    for msg in chat_result.chat_history:
        _assert_multimodal_content_handling(msg)
        _verify_content_str_processing(msg["content"])


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_sequential_chat_multimodal_carryover(credentials_gpt_4o_mini: Credentials) -> None:
    """Test sequential chats with multimodal content and carryover."""

    # Create agents for sequential workflow
    user_proxy = UserProxyAgent(
        name="project_manager", human_input_mode="NEVER", code_execution_config=False, max_consecutive_auto_reply=0
    )

    analyst = ConversableAgent(
        name="business_analyst",
        llm_config=credentials_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        system_message="You analyze business requirements. Be concise.",
        max_consecutive_auto_reply=1,
    )

    reviewer = ConversableAgent(
        name="technical_reviewer",
        llm_config=credentials_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        system_message="You review technical specifications. Be concise.",
        max_consecutive_auto_reply=1,
    )

    # Define sequential chat sequence with multimodal content
    multimodal_initial_message = [
        {"type": "text", "text": "Analyze this system architecture diagram:"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://media.githubusercontent.com/media/ag2ai/ag2/refs/heads/main/test/test_files/test_image.png"
            },
        },
    ]

    chat_sequence = [
        {
            "recipient": analyst,
            "message": {"content": multimodal_initial_message},
            "max_turns": 1,
            "summary_method": "last_msg",
        },
        {"recipient": reviewer, "message": "Review the analysis and provide technical feedback", "max_turns": 1},
    ]

    # Execute sequential chats
    chat_results = user_proxy.initiate_chats(chat_sequence)

    # Verify sequential execution
    assert len(chat_results) == 2, "Should have results from both chats"
    assert all(result.chat_history for result in chat_results), "All chats should have history"

    # Verify first chat has multimodal content
    first_chat = chat_results[0]
    first_msg = first_chat.chat_history[0]
    _assert_multimodal_content_handling(first_msg)
    assert isinstance(first_msg["content"], list), "First message should be multimodal"

    # Verify carryover worked - second chat should reference first
    second_chat = chat_results[1]
    second_first_msg = second_chat.chat_history[0]
    content_str_rep = str(second_first_msg.get("content", ""))

    # Should have carryover context or be longer than original message
    original_msg = "Review the analysis and provide technical feedback"
    assert len(content_str_rep) >= len(original_msg), "Should have carryover context"

    # Verify content_str works on all messages
    for result in chat_results:
        for msg in result.chat_history:
            _verify_content_str_processing(msg["content"])


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_multimodal_content_str_integration(credentials_gpt_4o_mini: Credentials) -> None:
    """Test content_str function with actual multimodal responses from agents."""

    assistant = ConversableAgent(
        name="content_processor",
        llm_config=credentials_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        system_message="Provide detailed responses about visual content.",
    )

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", code_execution_config=False, max_consecutive_auto_reply=1
    )

    # Test various multimodal content structures
    test_cases = [
        # Text only in list format
        [{"type": "text", "text": "This is text-only multimodal content."}],
        # Image only - using a real, accessible image URL
        [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://media.githubusercontent.com/media/ag2ai/ag2/refs/heads/main/test/test_files/test_image.png"
                },
            }
        ],
        # Mixed content - using real image URLs
        [
            {"type": "text", "text": "Start: "},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://media.githubusercontent.com/media/ag2ai/ag2/refs/heads/main/test/test_files/test_image.png"
                },
            },
            {"type": "text", "text": " End."},
        ],
    ]

    for i, test_content in enumerate(test_cases):
        # Send multimodal content to agent
        chat_result = user_proxy.initiate_chat(
            assistant, message={"content": test_content}, max_turns=1, clear_history=True
        )

        assert chat_result is not None, f"Chat {i} should complete"

        # Test content_str on input message
        input_msg = chat_result.chat_history[0]
        input_content = input_msg["content"]

        if isinstance(input_content, list):
            # Test content_str processing
            content_string = content_str(input_content)
            assert isinstance(content_string, str), f"content_str should return string for test case {i}"

            # Verify text content preserved
            text_items = [item for item in input_content if item.get("type") == "text"]
            for text_item in text_items:
                if text_item.get("text"):
                    assert text_item["text"] in content_string, f"Text should be preserved in case {i}"

            # Verify images converted to placeholders
            image_items = [item for item in input_content if item.get("type") == "image_url"]
            expected_image_count = len(image_items)
            actual_image_count = content_string.count("<image>")
            assert actual_image_count == expected_image_count, (
                f"Should have {expected_image_count} <image> placeholders in case {i}, got {actual_image_count}"
            )

        # Test content_str on response
        response_msg = None
        for msg in chat_result.chat_history:
            if msg.get("name") == "content_processor":
                response_msg = msg
                break

        if response_msg:
            _verify_content_str_processing(response_msg["content"])


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_multimodal_backwards_compatibility_integration(credentials_gpt_4o_mini: Credentials) -> None:
    """Test that multimodal changes don't break existing string/dict message patterns."""

    assistant = ConversableAgent(
        name="compatibility_test",
        llm_config=credentials_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        system_message="You are a helpful assistant. Keep responses brief.",
    )

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", code_execution_config=False, max_consecutive_auto_reply=1
    )

    # Test 1: Traditional string message
    chat_result1 = user_proxy.initiate_chat(
        assistant, message="Hello, this is a simple string message.", max_turns=1, clear_history=True
    )

    assert chat_result1 is not None, "String message chat should work"
    assert len(chat_result1.chat_history) >= 2, "Should have user and assistant messages"

    # Verify string content handling
    user_msg = chat_result1.chat_history[0]
    assert isinstance(user_msg["content"], str), "User message should be string"

    # Test 2: Traditional dict message
    chat_result2 = user_proxy.initiate_chat(
        assistant,
        message={"role": "user", "content": "This is a dictionary message format."},
        max_turns=1,
        clear_history=True,
    )

    assert chat_result2 is not None, "Dict message chat should work"
    assert len(chat_result2.chat_history) >= 2, "Should have user and assistant messages"

    # Test 3: Mixed conversation - start with string, verify responses work
    chat_result3 = assistant.initiate_chat(
        user_proxy, message="Start with string, then agent should respond normally", max_turns=2, clear_history=True
    )

    assert chat_result3 is not None, "Mixed conversation should work"

    # Test that all content can be processed by content_str (backwards compatibility)
    all_results = [chat_result1, chat_result2, chat_result3]
    for i, result in enumerate(all_results):
        for msg in result.chat_history:
            content = msg["content"]
            # content_str should handle both strings and lists
            try:
                content_string = content_str(content)
                assert isinstance(content_string, str), f"content_str should return string for result {i}"
                # Note: content_str can return empty string for None or empty content, which is valid
            except Exception as e:
                pytest.fail(f"content_str failed on backwards compatibility test {i}: {e}")


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_error_handling_multimodal_integration(credentials_gpt_4o_mini: Credentials) -> None:
    """Test error handling with malformed multimodal content in real scenarios."""

    assistant = ConversableAgent(
        name="error_test_agent",
        llm_config=credentials_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        system_message="You are a helpful assistant.",
    )

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", code_execution_config=False, max_consecutive_auto_reply=1
    )

    # Test that malformed content is handled by content_str
    # Based on actual content_str behavior from simple integration tests

    # Test cases that should raise errors
    error_test_cases = [
        # Missing text field should raise KeyError
        ([{"type": "text"}], KeyError),
        # Unknown type should raise ValueError
        ([{"type": "unknown", "data": "test"}], ValueError),
        # Non-dict items in list should raise TypeError
        (["not a dict"], TypeError),
        # Invalid content type (not str, list, or None) should raise TypeError
        (123, TypeError),
    ]

    for content, expected_error in error_test_cases:
        with pytest.raises(expected_error):
            content_str(content)

    # Test cases that are handled gracefully (don't raise errors)
    graceful_cases = [
        # Missing image_url field - returns "<image>" placeholder
        ([{"type": "image_url"}], "<image>"),
        # Empty image_url - returns "<image>" placeholder
        ([{"type": "image_url", "image_url": ""}], "<image>"),
    ]

    for case, expected_result in graceful_cases:
        result = content_str(case)
        assert isinstance(result, str), f"Should return string for case: {case}"
        assert result == expected_result, f"Expected {expected_result}, got {result} for case: {case}"

    # Test that well-formed content still works
    valid_content = [
        {"type": "text", "text": "This is valid multimodal content"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://media.githubusercontent.com/media/ag2ai/ag2/refs/heads/main/test/test_files/test_image.png"
            },
        },
    ]

    # Should not raise exception
    result = content_str(valid_content)
    assert isinstance(result, str), "Valid content should process correctly"
    assert "valid multimodal content" in result, "Text should be preserved"
    assert "<image>" in result, "Image should be converted to placeholder"

    # Test that agents can still handle valid multimodal content after error tests
    chat_result = user_proxy.initiate_chat(assistant, message={"content": valid_content}, max_turns=1)

    assert chat_result is not None, "Valid multimodal content should work in agent chat"
    assert len(chat_result.chat_history) >= 2, "Should complete successfully"


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_conversable_agent_run_multimodal(credentials_gpt_4o_mini: Credentials) -> None:
    """Test ConversableAgent::run method with multimodal content via agent.run()."""

    assistant = ConversableAgent(
        name="multimodal_runner",
        llm_config=credentials_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        system_message="You are a helpful assistant that processes text and images. Keep responses brief.",
    )

    # Test with text-only multimodal content first (simpler case)
    text_only_multimodal = [{"type": "text", "text": "This is text-only multimodal content. Respond briefly."}]

    # Use the actual run method - it returns a response object that needs to be processed
    run_response = assistant.run(
        message={"content": text_only_multimodal},
        user_input=False,  # No user input required
        max_turns=1,
        clear_history=True,
    )

    # Verify run response object
    assert run_response is not None, "Run should return response object"
    assert hasattr(run_response, "messages"), "Response should have messages attribute"
    assert hasattr(run_response, "process"), "Response should have process method"

    # Process the response to actually execute the conversation
    run_response.process()

    # After processing, the messages should be available
    messages_list = list(run_response.messages)

    # Verify we got the expected messages
    assert len(messages_list) >= 2, (
        f"Should have user message and assistant response, got {len(messages_list)} messages"
    )

    # Check that the first message contains multimodal content
    first_message = messages_list[0]
    _assert_multimodal_content_handling(first_message)
    assert isinstance(first_message["content"], list), "First message should be multimodal"

    # Verify assistant responded
    assistant_response = None
    for msg in messages_list:
        if msg.get("name") == "multimodal_runner":
            assistant_response = msg
            break

    assert assistant_response is not None, "Assistant should have responded"
    _assert_multimodal_content_handling(assistant_response)

    # Test content_str processing on messages
    for msg in messages_list:
        _verify_content_str_processing(msg["content"])

    # Test that the run method properly handles the background execution
    # and we can access the messages after completion
    print(f"Test completed successfully with {len(messages_list)} messages")


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_initiate_group_chat_multimodal(credentials_gpt_4o_mini: Credentials) -> None:
    """Test initiate_group_chat function with multimodal content."""
    from autogen.agentchat.group.multi_agent_chat import initiate_group_chat
    from autogen.agentchat.group.patterns.auto import AutoPattern

    # Create agents for group chat
    analyst = ConversableAgent(
        name="data_analyst",
        llm_config=credentials_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        system_message="You analyze data and provide insights. Keep responses brief.",
    )

    designer = ConversableAgent(
        name="ui_designer",
        llm_config=credentials_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        system_message="You review designs and interfaces. Keep responses brief.",
    )

    # Create user proxy for group chat
    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", code_execution_config=False, max_consecutive_auto_reply=0
    )

    # Create pattern for group orchestration
    pattern = AutoPattern(
        initial_agent=analyst,
        agents=[analyst, designer],
        user_agent=user_proxy,
        group_manager_args={"llm_config": credentials_gpt_4o_mini.llm_config},
    )

    # Test 1: String message
    chat_result1, context_vars1, last_speaker1 = initiate_group_chat(
        pattern=pattern, messages="Analyze this product roadmap and provide feedback.", max_rounds=3
    )

    assert chat_result1 is not None, "Group chat with string should complete"
    assert len(chat_result1.chat_history) >= 2, "Should have conversation history"
    assert context_vars1 is not None, "Should return context variables"
    assert last_speaker1 is not None, "Should return last speaker"

    # Test 2: Multimodal content
    multimodal_message = _create_test_multimodal_content()

    chat_result2, _, _ = initiate_group_chat(
        pattern=pattern, messages=[{"role": "user", "content": multimodal_message}], max_rounds=3
    )

    assert chat_result2 is not None, "Group chat with multimodal should complete"
    assert len(chat_result2.chat_history) >= 2, "Should have conversation history"

    # Verify multimodal content was preserved
    first_msg = chat_result2.chat_history[0]
    _assert_multimodal_content_handling(first_msg)
    if isinstance(first_msg["content"], list):
        assert any(item.get("type") == "text" for item in first_msg["content"]), "Should have text content"
        assert any(item.get("type") == "image_url" for item in first_msg["content"]), "Should have image content"

    # Verify all messages can be processed by content_str
    for msg in chat_result2.chat_history:
        _verify_content_str_processing(msg["content"])

    # Test that agents participated
    participant_names = {msg.get("name") for msg in chat_result2.chat_history if msg.get("name")}
    expected_agents = {"data_analyst", "ui_designer"}
    participating_agents = participant_names.intersection(expected_agents)
    assert len(participating_agents) >= 1, f"At least one agent should participate. Found: {participant_names}"


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_run_group_chat_multimodal(credentials_gpt_4o_mini: Credentials) -> None:
    """Test run_group_chat function with multimodal content and streaming."""
    from autogen.agentchat.group.multi_agent_chat import run_group_chat
    from autogen.agentchat.group.patterns.round_robin import RoundRobinPattern

    # Create agents for round-robin group chat
    reviewer = ConversableAgent(
        name="code_reviewer",
        llm_config=credentials_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        system_message="You review code and architectures. Keep responses very brief.",
    )

    tester = ConversableAgent(
        name="qa_tester",
        llm_config=credentials_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        system_message="You test software and find issues. Keep responses very brief.",
    )

    # Create user proxy for round-robin group chat
    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", code_execution_config=False, max_consecutive_auto_reply=0
    )

    # Create round-robin pattern
    pattern = RoundRobinPattern(initial_agent=reviewer, agents=[reviewer, tester], user_agent=user_proxy)

    # Test 1: String message with streaming response
    response1 = run_group_chat(
        pattern=pattern, messages="Review this system architecture for security issues.", max_rounds=2
    )

    assert response1 is not None, "Should return response object"
    assert hasattr(response1, "iostream"), "Should have iostream for streaming"

    # Wait briefly for async processing
    import time

    time.sleep(2)

    # Test 2: Multimodal content with streaming
    multimodal_content = [
        {"type": "text", "text": "Review this system diagram:"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://media.githubusercontent.com/media/ag2ai/ag2/refs/heads/main/test/test_files/test_image.png"
            },
        },
    ]

    response2 = run_group_chat(
        pattern=pattern, messages=[{"role": "user", "content": multimodal_content}], max_rounds=2
    )

    assert response2 is not None, "Multimodal group chat should return response"
    assert hasattr(response2, "iostream"), "Should have iostream for streaming"

    # Test that response objects are properly configured (if agents attribute exists)
    if hasattr(response2, "agents"):
        assert len(response2.agents) > 0, "Agents list should be populated"

    # Wait for processing
    time.sleep(2)


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_pattern_based_multimodal_orchestration(credentials_gpt_4o_mini: Credentials) -> None:
    """Test different orchestration patterns with multimodal content."""
    from autogen.agentchat.group.multi_agent_chat import initiate_group_chat
    from autogen.agentchat.group.patterns.auto import AutoPattern
    from autogen.agentchat.group.patterns.random import RandomPattern

    # Create agents
    analyst = ConversableAgent(
        name="analyst",
        llm_config=credentials_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        system_message="Analyze content briefly.",
    )

    critic = ConversableAgent(
        name="critic",
        llm_config=credentials_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        system_message="Provide critical feedback briefly.",
    )

    # Test multimodal content
    multimodal_msg = [
        {"type": "text", "text": "Evaluate this design:"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://media.githubusercontent.com/media/ag2ai/ag2/refs/heads/main/test/test_files/test_image.png"
            },
        },
    ]

    # Create user proxy for pattern tests
    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", code_execution_config=False, max_consecutive_auto_reply=0
    )

    # Test 1: AutoPattern with multimodal
    auto_pattern = AutoPattern(
        initial_agent=analyst,
        agents=[analyst, critic],
        user_agent=user_proxy,
        group_manager_args={"llm_config": credentials_gpt_4o_mini.llm_config},
    )

    result1, _, _ = initiate_group_chat(
        pattern=auto_pattern, messages=[{"role": "user", "content": multimodal_msg}], max_rounds=2
    )

    assert result1 is not None, "AutoPattern should handle multimodal content"
    assert len(result1.chat_history) >= 1, "Should have chat history"

    # Verify multimodal content preservation
    first_msg = result1.chat_history[0]
    _assert_multimodal_content_handling(first_msg)

    # Test content_str processing on all messages
    for msg in result1.chat_history:
        _verify_content_str_processing(msg["content"])

    # Test 2: RandomPattern with multimodal (shorter test due to randomness)
    random_pattern = RandomPattern(initial_agent=critic, agents=[analyst, critic], user_agent=user_proxy)

    result2, _, _ = initiate_group_chat(
        pattern=random_pattern, messages=[{"role": "user", "content": multimodal_msg}], max_rounds=2
    )

    assert result2 is not None, "RandomPattern should handle multimodal content"
    assert len(result2.chat_history) >= 1, "Should have chat history"

    # Verify content processing
    for msg in result2.chat_history:
        _verify_content_str_processing(msg["content"])


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_group_chat_context_variables_multimodal(credentials_gpt_4o_mini: Credentials) -> None:
    """Test context variables with multimodal content in group chats."""
    from autogen.agentchat.group.context_variables import ContextVariables
    from autogen.agentchat.group.multi_agent_chat import initiate_group_chat
    from autogen.agentchat.group.patterns.auto import AutoPattern

    # Create agent
    processor = ConversableAgent(
        name="content_processor",
        llm_config=credentials_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        system_message="Process content and maintain context. Keep responses brief.",
    )

    # Initialize context variables
    initial_context = ContextVariables()
    initial_context["session_id"] = "multimodal_test_123"
    initial_context["content_type"] = "mixed_media"

    # Create user proxy for context test
    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", code_execution_config=False, max_consecutive_auto_reply=0
    )

    # Create pattern with context
    pattern = AutoPattern(
        initial_agent=processor,
        agents=[processor],
        user_agent=user_proxy,
        context_variables=initial_context,
        group_manager_args={"llm_config": credentials_gpt_4o_mini.llm_config},
    )

    # Test with multimodal content
    multimodal_message = [
        {"type": "text", "text": "Process this content for session analysis:"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://media.githubusercontent.com/media/ag2ai/ag2/refs/heads/main/test/test_files/test_image.png"
            },
        },
    ]

    chat_result, final_context, last_speaker = initiate_group_chat(
        pattern=pattern, messages=[{"role": "user", "content": multimodal_message}], max_rounds=2
    )

    # Verify chat completed
    assert chat_result is not None, "Context-aware group chat should complete"
    assert len(chat_result.chat_history) >= 1, "Should have conversation"

    # Verify context variables were maintained
    assert final_context is not None, "Should return final context"
    assert final_context.get("session_id") == "multimodal_test_123", "Context should be preserved"
    assert final_context.get("content_type") == "mixed_media", "Context should be preserved"

    # Verify multimodal content handling
    first_msg = chat_result.chat_history[0]
    _assert_multimodal_content_handling(first_msg)

    # Test content_str on all messages
    for msg in chat_result.chat_history:
        _verify_content_str_processing(msg["content"])

    # Verify last speaker
    assert last_speaker is not None, "Should return last speaker"
