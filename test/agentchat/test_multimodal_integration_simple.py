# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT


import pytest

from autogen import ConversableAgent, UserProxyAgent
from autogen.agentchat.groupchat import GroupChat, GroupChatManager
from autogen.code_utils import content_str
from test.credentials import Credentials


@pytest.mark.integration
@pytest.mark.openai
def test_text_only_multimodal_format(credentials_gpt_4o_mini: Credentials) -> None:
    """Test agents can handle text-only content in list format (multimodal structure)."""

    assistant = ConversableAgent(
        name="text_assistant",
        llm_config=credentials_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        system_message="You are a helpful assistant. Keep responses brief.",
    )

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", code_execution_config=False, max_consecutive_auto_reply=1
    )

    # Test text-only content in multimodal list format
    text_multimodal_content = [{"type": "text", "text": "Hello, this is a test message in multimodal format."}]

    chat_result = user_proxy.initiate_chat(assistant, message={"content": text_multimodal_content}, max_turns=1)

    # Verify chat completed successfully
    assert chat_result is not None, "Chat should complete successfully"
    assert len(chat_result.chat_history) >= 2, "Should have user and assistant messages"

    # Verify first message has multimodal content structure
    first_message = chat_result.chat_history[0]
    content = first_message.get("content")

    # AG2 might convert it to string, which is fine
    if isinstance(content, list):
        assert len(content) == 1, "Should have one content item"
        assert content[0]["type"] == "text", "Should be text type"
        assert "test message" in content[0]["text"], "Should contain original text"
    else:
        # If converted to string, should still contain the text
        assert isinstance(content, str), "Content should be string if not list"
        assert "test message" in content, "Should contain original text"

    # Test content_str processing
    processed = content_str(text_multimodal_content)
    assert isinstance(processed, str), "content_str should return string"
    assert "test message" in processed, "Processed content should contain text"


@pytest.mark.integration
@pytest.mark.openai
def test_backwards_compatibility_string_messages(credentials_gpt_4o_mini: Credentials) -> None:
    """Test that string messages still work correctly."""

    assistant = ConversableAgent(
        name="compat_assistant",
        llm_config=credentials_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        system_message="You are a helpful assistant. Keep responses brief.",
    )

    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", code_execution_config=False, max_consecutive_auto_reply=1
    )

    # Test traditional string message
    chat_result = user_proxy.initiate_chat(assistant, message="This is a traditional string message.", max_turns=1)

    assert chat_result is not None, "String message chat should work"
    assert len(chat_result.chat_history) >= 2, "Should have user and assistant messages"

    # Verify string content handling
    user_msg = chat_result.chat_history[0]
    assert isinstance(user_msg["content"], str), "User message should be string"
    assert "traditional string" in user_msg["content"], "Should contain original text"

    # Test content_str on string content
    processed = content_str(user_msg["content"])
    assert isinstance(processed, str), "content_str should handle strings"
    assert processed == user_msg["content"], "String should pass through unchanged"


def test_content_str_error_handling() -> None:
    """Test content_str function error handling with malformed content."""

    # Test cases that should definitely raise errors based on content_str implementation
    error_test_cases = [
        # Non-dict items in list should raise TypeError
        (["not a dict"], TypeError),
        # Missing type field should raise AssertionError
        ([{"text": "missing type"}], AssertionError),
        # Unknown type should raise ValueError
        ([{"type": "unknown", "data": "test"}], ValueError),
        # Missing text field should raise KeyError
        ([{"type": "text"}], KeyError),
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
        ([{"type": "image_url", "image_url": {}}], "<image>"),
    ]

    for case, expected_result in graceful_cases:
        result = content_str(case)
        assert isinstance(result, str), f"Should return string for case: {case}"
        assert result == expected_result, f"Expected {expected_result}, got {result} for case: {case}"

    # Test that well-formed content definitely works
    valid_content = [{"type": "text", "text": "This is valid content"}]

    # Should not raise exception
    result = content_str(valid_content)
    assert isinstance(result, str), "Valid content should process correctly"
    assert "valid content" in result, "Text should be preserved"


@pytest.mark.integration
@pytest.mark.openai
def test_two_agent_text_multimodal_conversation(credentials_gpt_4o_mini: Credentials) -> None:
    """Test two-agent conversation with text in multimodal format."""

    analyst = ConversableAgent(
        name="data_analyst",
        llm_config=credentials_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        system_message="You are a data analyst. Keep responses concise.",
        max_consecutive_auto_reply=1,
    )

    designer = ConversableAgent(
        name="ui_designer",
        llm_config=credentials_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        system_message="You are a UI designer. Keep responses concise.",
        max_consecutive_auto_reply=1,
    )

    # Use text-only multimodal content to avoid image URL issues
    multimodal_content = [
        {"type": "text", "text": "Please review this design concept and provide feedback on usability and aesthetics."}
    ]

    chat_result = analyst.initiate_chat(designer, message={"content": multimodal_content}, max_turns=2)

    # Verify conversation worked
    assert chat_result is not None, "Chat should complete successfully"
    assert len(chat_result.chat_history) >= 2, "Should have multiple messages"

    # Verify both agents participated
    participant_names = {msg.get("name") for msg in chat_result.chat_history if msg.get("name")}
    assert "data_analyst" in participant_names, "Analyst should participate"
    assert "ui_designer" in participant_names, "Designer should participate"

    # Test that all content can be processed by content_str
    for msg in chat_result.chat_history:
        content = msg["content"]
        processed = content_str(content)
        assert isinstance(processed, str), "All content should be processable by content_str"


@pytest.mark.integration
@pytest.mark.openai
def test_group_chat_text_multimodal(credentials_gpt_4o_mini: Credentials) -> None:
    """Test group chat with text content in multimodal format."""

    # Create specialized agents
    analyst = ConversableAgent(
        name="analyst",
        llm_config=credentials_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        system_message="You analyze requirements. Be concise.",
    )

    designer = ConversableAgent(
        name="designer",
        llm_config=credentials_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        system_message="You design solutions. Be concise.",
    )

    # Create user proxy
    user_proxy = UserProxyAgent(
        name="user", human_input_mode="NEVER", code_execution_config=False, max_consecutive_auto_reply=0
    )

    # Create group chat
    groupchat = GroupChat(
        agents=[user_proxy, analyst, designer], messages=[], max_round=3, speaker_selection_method="round_robin"
    )

    manager = GroupChatManager(
        groupchat=groupchat, llm_config=credentials_gpt_4o_mini.llm_config, human_input_mode="NEVER"
    )

    # Start group conversation with text multimodal content
    multimodal_message = [
        {
            "type": "text",
            "text": "Team, please collaborate on improving our user onboarding process. Analyst, please identify key pain points. Designer, please suggest solutions.",
        }
    ]

    chat_result = user_proxy.initiate_chat(manager, message={"content": multimodal_message}, max_turns=2)

    # Verify group chat completed
    assert chat_result is not None, "Group chat should complete"
    assert len(chat_result.chat_history) >= 2, "Should have multiple messages"

    # Verify that team members participated
    participant_names = {msg.get("name") for msg in chat_result.chat_history if msg.get("name")}
    team_members = {"analyst", "designer"}
    participating_members = participant_names.intersection(team_members)
    assert len(participating_members) >= 1, f"At least one team member should participate. Found: {participant_names}"

    # Verify all messages can be processed
    for msg in chat_result.chat_history:
        content = msg["content"]
        processed = content_str(content)
        assert isinstance(processed, str), "All content should be processable"


@pytest.mark.integration
@pytest.mark.openai
def test_sequential_chat_text_multimodal_carryover(credentials_gpt_4o_mini: Credentials) -> None:
    """Test sequential chats with text multimodal content and carryover."""

    user_proxy = UserProxyAgent(
        name="project_manager", human_input_mode="NEVER", code_execution_config=False, max_consecutive_auto_reply=0
    )

    analyst = ConversableAgent(
        name="business_analyst",
        llm_config=credentials_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        system_message="You analyze requirements. Be very concise.",
        max_consecutive_auto_reply=1,
    )

    reviewer = ConversableAgent(
        name="technical_reviewer",
        llm_config=credentials_gpt_4o_mini.llm_config,
        human_input_mode="NEVER",
        system_message="You review proposals. Be very concise.",
        max_consecutive_auto_reply=1,
    )

    # Define sequential chat sequence with text multimodal content
    multimodal_initial_message = [
        {
            "type": "text",
            "text": "Analyze the requirements for our new feature: user authentication with social login options.",
        }
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
    content = first_msg.get("content")

    # Content should contain the authentication requirement
    if isinstance(content, list):
        assert any("authentication" in str(item.get("text", "")) for item in content if item.get("type") == "text")
    else:
        assert "authentication" in str(content)

    # Verify content_str works on all messages
    for result in chat_results:
        for msg in result.chat_history:
            processed = content_str(msg["content"])
            assert isinstance(processed, str), "All content should be processable"
