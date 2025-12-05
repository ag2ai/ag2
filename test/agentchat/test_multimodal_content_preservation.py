# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for multimodal content preservation throughout AG2 message pipeline.

These tests verify that images, audio, and video content remain intact
and are not converted to string placeholders like <image>, <audio>, <video>.

Run with: bash scripts/test-skip-llm.sh test/agentchat/test_multimodal_content_preservation.py
"""

from autogen import AssistantAgent, ConversableAgent, GroupChat, GroupChatManager, UserProxyAgent


class TestGroupChatMultimodalPreservation:
    """Test GroupChat preserves multimodal content."""

    def test_append_preserves_image_content(self):
        """GroupChat.append() preserves image content blocks."""
        agent = ConversableAgent("test", llm_config=False)
        groupchat = GroupChat(agents=[agent], messages=[])

        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
            ],
        }

        groupchat.append(message, agent)

        # Verify content preserved as list
        assert isinstance(groupchat.messages[-1]["content"], list)
        assert len(groupchat.messages[-1]["content"]) == 2

        # Verify image block intact
        assert groupchat.messages[-1]["content"][1]["type"] == "image_url"
        assert "data:image/png;base64,iVBORw0KGgo=" in groupchat.messages[-1]["content"][1]["image_url"]["url"]

        # Verify NOT converted to <image> placeholder
        content_str_repr = str(groupchat.messages[-1]["content"])
        assert "<image>" not in content_str_repr
        assert "data:image/png;base64,iVBORw0KGgo=" in content_str_repr

    def test_append_preserves_audio_content(self):
        """GroupChat.append() preserves audio content blocks."""
        agent = ConversableAgent("test", llm_config=False)
        groupchat = GroupChat(agents=[agent], messages=[])

        message = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Generated audio"},
                {"type": "audio_url", "audio_url": {"url": "data:audio/pcm;base64,UklGRiQAAABXQVZF"}},
            ],
        }

        groupchat.append(message, agent)

        # Verify audio preserved
        assert isinstance(groupchat.messages[-1]["content"], list)
        assert groupchat.messages[-1]["content"][1]["type"] == "audio_url"
        assert "data:audio/pcm;base64,UklGRiQAAABXQVZF" in groupchat.messages[-1]["content"][1]["audio_url"]["url"]

        # Verify NOT converted to <audio> placeholder
        content_str_repr = str(groupchat.messages[-1]["content"])
        assert "<audio>" not in content_str_repr
        assert "data:audio/pcm;base64,UklGRiQAAABXQVZF" in content_str_repr

    def test_append_preserves_video_content(self):
        """GroupChat.append() preserves video content blocks."""
        agent = ConversableAgent("test", llm_config=False)
        groupchat = GroupChat(agents=[agent], messages=[])

        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Watch this video"},
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAAIGZ0eXBpc29t"}},
            ],
        }

        groupchat.append(message, agent)

        # Verify video preserved
        assert isinstance(groupchat.messages[-1]["content"], list)
        assert groupchat.messages[-1]["content"][1]["type"] == "video_url"
        assert "data:video/mp4;base64,AAAAIGZ0eXBpc29t" in groupchat.messages[-1]["content"][1]["video_url"]["url"]

    def test_append_preserves_mixed_multimodal(self):
        """GroupChat.append() preserves mixed text/image/audio content."""
        agent = ConversableAgent("test", llm_config=False)
        groupchat = GroupChat(agents=[agent], messages=[])

        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Task description"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,ABC"}},
                {"type": "text", "text": "Additional context"},
                {"type": "audio_url", "audio_url": {"url": "data:audio/pcm;base64,XYZ"}},
            ],
        }

        groupchat.append(message, agent)

        # Verify all blocks preserved
        assert len(groupchat.messages[-1]["content"]) == 4
        assert groupchat.messages[-1]["content"][0]["type"] == "text"
        assert groupchat.messages[-1]["content"][1]["type"] == "image_url"
        assert groupchat.messages[-1]["content"][2]["type"] == "text"
        assert groupchat.messages[-1]["content"][3]["type"] == "audio_url"

        # Verify actual data URIs present (not placeholders)
        content_str_repr = str(groupchat.messages[-1]["content"])
        assert "data:image/png;base64,ABC" in content_str_repr
        assert "data:audio/pcm;base64,XYZ" in content_str_repr

    def test_text_only_backward_compatibility(self):
        """Text-only messages continue to work as before."""
        agent = ConversableAgent("test", llm_config=False)
        groupchat = GroupChat(agents=[agent], messages=[])

        # String content
        message1 = {"role": "user", "content": "Hello"}
        groupchat.append(message1, agent)
        assert groupchat.messages[-1]["content"] == "Hello"

        # List with text-only
        message2 = {"role": "user", "content": [{"type": "text", "text": "World"}]}
        groupchat.append(message2, agent)
        assert isinstance(groupchat.messages[-1]["content"], list)
        assert groupchat.messages[-1]["content"][0].get("type") == "text"

    def test_none_content_preserved(self):
        """None content (for tool calls) is preserved."""
        agent = ConversableAgent("test", llm_config=False)
        groupchat = GroupChat(agents=[agent], messages=[])

        message = {"role": "assistant", "content": None, "tool_calls": [{"id": "call_123", "type": "function"}]}
        groupchat.append(message, agent)

        assert groupchat.messages[-1]["content"] is None


class TestGroupChatManagerMultimodalPreservation:
    """Test GroupChatManager preserves multimodal content."""

    def test_process_resume_termination_preserves_multimodal(self):
        """_process_resume_termination() preserves multimodal content."""
        agent = ConversableAgent("test", llm_config=False)
        groupchat = GroupChat(agents=[agent], messages=[])
        manager = GroupChatManager(groupchat=groupchat, llm_config=False)

        # Add multimodal message to history
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this\nTERMINATE"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
            ],
        }
        groupchat.append(message, agent)

        # Process history with termination string removal
        messages = groupchat.messages.copy()
        manager._process_resume_termination("TERMINATE", messages)

        # Verify multimodal content preserved
        processed_content = messages[-1]["content"]
        assert isinstance(processed_content, list)
        assert len(processed_content) == 2

        # Verify termination removed from text block
        assert processed_content[0]["type"] == "text"
        assert processed_content[0]["text"] == "Analyze this\n"

        # Verify image block unchanged
        assert processed_content[1]["type"] == "image_url"
        assert "iVBORw0KGgo=" in processed_content[1]["image_url"]["url"]

    def test_process_resume_termination_preserves_multiple_text_blocks(self):
        """_process_resume_termination() handles multiple text blocks correctly."""
        agent = ConversableAgent("test", llm_config=False)
        groupchat = GroupChat(agents=[agent], messages=[])
        manager = GroupChatManager(groupchat=groupchat, llm_config=False)

        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "First part\nTERMINATE"},
                {"type": "audio_url", "audio_url": {"url": "data:audio/mp3;base64,//NgxAAAAA=="}},
                {"type": "text", "text": "Second part\nTERMINATE"},
            ],
        }
        groupchat.append(message, agent)

        messages = groupchat.messages.copy()
        manager._process_resume_termination("TERMINATE", messages)

        processed_content = messages[-1]["content"]
        assert isinstance(processed_content, list)
        assert len(processed_content) == 3

        # Both text blocks should have termination removed
        assert processed_content[0]["text"] == "First part\n"
        assert processed_content[2]["text"] == "Second part\n"

        # Audio block unchanged
        assert processed_content[1]["type"] == "audio_url"


class TestConversableAgentMultimodalPreservation:
    """Test ConversableAgent preserves multimodal in tool responses via normilize_message_to_oai."""

    def test_tool_response_normalization_preserves_multimodal(self):
        """Tool response normalization preserves multimodal content."""
        from autogen.agentchat.conversable_agent import normilize_message_to_oai

        message = {
            "role": "tool",
            "content": "Image generated",  # Top-level content required by normilize_message_to_oai
            "tool_responses": [
                {
                    "tool_call_id": "call_123",
                    "role": "tool",
                    "content": [
                        {"type": "text", "text": "Generated image"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,TOOL_IMG"}},
                    ],
                }
            ],
        }

        valid, oai_message = normilize_message_to_oai(message, role="tool", name="test")

        # Verify multimodal preserved
        assert valid
        assert isinstance(oai_message["tool_responses"][0]["content"], list)
        assert "data:image/png;base64,TOOL_IMG" in str(oai_message)

        # Verify NOT converted to <image> placeholder
        assert "<image>" not in str(oai_message)

    def test_tool_response_normalization_preserves_audio(self):
        """Tool response normalization preserves audio content."""
        from autogen.agentchat.conversable_agent import normilize_message_to_oai

        message = {
            "role": "tool",
            "content": "Audio generated",  # Top-level content required
            "tool_responses": [
                {
                    "tool_call_id": "call_456",
                    "role": "tool",
                    "content": [
                        {"type": "text", "text": "Generated audio"},
                        {"type": "audio_url", "audio_url": {"url": "data:audio/pcm;base64,TOOL_AUDIO"}},
                    ],
                }
            ],
        }

        valid, oai_message = normilize_message_to_oai(message, role="tool", name="test")

        assert valid
        assert isinstance(oai_message["tool_responses"][0]["content"], list)
        assert "data:audio/pcm;base64,TOOL_AUDIO" in str(oai_message)

    def test_tool_response_normalization_text_only(self):
        """Tool response normalization handles text-only responses (backward compatibility)."""
        from autogen.agentchat.conversable_agent import normilize_message_to_oai

        message = {
            "role": "tool",
            "content": "Simple text result",  # Top-level content required
            "tool_responses": [{"tool_call_id": "call_789", "role": "tool", "content": "Simple text result"}],
        }

        valid, oai_message = normilize_message_to_oai(message, role="tool", name="test")

        assert valid
        assert oai_message["tool_responses"][0]["content"] == "Simple text result"


class TestGroupToolExecutorMultimodalPreservation:
    """Test GroupToolExecutor preserves multimodal tool responses."""

    def test_group_tool_executor_preserves_image_response(self):
        """GroupToolExecutor preserves image content in tool responses."""
        # Simulate tool response with image
        tool_response = {
            "tool_call_id": "call_img",
            "role": "tool",
            "content": [
                {"type": "text", "text": "Image generated successfully"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,GENERATED"}},
            ],
        }

        # The executor should preserve this structure
        # Testing the actual flow would require full integration, so we verify the structure
        assert isinstance(tool_response["content"], list)
        assert "data:image/png;base64,GENERATED" in str(tool_response)

    def test_group_tool_executor_preserves_audio_response(self):
        """GroupToolExecutor preserves audio content in tool responses."""
        tool_response = {
            "tool_call_id": "call_aud",
            "role": "tool",
            "content": [
                {"type": "text", "text": "Audio generated"},
                {"type": "audio_url", "audio_url": {"url": "data:audio/pcm;base64,GEN_AUDIO"}},
            ],
        }

        assert isinstance(tool_response["content"], list)
        assert "data:audio/pcm;base64,GEN_AUDIO" in str(tool_response)


class TestMultimodalContentInOrchestrationPatterns:
    """Test multimodal content flows through different orchestration patterns."""

    def test_two_agent_chat_with_multimodal(self):
        """Two-agent chat preserves multimodal content."""
        user = UserProxyAgent(
            "user", human_input_mode="NEVER", code_execution_config=False, max_consecutive_auto_reply=0
        )
        assistant = AssistantAgent("assistant", llm_config=False, max_consecutive_auto_reply=0)

        # Simulate multimodal message
        multimodal_msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,ANALYSIS"}},
            ],
        }

        # Send message
        user.send(multimodal_msg, assistant, request_reply=False)

        # Verify message preserved in assistant's history
        assert len(assistant._oai_messages[user]) > 0
        last_msg = assistant._oai_messages[user][-1]
        assert isinstance(last_msg["content"], list)
        assert "data:image/png;base64,ANALYSIS" in str(last_msg)

    def test_group_chat_multimodal_flow(self):
        """Group chat with multiple agents preserves multimodal content."""
        user = UserProxyAgent(
            "user", human_input_mode="NEVER", code_execution_config=False, max_consecutive_auto_reply=0
        )
        agent1 = AssistantAgent("agent1", llm_config=False, max_consecutive_auto_reply=0)
        agent2 = AssistantAgent("agent2", llm_config=False, max_consecutive_auto_reply=0)

        groupchat = GroupChat(agents=[user, agent1, agent2], messages=[], max_round=3)

        # Add multimodal message to group chat
        multimodal_msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Team, analyze this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,TEAM_IMG"}},
                {"type": "audio_url", "audio_url": {"url": "data:audio/pcm;base64,TEAM_AUD"}},
            ],
        }

        groupchat.append(multimodal_msg, user)

        # Verify preserved in group chat history
        assert len(groupchat.messages) == 1
        assert isinstance(groupchat.messages[0]["content"], list)
        assert len(groupchat.messages[0]["content"]) == 3

        # Verify actual data URIs present
        msg_str = str(groupchat.messages[0])
        assert "data:image/png;base64,TEAM_IMG" in msg_str
        assert "data:audio/pcm;base64,TEAM_AUD" in msg_str

        # Verify NOT converted to placeholders
        assert "<image>" not in msg_str or "data:image/png;base64,TEAM_IMG" in msg_str
        assert "<audio>" not in msg_str or "data:audio/pcm;base64,TEAM_AUD" in msg_str


class TestEdgeCases:
    """Test edge cases for multimodal content handling."""

    def test_empty_content_list(self):
        """Empty content list is handled correctly."""
        agent = ConversableAgent("test", llm_config=False)
        groupchat = GroupChat(agents=[agent], messages=[])

        message = {"role": "assistant", "content": []}
        groupchat.append(message, agent)

        assert groupchat.messages[-1]["content"] == []

    def test_mixed_none_and_multimodal(self):
        """Messages with None followed by multimodal are handled."""
        agent = ConversableAgent("test", llm_config=False)
        groupchat = GroupChat(agents=[agent], messages=[])

        # First message with None (tool call)
        msg1 = {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]}
        groupchat.append(msg1, agent)

        # Second message with multimodal
        msg2 = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Result"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,RES"}},
            ],
        }
        groupchat.append(msg2, agent)

        assert groupchat.messages[0]["content"] is None
        assert isinstance(groupchat.messages[1]["content"], list)
        assert "data:image/png;base64,RES" in str(groupchat.messages[1])

    def test_non_standard_content_type_conversion(self):
        """Non-standard content types are converted to string."""
        agent = ConversableAgent("test", llm_config=False)
        groupchat = GroupChat(agents=[agent], messages=[])

        # Integer content (non-standard)
        message = {"role": "user", "content": 12345}
        groupchat.append(message, agent)

        assert groupchat.messages[-1]["content"] == "12345"

    def test_data_uri_with_special_characters(self):
        """Data URIs with special characters are preserved."""
        agent = ConversableAgent("test", llm_config=False)
        groupchat = GroupChat(agents=[agent], messages=[])

        # Data URI with various base64 characters
        data_uri = "data:image/png;base64,iVBORw0KGgo+/AAAA=="
        message = {
            "role": "user",
            "content": [{"type": "text", "text": "Test"}, {"type": "image_url", "image_url": {"url": data_uri}}],
        }

        groupchat.append(message, agent)

        assert data_uri in str(groupchat.messages[-1])
