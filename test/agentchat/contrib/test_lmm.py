# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# !/usr/bin/env python3 -m pytest

import unittest
from unittest.mock import MagicMock

import pytest
from pytest import MonkeyPatch

import autogen
from autogen.agentchat import GroupChat
from autogen.agentchat.contrib.img_utils import get_pil_image
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.import_utils import run_for_optional_imports
from test.const import MOCK_OPEN_AI_API_KEY

base64_encoded_image = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4"
    "//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="
)


@run_for_optional_imports(["PIL"], "unknown")
@pytest.mark.lmm
class TestMultimodalConversableAgent:
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.agent = MultimodalConversableAgent(
            name="TestAgent",
            llm_config={
                "timeout": 600,
                "seed": 42,
                "config_list": [
                    {"api_type": "openai", "model": "gpt-4-vision-preview", "api_key": MOCK_OPEN_AI_API_KEY}
                ],
            },
        )

    def test_system_message(self):
        # Test default system message
        assert self.agent.system_message == [
            {
                "type": "text",
                "text": "You are a helpful AI assistant.",
            }
        ]

        # Test updating system message
        new_message = f"We will discuss <img {base64_encoded_image}> in this conversation."
        self.agent.update_system_message(new_message)

        pil_image = get_pil_image(base64_encoded_image)
        assert self.agent.system_message == [
            {"type": "text", "text": "We will discuss "},
            {"type": "image_url", "image_url": {"url": pil_image}},
            {"type": "text", "text": " in this conversation."},
        ]

    def test_message_to_dict(self):
        # Test string message
        message_str = "Hello"
        expected_dict = {"content": [{"type": "text", "text": "Hello"}]}
        assert self.agent._message_to_dict(message_str) == expected_dict

        # Test list message
        message_list = [{"type": "text", "text": "Hello"}]
        expected_dict = {"content": message_list}
        assert self.agent._message_to_dict(message_list) == expected_dict

        # Test dictionary message
        message_dict = {"content": [{"type": "text", "text": "Hello"}]}
        assert self.agent._message_to_dict(message_dict) == message_dict

    def test_print_received_message(self):
        sender = ConversableAgent(name="SenderAgent", llm_config=False, code_execution_config=False)
        message_str = "Hello"
        self.agent._print_received_message = MagicMock()  # Mocking print method to avoid actual print
        self.agent._print_received_message(message_str, sender)
        self.agent._print_received_message.assert_called_with(message_str, sender)


@run_for_optional_imports(["PIL"], "unknown")
@pytest.mark.lmm
def test_group_chat_with_lmm(monkeypatch: MonkeyPatch):
    """Tests the group chat functionality with two MultimodalConversable Agents.
    Verifies that the chat is correctly limited by the max_round parameter.
    Each agent is set to describe an image in a unique style, but the chat should not exceed the specified max_rounds.
    """
    # Configuration parameters
    max_round = 5
    max_consecutive_auto_reply = 10
    llm_config = False

    # Creating two MultimodalConversable Agents with different descriptive styles
    agent1 = MultimodalConversableAgent(
        name="image-explainer-1",
        max_consecutive_auto_reply=max_consecutive_auto_reply,
        llm_config=llm_config,
        system_message="Your image description is poetic and engaging.",
    )
    agent2 = MultimodalConversableAgent(
        name="image-explainer-2",
        max_consecutive_auto_reply=max_consecutive_auto_reply,
        llm_config=llm_config,
        system_message="Your image description is factual and to the point.",
    )

    # Creating a user proxy agent for initiating the group chat
    user_proxy = autogen.UserProxyAgent(
        name="User_proxy",
        system_message="Ask both image explainer 1 and 2 for their description.",
        human_input_mode="NEVER",  # Options: 'ALWAYS' or 'NEVER'
        max_consecutive_auto_reply=max_consecutive_auto_reply,
        code_execution_config=False,
    )

    # Mock speaker selection so it doesn't require a GroupChatManager with an LLM
    monkeypatch.setattr(GroupChat, "_auto_select_speaker", lambda *args, **kwargs: agent1)

    # Setting up the group chat
    groupchat = autogen.GroupChat(agents=[agent1, agent2, user_proxy], messages=[], max_round=max_round)
    group_chat_manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=None)

    # Initiating the group chat and observing the number of rounds
    user_proxy.initiate_chat(group_chat_manager, message=f"What do you see? <img {base64_encoded_image}>")

    # Assertions to check if the number of rounds does not exceed max_round
    assert all(len(arr) <= max_round for arr in agent1._oai_messages.values()), "Agent 1 exceeded max rounds"
    assert all(len(arr) <= max_round for arr in agent2._oai_messages.values()), "Agent 2 exceeded max rounds"
    assert all(len(arr) <= max_round for arr in user_proxy._oai_messages.values()), "User proxy exceeded max rounds"


@run_for_optional_imports(["PIL"], "unknown")
@pytest.mark.lmm
class TestMultimodalConversableAgentImageTagProcessing:
    """Test that <img> tags are processed correctly in messages sent to MultimodalConversableAgent.

    This tests the fix for the regression introduced in v0.10.0 where <img> tags
    stopped being processed because _append_oai_message was refactored to use
    normilize_message_to_oai instead of self._message_to_dict.
    """

    @pytest.fixture(autouse=True)
    def setUp(self):
        self.visual_agent = MultimodalConversableAgent(
            name="visual_agent",
            llm_config={
                "timeout": 600,
                "seed": 42,
                "config_list": [
                    {"api_type": "openai", "model": "gpt-4-vision-preview", "api_key": MOCK_OPEN_AI_API_KEY}
                ],
            },
        )
        self.user_agent = ConversableAgent(
            name="user_agent",
            human_input_mode="NEVER",
            llm_config=False,
            code_execution_config=False,
        )

    def test_img_tag_processed_in_received_message(self):
        """Test that <img> tags in messages are converted to multimodal content.

        This is the key regression test - in v0.9.10 this worked, in v0.10.0 it stopped
        working because _append_oai_message no longer called self._message_to_dict.
        """
        # Send a message with an <img> tag from user to visual agent
        message_with_img = f"Describe this image: <img {base64_encoded_image}>"

        # Call receive directly to test the _append_oai_message flow
        self.visual_agent.receive(message_with_img, self.user_agent, request_reply=False)

        # Get the stored message from _oai_messages
        stored_messages = self.visual_agent._oai_messages[self.user_agent]
        assert len(stored_messages) == 1, "Should have one stored message"

        stored_message = stored_messages[0]
        content = stored_message["content"]

        # The content should be a list (multimodal format), not a string
        assert isinstance(content, list), (
            f"Content should be a list (multimodal format), not {type(content)}. "
            "This indicates <img> tags were not processed."
        )

        # Should have text and image_url parts
        text_parts = [item for item in content if item.get("type") == "text"]
        image_parts = [item for item in content if item.get("type") == "image_url"]

        assert len(text_parts) >= 1, "Should have at least one text part"
        assert len(image_parts) == 1, "Should have exactly one image part"

        # The text should contain "Describe this image:"
        all_text = " ".join(item.get("text", "") for item in text_parts)
        assert "Describe this image:" in all_text, "Text content should be preserved"

        # The image_url should contain the PIL image (not the raw string)
        image_item = image_parts[0]
        assert "image_url" in image_item, "Image part should have image_url field"
        assert "url" in image_item["image_url"], "image_url should have url field"

    def test_img_tag_processed_in_sent_message(self):
        """Test that <img> tags are also processed when visual agent sends a message."""
        message_with_img = f"Here is the analysis: <img {base64_encoded_image}>"

        # Visual agent sends a message to user
        # Using send directly to test the flow
        self.visual_agent.send(message_with_img, self.user_agent, request_reply=False)

        # The message stored in visual_agent's _oai_messages (as sent to user_agent)
        stored_messages = self.visual_agent._oai_messages[self.user_agent]
        assert len(stored_messages) == 1, "Should have one stored message"

        stored_message = stored_messages[0]
        content = stored_message["content"]

        # Should be multimodal format
        assert isinstance(content, list), "Sent message content should be multimodal format"

    def test_message_without_img_tag_preserved(self):
        """Test that regular messages without <img> tags still work correctly."""
        regular_message = "This is a regular text message without any images."

        self.visual_agent.receive(regular_message, self.user_agent, request_reply=False)

        stored_messages = self.visual_agent._oai_messages[self.user_agent]
        assert len(stored_messages) == 1

        stored_message = stored_messages[0]
        content = stored_message["content"]

        # Even without images, _message_to_dict wraps in multimodal format
        assert isinstance(content, list), "Content should still be in list format"
        assert len(content) == 1, "Should have one text item"
        assert content[0]["type"] == "text"
        assert content[0]["text"] == regular_message

    def test_dict_message_with_string_content_processed(self):
        """Test that dict messages with string content containing <img> tags are processed."""
        message_dict = {"content": f"Check this: <img {base64_encoded_image}>", "role": "user"}

        self.visual_agent.receive(message_dict, self.user_agent, request_reply=False)

        stored_messages = self.visual_agent._oai_messages[self.user_agent]
        stored_message = stored_messages[0]
        content = stored_message["content"]

        # Should be converted to multimodal format
        assert isinstance(content, list), "Dict message content should be converted to multimodal format"

        image_parts = [item for item in content if item.get("type") == "image_url"]
        assert len(image_parts) == 1, "Should have processed the <img> tag"


if __name__ == "__main__":
    unittest.main()
