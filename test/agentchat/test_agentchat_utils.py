# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

import pytest

from autogen import agentchat

TAG_PARSING_TESTS = [
    {
        "message": "Hello agent, can you take a look at this image <img http://example.com/image.png>",
        "expected": [{"tag": "img", "attr": {"src": "http://example.com/image.png"}}],
    },
    {
        "message": "Can you transcribe this audio? <audio http://example.com/au=dio.mp3>",
        "expected": [{"tag": "audio", "attr": {"src": "http://example.com/au=dio.mp3"}}],
    },
    {
        "message": "Can you describe what's in this image <img url='http://example.com/=image.png'>",
        "expected": [{"tag": "img", "attr": {"url": "http://example.com/=image.png"}}],
    },
    {
        "message": "Can you describe what's in this image <img http://example.com/image.png> and transcribe this audio? <audio http://example.com/audio.mp3>",
        "expected": [
            {"tag": "img", "attr": {"src": "http://example.com/image.png"}},
            {"tag": "audio", "attr": {"src": "http://example.com/audio.mp3"}},
        ],
    },
    {
        "message": "Can you generate this audio? <audio text='Hello I'm a robot' prompt='whisper'>",
        "expected": [{"tag": "audio", "attr": {"text": "Hello I'm a robot", "prompt": "whisper"}}],
    },
    {
        "message": "Can you describe what's in this image <img http://example.com/image.png width='100'> and this image <img http://hello.com/image=.png>?",
        "expected": [
            {"tag": "img", "attr": {"src": "http://example.com/image.png", "width": "100"}},
            {"tag": "img", "attr": {"src": "http://hello.com/image=.png"}},
        ],
    },
    {
        "message": "Text with no tags",
        "expected": [],
    },
]


def _delete_unused_keys(d: dict) -> None:
    if "match" in d:
        del d["match"]


@pytest.mark.parametrize("test_case", TAG_PARSING_TESTS)
def test_tag_parsing(test_case: dict[str, str | list[dict[str, str | dict[str, str]]]]) -> None:
    """Test the tag_parsing function."""
    message = test_case["message"]
    expected = test_case["expected"]
    tags = ["img", "audio", "random"]

    result = []
    for tag in tags:
        parsed_tags = agentchat.utils.parse_tags_from_content(tag, message)
        for item in parsed_tags:
            _delete_unused_keys(item)

        result.extend(parsed_tags)
    assert result == expected

    result = []
    for tag in tags:
        content = [{"type": "text", "text": message}]
        parsed_tags = agentchat.utils.parse_tags_from_content(tag, content)
        for item in parsed_tags:
            _delete_unused_keys(item)

        result.extend(parsed_tags)
    assert result == expected


def test_normalize_message_to_list() -> None:
    """Test the normalize_message_to_list function."""
    # Test with string input
    result = agentchat.utils.normalize_message_to_list("Hello, world!")
    assert result == [{"content": "Hello, world!", "role": "assistant"}]

    # Test with string input and custom role
    result = agentchat.utils.normalize_message_to_list("Hello, user!", role="user")
    assert result == [{"content": "Hello, user!", "role": "user"}]

    # Test with dict input
    message_dict = {"content": "Test message", "role": "user"}
    result = agentchat.utils.normalize_message_to_list(message_dict)
    assert result == [message_dict]

    # Test with list input
    message_list = [
        {"content": "Message 1", "role": "user"},
        {"content": "Message 2", "role": "assistant"},
    ]
    result = agentchat.utils.normalize_message_to_list(message_list)
    assert result == message_list


def test_message_to_dict() -> None:
    """Test the message_to_dict function."""
    # Test with string input
    result = agentchat.utils.message_to_dict("Hello, world!")
    assert result == {"content": "Hello, world!"}

    # Test with dict input
    message_dict = {"content": "Test message", "role": "user", "name": "Alice"}
    result = agentchat.utils.message_to_dict(message_dict)
    assert result == message_dict

    # Test with empty string
    result = agentchat.utils.message_to_dict("")
    assert result == {"content": ""}


def test_normalize_message_to_oai() -> None:
    """Test the normalize_message_to_oai function."""
    # Test with string message
    valid, oai_message = agentchat.utils.normalize_message_to_oai("Hello, world!", name="Agent1")
    assert valid is True
    assert oai_message["content"] == "Hello, world!"
    assert oai_message["role"] == "assistant"
    assert oai_message["name"] == "Agent1"

    # Test with dict message containing content
    message = {"content": "Test message"}
    valid, oai_message = agentchat.utils.normalize_message_to_oai(message, name="Agent2", role="user")
    assert valid is True
    assert oai_message["content"] == "Test message"
    assert oai_message["role"] == "user"
    assert oai_message["name"] == "Agent2"

    # Test with function_call (should set role to assistant and content to None)
    message = {"function_call": {"name": "test_func", "arguments": "{}"}}
    valid, oai_message = agentchat.utils.normalize_message_to_oai(message, name="Agent3")
    assert valid is True
    assert oai_message["content"] is None
    assert oai_message["role"] == "assistant"
    assert oai_message["function_call"] == {"name": "test_func", "arguments": "{}"}

    # Test with tool_calls (should set role to assistant and content to None)
    message = {"tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "test_func"}}]}
    valid, oai_message = agentchat.utils.normalize_message_to_oai(message, name="Agent4")
    assert valid is True
    assert oai_message["content"] is None
    assert oai_message["role"] == "assistant"
    assert len(oai_message["tool_calls"]) == 1

    # Test with tool responses (role should be "tool")
    message = {
        "role": "tool",
        "content": "Result",
        "tool_responses": [{"tool_call_id": "call_1", "content": "Result"}],
    }
    valid, oai_message = agentchat.utils.normalize_message_to_oai(message, name="Agent5")
    assert valid is True
    assert oai_message["role"] == "tool"
    assert oai_message["tool_responses"][0]["content"] == "Result"

    # Test with override_role
    message = {"content": "Test", "override_role": "system"}
    valid, oai_message = agentchat.utils.normalize_message_to_oai(message, name="Agent6")
    assert valid is True
    assert oai_message["role"] == "system"

    # Test invalid message (no content, no function_call, no tool_calls)
    message = {"invalid_field": "value"}
    valid, oai_message = agentchat.utils.normalize_message_to_oai(message, name="Agent7")
    assert valid is False

    # Test preserve_custom_fields=False
    message = {
        "content": "Test",
        "custom_field": "custom_value",
        "role": "user",
    }
    valid, oai_message = agentchat.utils.normalize_message_to_oai(message, name="Agent8", preserve_custom_fields=False)
    assert valid is True
    assert "custom_field" not in oai_message
    assert oai_message["content"] == "Test"

    # Test preserve_custom_fields=True (default)
    valid, oai_message = agentchat.utils.normalize_message_to_oai(message, name="Agent9", preserve_custom_fields=True)
    assert valid is True
    assert oai_message["custom_field"] == "custom_value"

    # Test with name already in message (should not append name field for function/tool calls)
    message = {
        "content": None,
        "function_call": {"name": "test_func"},
        "name": "existing_name",
    }
    valid, oai_message = agentchat.utils.normalize_message_to_oai(message, name="Agent10")
    assert valid is True
    assert oai_message["name"] == "existing_name"
    assert oai_message["role"] == "assistant"

    # Test tool_responses with list content
    message = {
        "role": "tool",
        "content": [{"type": "text", "text": "item1"}, {"type": "text", "text": "item2"}],
        "tool_responses": [
            {
                "tool_call_id": "call_1",
                "content": [{"type": "text", "text": "item1"}, {"type": "text", "text": "item2"}],
            }
        ],
    }
    valid, oai_message = agentchat.utils.normalize_message_to_oai(message, name="Agent11")
    assert valid is True
    assert isinstance(oai_message["tool_responses"][0]["content"], str)


if __name__ == "__main__":
    test_tag_parsing(TAG_PARSING_TESTS[0])
    test_normalize_message_to_list()
    test_message_to_dict()
    test_normalize_message_to_oai()
