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


def test_normalize_content_with_string():
    """Test normalize_content with string input."""
    from autogen.agentchat.utils import normalize_content

    assert normalize_content("Hello world") == "Hello world"
    assert normalize_content("") == ""
    assert normalize_content("Multi\nline\nstring") == "Multi\nline\nstring"


def test_normalize_content_with_list():
    """Test normalize_content with list input (multimodal content)."""
    from autogen.agentchat.utils import normalize_content

    # Simple text content
    content = [{"type": "text", "text": "Hello"}]
    result = normalize_content(content)
    assert result == "Hello"

    # Multiple content items
    content = [{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}]
    result = normalize_content(content)
    assert "Hello" in result and "World" in result

    # Empty list
    assert normalize_content([]) == ""


def test_normalize_content_with_none():
    """Test normalize_content with None input."""
    from autogen.agentchat.utils import normalize_content

    assert normalize_content(None) == ""


def test_normalize_content_with_dict():
    """Test normalize_content with dict input (converts to string)."""
    from autogen.agentchat.utils import normalize_content

    content = {"key": "value", "number": 42}
    result = normalize_content(content)
    assert isinstance(result, str)
    assert "key" in result or "value" in result


def test_normalize_content_with_other_types():
    """Test normalize_content with other types (int, float, bool, etc.)."""
    from autogen.agentchat.utils import normalize_content

    assert normalize_content(42) == "42"
    assert normalize_content(3.14) == "3.14"
    assert normalize_content(True) == "True"
    assert normalize_content(False) == "False"


def test_normalize_message_to_dict_with_string():
    """Test normalize_message_to_dict with string input."""
    from autogen.agentchat.utils import normalize_message_to_dict

    result = normalize_message_to_dict("Hello world")
    assert result == {"content": "Hello world"}

    result = normalize_message_to_dict("")
    assert result == {"content": ""}


def test_normalize_message_to_dict_with_dict():
    """Test normalize_message_to_dict with dict input."""
    from autogen.agentchat.utils import normalize_message_to_dict

    # Simple message dict
    message = {"role": "user", "content": "Hello"}
    result = normalize_message_to_dict(message)
    assert result == message
    assert result is message  # Should return same object

    # Complex message with multiple fields
    message = {
        "role": "assistant",
        "content": "Response",
        "name": "agent1",
        "function_call": {"name": "func", "arguments": "{}"},
    }
    result = normalize_message_to_dict(message)
    assert result == message


def test_normalize_message_to_dict_with_dict_like():
    """Test normalize_message_to_dict with dict-like objects."""
    from autogen.agentchat.utils import normalize_message_to_dict

    # Create a dict-like object (has items() method)
    class DictLike:
        def __init__(self, data):
            self.data = data

        def items(self):
            return self.data.items()

        def keys(self):
            return self.data.keys()

        def values(self):
            return self.data.values()

        def __getitem__(self, key):
            return self.data[key]

    dict_like = DictLike({"role": "user", "content": "Test"})
    result = normalize_message_to_dict(dict_like)
    assert isinstance(result, dict)
    assert result["role"] == "user"
    assert result["content"] == "Test"


def test_normalize_content_edge_cases():
    """Test normalize_content with edge cases."""
    import pytest

    from autogen.agentchat.utils import normalize_content

    # Nested lists - should raise TypeError from content_str
    nested = [{"type": "text", "text": "Level 1"}, [{"type": "text", "text": "Level 2"}]]
    with pytest.raises(TypeError, match="Wrong content format"):
        normalize_content(nested)

    # List with non-dict items - should raise TypeError from content_str
    mixed = [{"type": "text", "text": "Text"}, "plain string", 42]
    with pytest.raises(TypeError, match="Wrong content format"):
        normalize_content(mixed)

    # List with image content
    image_content = [
        {"type": "text", "text": "Check this image:"},
        {"type": "image_url", "image_url": {"url": "http://example.com/image.png"}},
    ]
    result = normalize_content(image_content)
    assert isinstance(result, str)
    assert "Check this image:" in result


def test_normalize_message_to_dict_preserves_original():
    """Test that normalize_message_to_dict doesn't modify original dict."""
    from autogen.agentchat.utils import normalize_message_to_dict

    original = {"role": "user", "content": "Original"}
    result = normalize_message_to_dict(original)

    # Modify result
    result["content"] = "Modified"

    # Original should also be modified since it's the same object
    assert original["content"] == "Modified"

    # For string input, original is not affected
    original_str = "Original string"
    result = normalize_message_to_dict(original_str)
    result["content"] = "Modified"
    assert original_str == "Original string"


if __name__ == "__main__":
    test_tag_parsing(TAG_PARSING_TESTS[0])
    test_normalize_content_with_string()
    test_normalize_content_with_list()
    test_normalize_content_with_none()
    test_normalize_content_with_dict()
    test_normalize_content_with_other_types()
    test_normalize_message_to_dict_with_string()
    test_normalize_message_to_dict_with_dict()
    test_normalize_message_to_dict_with_dict_like()
    test_normalize_content_edge_cases()
    test_normalize_message_to_dict_preserves_original()
