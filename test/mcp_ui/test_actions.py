# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from mcp_ui_server import (
    UIActionResultIntent,
    UIActionResultLink,
    UIActionResultNotification,
    UIActionResultPrompt,
    UIActionResultToolCall,
)

from ag2.mcp_ui import intent, link, notify, post_message, prompt, tool_call


class TestUIActions:
    def test_tool_call(self) -> None:
        result = tool_call("add_to_cart", {"good_id": "42"})

        assert result == UIActionResultToolCall(
            type="tool",
            payload=UIActionResultToolCall.ToolCallPayload(toolName="add_to_cart", params={"good_id": "42"}),
        )

    def test_prompt(self) -> None:
        result = prompt("What is the weather?")

        assert result == UIActionResultPrompt(
            type="prompt",
            payload=UIActionResultPrompt.PromptPayload(prompt="What is the weather?"),
        )

    def test_link(self) -> None:
        result = link("https://docs.ag2.ai/")

        assert result == UIActionResultLink(
            type="link",
            payload=UIActionResultLink.LinkPayload(url="https://docs.ag2.ai/"),
        )

    def test_intent(self) -> None:
        result = intent("checkout", {"cart_id": "7"})

        assert result == UIActionResultIntent(
            type="intent",
            payload=UIActionResultIntent.IntentPayload(intent="checkout", params={"cart_id": "7"}),
        )

    def test_notify(self) -> None:
        result = notify("Widget loaded")

        assert result == UIActionResultNotification(
            type="notify",
            payload=UIActionResultNotification.NotificationPayload(message="Widget loaded"),
        )


class TestPostMessage:
    def test_builds_post_message_call(self) -> None:
        result = post_message(tool_call("add_to_cart", {"good_id": "42"}))

        assert result == (
            "window.parent.postMessage("
            "{&quot;type&quot;: &quot;tool&quot;, &quot;payload&quot;: "
            "{&quot;toolName&quot;: &quot;add_to_cart&quot;, &quot;params&quot;: {&quot;good_id&quot;: &quot;42&quot;}}}"
            ", &#x27;*&#x27;)"
        )

    def test_quotes_cannot_break_out_of_the_attribute(self) -> None:
        # An apostrophe in the payload must not terminate a quoted HTML attribute.
        result = post_message(prompt("What's the status of my order?"))

        assert "'" not in result
        assert '"' not in result
