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

from ag2.mcp_ui import actions


class TestUIActions:
    def test_tool_call(self) -> None:
        result = actions.tool_call("add_to_cart", {"good_id": "42"})

        assert result == UIActionResultToolCall(
            type="tool",
            payload=UIActionResultToolCall.ToolCallPayload(toolName="add_to_cart", params={"good_id": "42"}),
        )

    def test_prompt(self) -> None:
        result = actions.prompt("What is the weather?")

        assert result == UIActionResultPrompt(
            type="prompt",
            payload=UIActionResultPrompt.PromptPayload(prompt="What is the weather?"),
        )

    def test_link(self) -> None:
        result = actions.link("https://docs.ag2.ai/")

        assert result == UIActionResultLink(
            type="link",
            payload=UIActionResultLink.LinkPayload(url="https://docs.ag2.ai/"),
        )

    def test_intent(self) -> None:
        result = actions.intent("checkout", {"cart_id": "7"})

        assert result == UIActionResultIntent(
            type="intent",
            payload=UIActionResultIntent.IntentPayload(intent="checkout", params={"cart_id": "7"}),
        )

    def test_notify(self) -> None:
        result = actions.notify("Widget loaded")

        assert result == UIActionResultNotification(
            type="notify",
            payload=UIActionResultNotification.NotificationPayload(message="Widget loaded"),
        )
