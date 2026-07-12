# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from mcp_ui_server import (
    UIActionResultIntent,
    UIActionResultLink,
    UIActionResultNotification,
    UIActionResultPrompt,
    UIActionResultToolCall,
    ui_action_result_intent,
    ui_action_result_link,
    ui_action_result_notification,
    ui_action_result_prompt,
    ui_action_result_tool_call,
)

__all__ = (
    "intent",
    "link",
    "notify",
    "prompt",
    "tool_call",
)


def tool_call(tool_name: str, params: dict[str, Any]) -> UIActionResultToolCall:
    """A ``tool`` action: ask the host to call the MCP tool ``tool_name(params)``."""
    return ui_action_result_tool_call(tool_name, params)


def prompt(text: str) -> UIActionResultPrompt:
    """A ``prompt`` action: send ``text`` into the conversation as a follow-up."""
    return ui_action_result_prompt(text)


def link(url: str) -> UIActionResultLink:
    """A ``link`` action: ask the host to open ``url``."""
    return ui_action_result_link(url)


def intent(name: str, params: dict[str, Any]) -> UIActionResultIntent:
    """An ``intent`` action: emit a host-defined ``name`` intent with ``params``."""
    return ui_action_result_intent(name, params)


def notify(message: str) -> UIActionResultNotification:
    """A ``notify`` action: send ``message`` to the host as a notification/log."""
    return ui_action_result_notification(message)
