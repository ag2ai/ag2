# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.exceptions import missing_optional_dependency

from .from_agui import map_agui_messages_to_events
from .history import events_to_agui_messages
from .registry import KNOWN_BUILTIN_NAMES, call_from_agui, result_from_agui

try:
    from .openai import openai_call_from_agui, openai_result_from_agui
except ImportError as e:
    openai_call_from_agui = missing_optional_dependency("openai_call_from_agui", "openai", e)  # type: ignore[assignment]
    openai_result_from_agui = missing_optional_dependency("openai_result_from_agui", "openai", e)  # type: ignore[assignment]

try:
    from .anthropic import anthropic_call_from_agui, anthropic_result_from_agui
except ImportError as e:
    anthropic_call_from_agui = missing_optional_dependency("anthropic_call_from_agui", "anthropic", e)  # type: ignore[assignment]
    anthropic_result_from_agui = missing_optional_dependency("anthropic_result_from_agui", "anthropic", e)  # type: ignore[assignment]

try:
    from .gemini import gemini_call_from_agui, gemini_result_from_agui
except ImportError as e:
    gemini_call_from_agui = missing_optional_dependency("gemini_call_from_agui", "gemini", e)  # type: ignore[assignment]
    gemini_result_from_agui = missing_optional_dependency("gemini_result_from_agui", "gemini", e)  # type: ignore[assignment]

__all__ = (
    "KNOWN_BUILTIN_NAMES",
    "anthropic_call_from_agui",
    "anthropic_result_from_agui",
    "call_from_agui",
    "events_to_agui_messages",
    "gemini_call_from_agui",
    "gemini_result_from_agui",
    "map_agui_messages_to_events",
    "openai_call_from_agui",
    "openai_result_from_agui",
    "result_from_agui",
)
