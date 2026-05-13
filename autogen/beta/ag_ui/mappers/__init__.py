# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from collections.abc import Callable
from typing import Any

from autogen.beta.events import BuiltinToolCallEvent, BuiltinToolResultEvent
from autogen.beta.exceptions import missing_optional_dependency
from autogen.beta.tools.builtin.code_execution import CODE_EXECUTION_TOOL_NAME
from autogen.beta.tools.builtin.image_generation import IMAGE_GENERATION_TOOL_NAME
from autogen.beta.tools.builtin.web_fetch import WEB_FETCH_TOOL_NAME
from autogen.beta.tools.builtin.web_search import WEB_SEARCH_TOOL_NAME

logger = logging.getLogger(__name__)

KNOWN_BUILTIN_NAMES: frozenset[str] = frozenset({
    WEB_SEARCH_TOOL_NAME,
    WEB_FETCH_TOOL_NAME,
    CODE_EXECUTION_TOOL_NAME,
    IMAGE_GENERATION_TOOL_NAME,
})

# Each provider mapper imports the corresponding SDK at module load. If the
# optional extra is not installed we fall back to `missing_optional_dependency`
# stubs for the public exports (matching the pattern used in
# `autogen.beta.config.__init__`) and skip the mapper in the dispatcher.

try:
    from .openai import openai_call_from_agui, openai_result_from_agui
except ImportError as e:
    openai_call_from_agui = missing_optional_dependency("openai_call_from_agui", "openai", e)  # type: ignore[assignment]
    openai_result_from_agui = missing_optional_dependency("openai_result_from_agui", "openai", e)  # type: ignore[assignment]
    _HAS_OPENAI = False
else:
    _HAS_OPENAI = True

try:
    from .anthropic import anthropic_call_from_agui, anthropic_result_from_agui
except ImportError as e:
    anthropic_call_from_agui = missing_optional_dependency("anthropic_call_from_agui", "anthropic", e)  # type: ignore[assignment]
    anthropic_result_from_agui = missing_optional_dependency("anthropic_result_from_agui", "anthropic", e)  # type: ignore[assignment]
    _HAS_ANTHROPIC = False
else:
    _HAS_ANTHROPIC = True

try:
    from .gemini import gemini_call_from_agui, gemini_result_from_agui
except ImportError as e:
    gemini_call_from_agui = missing_optional_dependency("gemini_call_from_agui", "gemini", e)  # type: ignore[assignment]
    gemini_result_from_agui = missing_optional_dependency("gemini_result_from_agui", "gemini", e)  # type: ignore[assignment]
    _HAS_GEMINI = False
else:
    _HAS_GEMINI = True


_CALL_MAPPERS: list[Callable[[str, str, dict[str, Any]], BuiltinToolCallEvent | None]] = []
_RESULT_MAPPERS: list[Callable[[object, str], BuiltinToolResultEvent | None]] = []

if _HAS_OPENAI:
    _CALL_MAPPERS.append(openai_call_from_agui)
    _RESULT_MAPPERS.append(openai_result_from_agui)

if _HAS_ANTHROPIC:
    _CALL_MAPPERS.append(anthropic_call_from_agui)
    _RESULT_MAPPERS.append(anthropic_result_from_agui)

if _HAS_GEMINI:
    _CALL_MAPPERS.append(gemini_call_from_agui)
    _RESULT_MAPPERS.append(gemini_result_from_agui)


def call_from_agui(name: str, call_id: str, arguments: str) -> BuiltinToolCallEvent | None:
    payload = _safe_load(arguments)
    if payload is None:
        return None

    for fn in _CALL_MAPPERS:
        if (event := fn(name, call_id, payload)) is not None:
            return event

    return None


def result_from_agui(
    call: BuiltinToolCallEvent,
    content: str,
) -> BuiltinToolResultEvent | None:
    for fn in _RESULT_MAPPERS:
        if (event := fn(call, content)) is not None:
            return event
    return None


def _safe_load(arguments: str) -> dict[str, Any] | None:
    if not arguments:
        return {}
    try:
        loaded = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded


__all__ = (
    "KNOWN_BUILTIN_NAMES",
    "anthropic_call_from_agui",
    "anthropic_result_from_agui",
    "call_from_agui",
    "gemini_call_from_agui",
    "gemini_result_from_agui",
    "openai_call_from_agui",
    "openai_result_from_agui",
    "result_from_agui",
)
