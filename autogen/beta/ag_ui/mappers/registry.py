# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from collections.abc import Callable
from typing import Any

from autogen.beta.config import ModelConfig
from autogen.beta.events import BuiltinToolCallEvent, BuiltinToolResultEvent
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

CallMapper = Callable[[str, str, dict[str, Any]], BuiltinToolCallEvent | None]
ResultMapper = Callable[[object, str], BuiltinToolResultEvent | None]

_CALL_MAPPERS: dict[type[ModelConfig], CallMapper] = {}
_RESULT_MAPPERS: dict[type[ModelConfig], ResultMapper] = {}

try:
    from autogen.beta.config import OpenAIConfig, OpenAIResponsesConfig

    from .openai import openai_call_from_agui, openai_result_from_agui
except ImportError:
    pass
else:
    for _openai_config in (OpenAIConfig, OpenAIResponsesConfig):
        _CALL_MAPPERS[_openai_config] = openai_call_from_agui
        _RESULT_MAPPERS[_openai_config] = openai_result_from_agui

try:
    from autogen.beta.config import AnthropicConfig

    from .anthropic import anthropic_call_from_agui, anthropic_result_from_agui
except ImportError:
    pass
else:
    _CALL_MAPPERS[AnthropicConfig] = anthropic_call_from_agui
    _RESULT_MAPPERS[AnthropicConfig] = anthropic_result_from_agui

try:
    from autogen.beta.config import GeminiConfig, VertexAIConfig

    from .gemini import gemini_call_from_agui, gemini_result_from_agui
except ImportError:
    pass
else:
    for _gemini_config in (GeminiConfig, VertexAIConfig):
        _CALL_MAPPERS[_gemini_config] = gemini_call_from_agui
        _RESULT_MAPPERS[_gemini_config] = gemini_result_from_agui


def call_from_agui(
    config: ModelConfig | None,
    name: str,
    call_id: str,
    arguments: str,
) -> BuiltinToolCallEvent | None:
    mapper = _CALL_MAPPERS.get(type(config)) if config is not None else None
    if mapper is None:
        return None
    payload = _safe_load(arguments)
    if payload is None:
        return None
    return mapper(name, call_id, payload)


def result_from_agui(
    config: ModelConfig | None,
    call: BuiltinToolCallEvent,
    content: str,
) -> BuiltinToolResultEvent | None:
    mapper = _RESULT_MAPPERS.get(type(config)) if config is not None else None
    if mapper is None:
        return None
    return mapper(call, content)


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
    "call_from_agui",
    "result_from_agui",
)
