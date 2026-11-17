# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Renderer→agent input: envelope validation, decoding into typed results, and the prompts the agent gets."""

from .._sanitize import sanitize_for_prompt
from ._decode import decode_envelopes, parse_incoming_message
from ._prompts import action_to_prompt, error_to_prompt, function_response_to_prompt, incoming_prompts
from ._renderer import A2UIRendererInput, read_renderer_input
from ._results import (
    A2UIFunctionError,
    A2UIIncomingAction,
    A2UIIncomingActionResult,
    A2UIIncomingError,
    A2UIIncomingErrorResult,
    A2UIIncomingFunctionCall,
    A2UIIncomingFunctionCallResult,
    A2UIIncomingFunctionError,
    A2UIIncomingFunctionResponse,
    A2UIIncomingFunctionResponseResult,
    A2UIIncomingInvalidFunctionCallResult,
    A2UIIncomingParseResult,
    A2UIIncomingSurfaceError,
    A2UIIncomingUnknownResult,
)
from ._rules import envelope_problem

__all__ = (
    "A2UIFunctionError",
    "A2UIIncomingAction",
    "A2UIIncomingActionResult",
    "A2UIIncomingError",
    "A2UIIncomingErrorResult",
    "A2UIIncomingFunctionCall",
    "A2UIIncomingFunctionCallResult",
    "A2UIIncomingFunctionError",
    "A2UIIncomingFunctionResponse",
    "A2UIIncomingFunctionResponseResult",
    "A2UIIncomingInvalidFunctionCallResult",
    "A2UIIncomingParseResult",
    "A2UIIncomingSurfaceError",
    "A2UIIncomingUnknownResult",
    "A2UIRendererInput",
    "action_to_prompt",
    "decode_envelopes",
    "envelope_problem",
    "error_to_prompt",
    "function_response_to_prompt",
    "incoming_prompts",
    "parse_incoming_message",
    "read_renderer_input",
    "sanitize_for_prompt",
)
