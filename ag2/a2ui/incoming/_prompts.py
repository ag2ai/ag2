# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Rewrite decoded renderer→agent interactions as sanitised prompts for the agent."""

import json
import logging
from collections.abc import Callable, Iterable, Iterator
from typing import assert_never

from .._sanitize import sanitize_for_prompt
from ..actions import A2UIEventAction
from ._results import (
    A2UIIncomingAction,
    A2UIIncomingActionResult,
    A2UIIncomingError,
    A2UIIncomingErrorResult,
    A2UIIncomingFunctionCallResult,
    A2UIIncomingFunctionError,
    A2UIIncomingFunctionResponse,
    A2UIIncomingFunctionResponseResult,
    A2UIIncomingInvalidFunctionCallResult,
    A2UIIncomingParseResult,
    A2UIIncomingUnknownResult,
)

logger = logging.getLogger(__name__)


def action_to_prompt(action: A2UIIncomingAction) -> str | None:
    """Rewrite a client A2UI ``action`` as a generic LLM instruction.

    Used only for clicks with **no** registered server action: buttons are
    rendered dynamically by the LLM, so the click is rewritten generically and
    the LLM continues the conversation (a registered action is handled on the
    server and never reaches this path). Returns ``None`` only for a nameless
    (malformed) action. All client-supplied values pass through
    :func:`sanitize_for_prompt`.
    """
    if not action.name:
        return None

    name = sanitize_for_prompt(action.name)
    ctx_json = sanitize_for_prompt(json.dumps(action.context))
    origin_bits: list[str] = []
    if action.surface_id:
        origin_bits.append(f"surface={sanitize_for_prompt(action.surface_id)}")
    if action.source_component_id:
        origin_bits.append(f"component={sanitize_for_prompt(action.source_component_id)}")
    if action.timestamp:
        origin_bits.append(f"at={sanitize_for_prompt(action.timestamp)}")
    origin = f" ({', '.join(origin_bits)})" if origin_bits else ""

    user_message = f" User message: {sanitize_for_prompt(action.user_message)}" if action.user_message else ""

    return f"The user clicked the '{name}' button{origin}. Context: {ctx_json}{user_message}"


def error_to_prompt(err: A2UIIncomingError) -> str:
    """Rewrite a client-reported A2UI ``error`` as a corrective LLM instruction.

    Handles both error arms: a surface-scoped validation/render error, and the
    v1.0 generic error reporting a failed server-initiated ``callRendererFunction``
    (carrying a ``functionCallId`` instead of a surface).
    """
    code = sanitize_for_prompt(err.code) if err.code else "(none)"
    message = sanitize_for_prompt(err.message)

    if isinstance(err, A2UIIncomingFunctionError):
        fc_id = sanitize_for_prompt(err.function_call_id)
        return (
            f"The client reported an A2UI error for function call '{fc_id}'. "
            f"Code: {code}. Message: {message}. "
            "Adjust your approach and continue without relying on that function's result."
        )

    path_hint = sanitize_for_prompt(err.path) if err.path else "(unknown)"
    surface = sanitize_for_prompt(err.surface_id)
    return (
        f"The client reported an A2UI error on surface '{surface}'. "
        f"Code: {code}. Path: {path_hint}. Message: {message}. "
        "Please regenerate the UI with this issue corrected."
    )


def function_response_to_prompt(fr: A2UIIncomingFunctionResponse) -> str:
    """Rewrite a client ``rendererFunctionResponse`` (v1.0) as a continuation instruction.

    The client has run a server-initiated ``callRendererFunction`` and returned
    its result or its failure. All client-supplied values pass through
    :func:`sanitize_for_prompt`.
    """
    fc_id = sanitize_for_prompt(fr.function_call_id)
    if fr.error is not None:
        code = sanitize_for_prompt(fr.error.code) if fr.error.code else "(none)"
        return (
            f"The client's function call (id={fc_id}) failed. Code: {code}. "
            f"Message: {sanitize_for_prompt(fr.error.message)}. "
            "Adjust your approach and continue without relying on that function's result."
        )
    try:
        value_json = sanitize_for_prompt(json.dumps(fr.value))
    except (TypeError, ValueError):
        logger.warning(
            "rendererFunctionResponse value for call '%s' is not JSON-serializable; using a placeholder.", fc_id
        )
        value_json = "<non-serializable>"
    return (
        f"The client executed the function call (id={fc_id}) and returned: {value_json}. Use this result to continue."
    )


def incoming_prompts(
    interactions: Iterable[A2UIIncomingParseResult],
    resolve_action: Callable[[str], A2UIEventAction | None],
) -> Iterator[str]:
    """Rewrite decoded client→server interactions into prompt strings for the agent.

    Shared by every transport so the security-sensitive ``sanitize_for_prompt``
    handling lives in one place. The server answers some interactions itself, so
    they yield no prompt: a click on a **registered** action (``resolve_action``
    returns its declaration) runs via ``run_server_action``, and every
    ``callAgentFunction``, malformed or not, is answered via ``run_client_interactions``. A nameless
    ``action`` is dropped with a warning.
    """
    for result in interactions:
        if isinstance(result, A2UIIncomingActionResult):
            if resolve_action(result.action.name) is not None:
                continue
            prompt = action_to_prompt(result.action)
            if prompt is None:
                logger.warning("Dropping A2UI action with no name.")
                continue
            yield prompt
        elif isinstance(result, (A2UIIncomingFunctionCallResult, A2UIIncomingInvalidFunctionCallResult)):
            continue
        elif isinstance(result, A2UIIncomingFunctionResponseResult):
            yield function_response_to_prompt(result.function_response)
        elif isinstance(result, A2UIIncomingErrorResult):
            yield error_to_prompt(result.error)
        elif isinstance(result, A2UIIncomingUnknownResult):
            continue
        else:
            assert_never(result)
