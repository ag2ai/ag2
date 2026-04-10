# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from unittest.mock import MagicMock

import pytest

from autogen.beta import Context
from autogen.beta.events import BaseEvent, ModelMessage, ModelResponse, TextInput
from autogen.beta.middleware import RetryMiddleware


class TransientError(Exception):
    pass


class PermanentError(Exception):
    pass


@pytest.mark.asyncio()
async def test_llm_retry_calls_next_once_when_successful(mock: MagicMock) -> None:
    retry_middleware = RetryMiddleware(max_retries=3)

    async def llm_call(events: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        mock.llm_call(events)
        return ModelResponse(ModelMessage("result"))

    middleware = retry_middleware(TextInput("Hi!"), mock)
    response = await middleware.on_llm_call(llm_call, [TextInput("Hi!")], mock)

    assert response == ModelResponse(ModelMessage("result"))
    mock.llm_call.assert_called_once_with([TextInput("Hi!")])


@pytest.mark.asyncio()
async def test_llm_retry_retries_matching_errors_until_success(mock: MagicMock) -> None:
    retry_middleware = RetryMiddleware(max_retries=2, retry_on=(TransientError,))
    attempts = 0

    async def llm_call(events: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        mock.llm_call(events)
        if attempts < 3:
            raise TransientError(f"transient failure {attempts}")
        return ModelResponse(ModelMessage("result"))

    middleware = retry_middleware(TextInput("Hi!"), mock)
    response = await middleware.on_llm_call(llm_call, [TextInput("Hi!")], mock)

    assert response == ModelResponse(ModelMessage("result"))
    assert mock.llm_call.call_count == attempts == 3


@pytest.mark.asyncio()
async def test_llm_retry_raises_after_exhausting_retries(mock: MagicMock) -> None:
    retry_middleware = RetryMiddleware(max_retries=2, retry_on=(TransientError,))

    async def llm_call(events: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        mock.llm_call(events)
        raise TransientError("still failing")

    middleware = retry_middleware(TextInput("Hi!"), mock)
    with pytest.raises(TransientError, match="still failing"):
        await middleware.on_llm_call(llm_call, [TextInput("Hi!")], mock)

    assert mock.llm_call.call_count == 3


@pytest.mark.asyncio()
async def test_llm_retry_does_not_retry_non_matching_errors(mock: MagicMock) -> None:
    retry_middleware = RetryMiddleware(max_retries=3, retry_on=(TransientError,))
    middleware = retry_middleware(TextInput("Hi!"), mock)

    async def llm_call(events: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        mock.llm_call(events)
        raise PermanentError("do not retry")

    with pytest.raises(PermanentError, match="do not retry"):
        await middleware.on_llm_call(llm_call, [TextInput("Hi!")], mock)

    mock.llm_call.assert_called_once_with([TextInput("Hi!")])
