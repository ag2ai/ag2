# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from unittest.mock import MagicMock

import pytest

from autogen.beta import Context
from autogen.beta.events import BaseEvent, ModelMessage, ModelRequest, ModelResponse
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
        return ModelResponse(message=ModelMessage(content="result"))

    middleware = retry_middleware(ModelRequest(content="Hi!"), mock)
    response = await middleware.on_llm_call(llm_call, [ModelRequest(content="Hi!")], mock)

    assert response == ModelResponse(message=ModelMessage(content="result"))
    mock.llm_call.assert_called_once_with([ModelRequest(content="Hi!")])


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
        return ModelResponse(message=ModelMessage(content="result"))

    middleware = retry_middleware(ModelRequest(content="Hi!"), mock)
    response = await middleware.on_llm_call(llm_call, [ModelRequest(content="Hi!")], mock)

    assert response == ModelResponse(message=ModelMessage(content="result"))
    assert mock.llm_call.call_count == attempts == 3


@pytest.mark.asyncio()
async def test_llm_retry_raises_after_exhausting_retries(mock: MagicMock) -> None:
    retry_middleware = RetryMiddleware(max_retries=2, retry_on=(TransientError,))

    async def llm_call(events: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        mock.llm_call(events)
        raise TransientError("still failing")

    middleware = retry_middleware(ModelRequest(content="Hi!"), mock)
    with pytest.raises(TransientError, match="still failing"):
        await middleware.on_llm_call(llm_call, [ModelRequest(content="Hi!")], mock)

    assert mock.llm_call.call_count == 3


@pytest.mark.asyncio()
async def test_llm_retry_does_not_retry_non_matching_errors(mock: MagicMock) -> None:
    retry_middleware = RetryMiddleware(max_retries=3, retry_on=(TransientError,))
    middleware = retry_middleware(ModelRequest(content="Hi!"), mock)

    async def llm_call(events: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        mock.llm_call(events)
        raise PermanentError("do not retry")

    with pytest.raises(PermanentError, match="do not retry"):
        await middleware.on_llm_call(llm_call, [ModelRequest(content="Hi!")], mock)

    mock.llm_call.assert_called_once_with([ModelRequest(content="Hi!")])
