# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import asyncio
import functools
import time
from collections.abc import Callable
from json.decoder import JSONDecodeError
from typing import Any, TypeVar

import pytest

from autogen.fast_depends.utils import is_coroutine_callable
from autogen.import_utils import optional_import_block

T = TypeVar("T", bound=Callable[..., Any])


def suppress(
    exception: type[BaseException],
    *,
    retries: int = 0,
    timeout: int = 60,
    error_filter: Callable[[BaseException], bool] | None = None,
) -> Callable[[T], T]:
    """Suppresses the specified exception and retries the function a specified number of times.

    Args:
        exception: The exception to suppress.
        retries: The number of times to retry the function. If None, the function will tried once and just return in case of exception raised. Defaults to None.
        timeout: The time to wait between retries in seconds. Defaults to 60.
        error_filter: A function that takes an exception as input and returns a boolean indicating whether the exception should be suppressed. Defaults to None.
    """

    def decorator(
        func: T,
        exception: type[BaseException] = exception,
        retries: int = retries,
        timeout: int = timeout,
        error_filter: Callable[[BaseException], bool] | None = error_filter,
    ) -> T:
        if is_coroutine_callable(func):

            @functools.wraps(func)
            async def wrapper(
                *args: Any,
                exception: type[BaseException] = exception,
                retries: int = retries,
                timeout: int = timeout,
                **kwargs: Any,
            ) -> Any:
                for i in range(retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except exception as e:
                        if error_filter and not error_filter(e):  # type: ignore [arg-type]
                            raise
                        if i >= retries - 1:
                            pytest.xfail(f"Suppressed '{exception}' raised {i + 1} times")
                            raise
                        await asyncio.sleep(timeout)

        else:

            @functools.wraps(func)
            def wrapper(
                *args: Any,
                exception: type[BaseException] = exception,
                retries: int = retries,
                timeout: int = timeout,
                **kwargs: Any,
            ) -> Any:
                for i in range(retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except exception as e:
                        if error_filter and not error_filter(e):  # type: ignore [arg-type]
                            raise
                        if i >= retries - 1:
                            pytest.xfail(f"Suppressed '{exception}' raised {i + 1} times")
                            raise
                        time.sleep(timeout)

        return wrapper  # type: ignore[return-value]

    return decorator


def suppress_gemini_resource_exhausted(func: T) -> T:
    with optional_import_block():
        from google.genai.errors import ClientError

        # Catch only code 429 which is RESOURCE_EXHAUSTED error instead of catching all the client errors
        def is_resource_exhausted_error(e: BaseException) -> bool:
            return isinstance(e, ClientError) and getattr(e, "code", None) in [429, 503]

        return suppress(ClientError, retries=2, error_filter=is_resource_exhausted_error)(func)

    return func


def suppress_json_decoder_error(func: T) -> T:
    return suppress(JSONDecodeError)(func)
