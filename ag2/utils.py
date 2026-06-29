# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from functools import wraps
from typing import Any

from fast_depends import dependency_provider
from fast_depends.core import CallModel, build_call_model
from fast_depends.pydantic import PydanticSerializer
from fast_depends.utils import is_coroutine_callable, run_in_threadpool

CONTEXT_OPTION_NAME = "__ctx__"


def build_model(
    f: Callable[..., Any],
    *,
    sync_to_thread: bool = True,
    serialize_result: bool = True,
) -> CallModel:
    return build_call_model(
        _to_async(f, sync_to_thread=sync_to_thread),
        dependency_provider=dependency_provider,
        serializer_cls=PydanticSerializer(
            pydantic_config={"arbitrary_types_allowed": True},
            use_fastdepends_errors=False,
        ),
        serialize_result=serialize_result,
    )


def _to_async(
    func: Callable[..., Any],
    *,
    sync_to_thread: bool = True,
) -> Callable[..., Any]:
    if is_coroutine_callable(func):
        return func

    if sync_to_thread:

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return await run_in_threadpool(func, *args, **kwargs)

    else:

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

    return async_wrapper
