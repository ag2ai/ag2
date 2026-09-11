# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any

from fast_depends.utils import is_coroutine_callable, run_in_threadpool


async def call_user_fn(fn: Callable[..., Any], *args: Any) -> Any:
    """Invoke a user-supplied resource/prompt callable, awaiting if it is async.

    Sync callables run in a worker thread, so blocking I/O never blocks the loop.
    """
    if is_coroutine_callable(fn):
        return await fn(*args)
    return await run_in_threadpool(fn, *args)
