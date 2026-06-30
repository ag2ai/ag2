# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any

from fast_depends.utils import is_coroutine_callable, run_in_threadpool


async def call_user_fn(fn: Callable[..., Any], *args: Any) -> Any:
    """Invoke a user-supplied resource/prompt callable, awaiting if it is async.

    Async callables are awaited directly; sync callables run in a worker thread
    (they may do blocking file/network I/O) so the event loop is never blocked.
    """
    if is_coroutine_callable(fn):
        return await fn(*args)
    return await run_in_threadpool(fn, *args)
