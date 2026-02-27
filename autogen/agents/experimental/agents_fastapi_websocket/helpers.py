import asyncio
from collections.abc import Awaitable
from typing import Any


def async_to_sync(awaitable: Awaitable[Any]) -> Any:
    """
    Convert an awaitable to a synchronous function call.

    Args:
        awaitable: An awaitable object to execute synchronously

    Returns:
        The result of the awaitable
    """
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(awaitable)
