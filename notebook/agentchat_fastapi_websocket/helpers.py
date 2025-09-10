import asyncio


def async_to_sync(awaitable):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(awaitable)
