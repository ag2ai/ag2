import asyncio

from a2a.types import TaskState

from ag2.a2a import A2AConfig
from ag2.a2a.tasks import cancel_task, get_task, list_tasks

config = A2AConfig(card_url="http://127.0.0.1:8000", prefer="rest")


async def main() -> None:
    # list — pagination is the caller's responsibility
    page = await list_tasks(
        config,
        page_size=10,
        status=TaskState.TASK_STATE_WORKING,
    )
    print(page.next_page_token, page.total_size)
    for t in page.tasks:
        print(t.id, TaskState.Name(t.status.state))

    # get — with truncated history (config.history_length is the default)
    full = await get_task(config, page.tasks[0].id)
    truncated = await get_task(config, page.tasks[0].id, history_length=1)
    print(full.id, truncated.id)

    # cancel — with arbitrary metadata forwarded to server-side handlers
    await cancel_task(config, page.tasks[0].id, metadata={"reason": "operator override"})


if __name__ == "__main__":
    asyncio.run(main())
