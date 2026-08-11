import asyncio

from ag2.a2a import A2AConfig
from ag2.a2a.push import (
    A2APushAuthentication,
    A2APushConfig,
    create_push_notification_config,
    delete_push_notification_config,
    get_push_notification_config,
    list_push_notification_configs,
)

config = A2AConfig(card_url="http://127.0.0.1:8000")


async def main() -> None:
    task_id = "task-abc-123"

    push = A2APushConfig(
        url="https://hooks.example.com/a2a",
        token="webhook-token",
        authentication=A2APushAuthentication(scheme="bearer", credentials="abc..."),
    )

    created = await create_push_notification_config(config, task_id, push)
    fetched = await get_push_notification_config(config, task_id, created.id)
    listed = await list_push_notification_configs(config, task_id, page_size=10)
    print(created.id, fetched.id, len(listed.configs))

    await delete_push_notification_config(config, task_id, created.id)


if __name__ == "__main__":
    asyncio.run(main())
