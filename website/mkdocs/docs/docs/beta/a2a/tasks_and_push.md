---
title: Managing Tasks and Push Notifications
sidebarTitle: Tasks & Push
---

A2A models every conversation as a `#!python Task` with a server-side lifecycle. `#!python autogen.beta.a2a.tasks` exposes helpers for inspecting and aborting tasks; `#!python autogen.beta.a2a.push` manages webhook subscriptions for asynchronous delivery of task updates. All helpers accept an `#!python A2AConfig` and work over any transport the server's card declares.

## Tasks

```python linenums="1"
from autogen.beta.a2a import A2AConfig
from autogen.beta.a2a.tasks import cancel_task, get_task, list_tasks

config = A2AConfig(card_url="http://127.0.0.1:8000")
```

### Listing

`#!python list_tasks` returns a `#!python ListedTasks` dataclass — the page's `#!python tasks` plus the server-reported pagination metadata (`#!python next_page_token`, `#!python page_size`, `#!python total_size`). Pagination is handled by the caller — pass `#!python page_token` from a prior response back in to advance.

```python linenums="1"
from a2a.types import TaskState

page = await list_tasks(config, page_size=10)
for task in page.tasks:
    print(task.id, task.status.state)
if page.next_page_token:
    next_page = await list_tasks(config, page_size=10, page_token=page.next_page_token)

running = await list_tasks(config, status=TaskState.TASK_STATE_WORKING)
```

| Argument | Purpose |
|---|---|
| `#!python tenant` | Scope the call to a specific tenant on a shared backend |
| `#!python context_id` | Filter to a single conversation context |
| `#!python status` | Filter by `#!python TaskState` enum value (e.g. `#!python TaskState.TASK_STATE_WORKING`) |
| `#!python page_size` / `#!python page_token` | Caller-driven pagination |
| `#!python history_length` | Server-side hint to truncate echoed `#!python Task.history` |
| `#!python include_artifacts` | Include task artifacts in the response (default: `#!python False`) |
| `#!python status_timestamp_after` | Filter to tasks whose status timestamp is after this `#!python datetime` |

### Fetching a Single Task

```python linenums="1"
full = await get_task(config, task_id)
truncated = await get_task(config, task_id, history_length=1)
```

`#!python history_length` is a hint — some server versions ignore it and return the full history. Don't rely on it for security boundaries.

### Cancelling

```python linenums="1"
cancelled = await cancel_task(
    config,
    task_id,
    metadata={"reason": "operator override"},
)
```

`#!python metadata` is forwarded to the server's cancel handler — useful for auditing who cancelled what and why. Already-terminal tasks (`#!python COMPLETED`, `#!python FAILED`, etc.) are typically refused with a `#!python 409`-style error; surface the exception, don't paper over it.

## Push Notifications

A2A push notifications let the server POST task updates to a webhook the client registered ahead of time — useful when the client doesn't want to hold an open SSE stream for the whole task lifetime.

!!! note
    The server must be constructed with a `#!python push_config_store` for push CRUD to work. The default `#!python A2AServer(agent)` does not enable push. See [Server → What `A2AServer` Holds](server.md#what-a2aserver-holds).

### Registering a Webhook

```python linenums="1"
from autogen.beta.a2a.push import (
    A2APushAuthentication,
    A2APushConfig,
    create_push_notification_config,
)

push = A2APushConfig(
    url="https://hooks.example.com/a2a",
    token="webhook-token",
    authentication=A2APushAuthentication(scheme="bearer", credentials="abc..."),
)
created = await create_push_notification_config(config, task_id, push)
print(created.id)  # server-issued config id
```

`#!python A2APushAuthentication` round-trips `#!python scheme` / `#!python credentials` through the wire format without loss. Receiving handlers verify these against the inbound request.

### Reading and Deleting

```python linenums="1"
from autogen.beta.a2a.push import (
    delete_push_notification_config,
    get_push_notification_config,
    list_push_notification_configs,
)

fetched = await get_push_notification_config(config, task_id, created.id)
listed = await list_push_notification_configs(config, task_id, page_size=10)
await delete_push_notification_config(config, task_id, created.id)
```

| Helper | Purpose |
|---|---|
| `#!python create_push_notification_config` | Register a webhook for a task |
| `#!python get_push_notification_config` | Fetch a single config by id |
| `#!python list_push_notification_configs` | List configs registered for a task (paginated) |
| `#!python delete_push_notification_config` | Remove a registered config |

## Multi-tenant Scoping

Both modules accept a per-call `#!python tenant=...` kwarg that overrides the tenant baked into `#!python A2AConfig` for that single request. Per-call overrides are also available via `#!python context.variables["a2a:tenant"]` — useful when a single client serves multiple tenants on the same shared backend.

```python linenums="1"
acme_tasks = await list_tasks(config, tenant="acme-corp", page_size=10)
```
