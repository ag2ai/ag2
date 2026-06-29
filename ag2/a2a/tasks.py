# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from a2a.types import (
    CancelTaskRequest,
    GetTaskRequest,
    ListTasksRequest,
    Task,
    TaskState,
)

from ._session import open_session, with_tenant
from .config import A2AConfig
from .mappers import struct_from_dict


@dataclass(slots=True)
class ListedTasks:
    """Result of :func:`list_tasks` — tasks plus pagination metadata.

    ``next_page_token`` is empty when there's no further page;
    ``total_size`` is the server-reported total across all pages (may be
    ``0`` if the server doesn't compute it).
    """

    tasks: list[Task] = field(default_factory=list)
    next_page_token: str = ""
    page_size: int = 0
    total_size: int = 0


def _resolve_history(config: A2AConfig, override: int | None) -> int | None:
    """Per-call override wins over ``config.history_length`` default."""
    return override if override is not None else config.history_length


async def cancel_task(
    config: A2AConfig,
    task_id: str,
    *,
    tenant: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Task:
    """Cancel a task; ``metadata`` is forwarded to server-side handlers."""
    async with open_session(config) as sdk:
        kwargs = with_tenant(config, tenant, id=task_id)
        if metadata:
            kwargs["metadata"] = struct_from_dict(dict(metadata))
        return await sdk.cancel_task(CancelTaskRequest(**kwargs))


async def get_task(
    config: A2AConfig,
    task_id: str,
    *,
    tenant: str | None = None,
    history_length: int | None = None,
) -> Task:
    """Fetch a task; ``history_length`` truncates ``task.history`` server-side."""
    async with open_session(config) as sdk:
        kwargs = with_tenant(config, tenant, id=task_id)
        history = _resolve_history(config, history_length)
        if history is not None:
            kwargs["history_length"] = history
        return await sdk.get_task(GetTaskRequest(**kwargs))


async def list_tasks(
    config: A2AConfig,
    *,
    tenant: str | None = None,
    context_id: str | None = None,
    status: TaskState | None = None,
    page_size: int | None = None,
    page_token: str | None = None,
    history_length: int | None = None,
    include_artifacts: bool = False,
    status_timestamp_after: datetime | None = None,
) -> ListedTasks:
    """List tasks; pagination is the caller's responsibility via ``next_page_token``.

    Returns a :class:`ListedTasks` carrying both the page's tasks and the
    server-reported pagination metadata. Iterate over the result or call
    ``.tasks`` for the list directly.
    """
    async with open_session(config) as sdk:
        kwargs = with_tenant(config, tenant)
        optional = {
            "context_id": context_id,
            "status": status,
            "page_size": page_size,
            "page_token": page_token,
            "history_length": _resolve_history(config, history_length),
            "status_timestamp_after": status_timestamp_after,
        }
        kwargs.update({k: v for k, v in optional.items() if v is not None})
        if include_artifacts:
            kwargs["include_artifacts"] = True
        response = await sdk.list_tasks(ListTasksRequest(**kwargs))
        return ListedTasks(
            tasks=list(response.tasks),
            next_page_token=response.next_page_token,
            page_size=response.page_size,
            total_size=response.total_size,
        )


__all__ = ("ListedTasks", "cancel_task", "get_task", "list_tasks")
