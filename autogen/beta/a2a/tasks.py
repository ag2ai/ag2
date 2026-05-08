# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any

from a2a.types import (
    CancelTaskRequest,
    GetTaskRequest,
    ListTasksRequest,
    Task,
)

from ._session import open_session
from .config import A2AConfig
from .mappers.parts import struct_from_dict


async def cancel_task(
    config: A2AConfig,
    task_id: str,
    *,
    tenant: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Task:
    """Cancel a task on the remote A2A server.

    ``tenant`` overrides the tenant baked into ``config`` for this call.
    ``metadata`` is attached to the request for server-side handlers
    that need extra context (e.g. operator id, reason).
    """
    async with open_session(config) as sdk:
        request_kwargs = _with_tenant(config, tenant, id=task_id)
        if metadata:
            request_kwargs["metadata"] = struct_from_dict(dict(metadata))
        return await sdk.cancel_task(CancelTaskRequest(**request_kwargs))


async def get_task(
    config: A2AConfig,
    task_id: str,
    *,
    tenant: str | None = None,
    history_length: int | None = None,
) -> Task:
    """Fetch a task by id.

    ``history_length`` requests the server to truncate ``task.history``
    to the most recent N messages — useful for status dashboards that
    don't need the full conversation.
    """
    async with open_session(config) as sdk:
        kwargs = _with_tenant(config, tenant, id=task_id)
        resolved_history = history_length if history_length is not None else config.history_length
        if resolved_history is not None:
            kwargs["history_length"] = resolved_history
        return await sdk.get_task(GetTaskRequest(**kwargs))


async def list_tasks(
    config: A2AConfig,
    *,
    tenant: str | None = None,
    context_id: str | None = None,
    status: str | None = None,
    page_size: int | None = None,
    page_token: str | None = None,
    history_length: int | None = None,
    include_artifacts: bool = False,
) -> list[Task]:
    """List tasks on the remote A2A server.

    Pagination is handled by the caller — the server-issued
    ``next_page_token`` is *not* surfaced through this helper. Pass it
    back via ``page_token`` to fetch the next page.
    """
    async with open_session(config) as sdk:
        kwargs = _with_tenant(config, tenant)
        if context_id:
            kwargs["context_id"] = context_id
        if status:
            kwargs["status"] = status
        if page_size is not None:
            kwargs["page_size"] = page_size
        if page_token:
            kwargs["page_token"] = page_token
        resolved_history = history_length if history_length is not None else config.history_length
        if resolved_history is not None:
            kwargs["history_length"] = resolved_history
        if include_artifacts:
            kwargs["include_artifacts"] = True
        response = await sdk.list_tasks(ListTasksRequest(**kwargs))
        return list(response.tasks)


def _with_tenant(config: A2AConfig, override: str | None, **kwargs: Any) -> dict[str, Any]:
    tenant = override if override is not None else config.tenant
    if tenant:
        kwargs["tenant"] = tenant
    return kwargs


__all__ = ("cancel_task", "get_task", "list_tasks")
