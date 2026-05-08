# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any

from a2a.types import (
    AuthenticationInfo,
    DeleteTaskPushNotificationConfigRequest,
    GetTaskPushNotificationConfigRequest,
    ListTaskPushNotificationConfigsRequest,
    TaskPushNotificationConfig,
)

from ._session import open_session
from .config import A2AConfig


@dataclass(slots=True, kw_only=True)
class A2APushAuthentication:
    """Authentication metadata attached to a push-notification webhook.

    ``scheme`` is the auth scheme name (e.g. ``"bearer"``, ``"basic"``);
    ``credentials`` is the opaque secret the server presents to the
    receiving webhook. Server-side handlers verify these values when
    incoming push deliveries arrive.
    """

    scheme: str
    credentials: str | None = None


@dataclass(slots=True, kw_only=True)
class A2APushConfig:
    """Push-notification subscription record.

    ``id`` is the server-issued config id (populated on responses; can
    be left unset on create requests where the server picks the id).
    """

    url: str
    token: str | None = None
    authentication: A2APushAuthentication | None = None
    id: str | None = None


async def create_push_notification_config(
    config: A2AConfig,
    task_id: str,
    push_config: A2APushConfig,
    *,
    tenant: str | None = None,
) -> A2APushConfig:
    """Register a push-notification webhook for a task."""
    async with open_session(config) as sdk:
        request = _to_proto(config, tenant, task_id=task_id, push=push_config)
        response = await sdk.create_task_push_notification_config(request)
        return _from_proto(response)


async def get_push_notification_config(
    config: A2AConfig,
    task_id: str,
    config_id: str,
    *,
    tenant: str | None = None,
) -> A2APushConfig:
    """Fetch a previously-registered push config by id."""
    async with open_session(config) as sdk:
        kwargs: dict[str, Any] = {"task_id": task_id, "id": config_id}
        resolved_tenant = tenant if tenant is not None else config.tenant
        if resolved_tenant:
            kwargs["tenant"] = resolved_tenant
        response = await sdk.get_task_push_notification_config(
            GetTaskPushNotificationConfigRequest(**kwargs),
        )
        return _from_proto(response)


async def list_push_notification_configs(
    config: A2AConfig,
    task_id: str,
    *,
    tenant: str | None = None,
    page_size: int | None = None,
    page_token: str | None = None,
) -> list[A2APushConfig]:
    """List push-notification configs registered against ``task_id``.

    Pagination follows the SDK pattern: pass ``page_token`` from the
    previous response to fetch the next page. The token itself is *not*
    surfaced through this helper — callers tracking pagination should
    use the SDK directly.
    """
    async with open_session(config) as sdk:
        kwargs: dict[str, Any] = {"task_id": task_id}
        resolved_tenant = tenant if tenant is not None else config.tenant
        if resolved_tenant:
            kwargs["tenant"] = resolved_tenant
        if page_size is not None:
            kwargs["page_size"] = page_size
        if page_token:
            kwargs["page_token"] = page_token
        response = await sdk.list_task_push_notification_configs(
            ListTaskPushNotificationConfigsRequest(**kwargs),
        )
        return [_from_proto(cfg) for cfg in response.configs]


async def delete_push_notification_config(
    config: A2AConfig,
    task_id: str,
    config_id: str,
    *,
    tenant: str | None = None,
) -> None:
    """Delete a registered push-notification config."""
    async with open_session(config) as sdk:
        kwargs: dict[str, Any] = {"task_id": task_id, "id": config_id}
        resolved_tenant = tenant if tenant is not None else config.tenant
        if resolved_tenant:
            kwargs["tenant"] = resolved_tenant
        await sdk.delete_task_push_notification_config(
            DeleteTaskPushNotificationConfigRequest(**kwargs),
        )


def _to_proto(
    config: A2AConfig,
    tenant_override: str | None,
    *,
    task_id: str,
    push: A2APushConfig,
) -> TaskPushNotificationConfig:
    kwargs: dict[str, Any] = {
        "task_id": task_id,
        "url": push.url,
    }
    if push.id:
        kwargs["id"] = push.id
    if push.token:
        kwargs["token"] = push.token
    if push.authentication is not None:
        kwargs["authentication"] = AuthenticationInfo(
            scheme=push.authentication.scheme,
            credentials=push.authentication.credentials or "",
        )
    resolved_tenant = tenant_override if tenant_override is not None else config.tenant
    if resolved_tenant:
        kwargs["tenant"] = resolved_tenant
    return TaskPushNotificationConfig(**kwargs)


def _from_proto(proto: TaskPushNotificationConfig) -> A2APushConfig:
    auth: A2APushAuthentication | None = None
    if proto.HasField("authentication"):
        auth = A2APushAuthentication(
            scheme=proto.authentication.scheme,
            credentials=proto.authentication.credentials or None,
        )
    return A2APushConfig(
        url=proto.url,
        token=proto.token or None,
        authentication=auth,
        id=proto.id or None,
    )


__all__ = (
    "A2APushAuthentication",
    "A2APushConfig",
    "create_push_notification_config",
    "delete_push_notification_config",
    "get_push_notification_config",
    "list_push_notification_configs",
)
