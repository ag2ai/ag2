# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from a2a.types import (
    AuthenticationInfo,
    DeleteTaskPushNotificationConfigRequest,
    GetTaskPushNotificationConfigRequest,
    ListTaskPushNotificationConfigsRequest,
    TaskPushNotificationConfig,
)

from ._session import open_session, with_tenant
from .config import A2AConfig


@dataclass(slots=True, kw_only=True)
class A2APushAuthentication:
    """Auth metadata attached to a push-notification webhook (scheme + opaque credentials)."""

    scheme: str
    credentials: str | None = None


@dataclass(slots=True, kw_only=True)
class A2APushConfig:
    """Push-notification subscription record; ``id`` is server-issued on create."""

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
        response = await sdk.create_task_push_notification_config(
            _to_proto(config, tenant, task_id=task_id, push=push_config),
        )
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
        kwargs = with_tenant(config, tenant, task_id=task_id, id=config_id)
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
    """List push-notification configs for ``task_id``; caller passes ``page_token`` for next page."""
    async with open_session(config) as sdk:
        kwargs = with_tenant(config, tenant, task_id=task_id)
        optional = {"page_size": page_size, "page_token": page_token}
        kwargs.update({k: v for k, v in optional.items() if v is not None})
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
        kwargs = with_tenant(config, tenant, task_id=task_id, id=config_id)
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
    kwargs = with_tenant(config, tenant_override, task_id=task_id, url=push.url)
    if push.id:
        kwargs["id"] = push.id
    if push.token:
        kwargs["token"] = push.token
    if push.authentication is not None:
        kwargs["authentication"] = AuthenticationInfo(
            scheme=push.authentication.scheme,
            credentials=push.authentication.credentials or "",
        )
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
