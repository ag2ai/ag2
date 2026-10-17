# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Cohesivity backend infrastructure extension for AG2.

Provides a toolkit that gives agents on-demand backend resources through
the Cohesivity API: databases, hosting, email, object storage, and more. No
account or API keys are required to start. An agent can create an ephemeral
tenant, provision what it needs, and begin using resources immediately.

Maintainer: AragAgrawal
Docs: https://docs.ag2.ai/docs/user-guide/extensions/tools/cohesivity/
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Annotated, Any

import httpx
from pydantic import Field

from ag2.annotations import Context
from ag2.events import ToolResult
from ag2.middleware import ToolMiddleware
from ag2.tools.final import Toolkit, tool
from ag2.tools.final.function_tool import FunctionTool

_DEFAULT_BASE_URL = "https://cohesivity.ai"
_DEFAULT_ATTRIBUTION = "gh-ag2"


def _user_agent(attribution: str) -> str:
    """Build the User-Agent in the installer's format: ``{npx:ag2.<attribution>}``."""
    return "{npx:ag2." + attribution + "}" if attribution else "{npx:ag2}"


def _parse_dotfile(text: str) -> dict[str, str]:
    """Parse a ``.cohesivity`` key=value response into a dict."""
    result: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        eq = line.find("=")
        if eq > 0:
            result[line[:eq].strip()] = line[eq + 1 :].strip()
    return result


@dataclass(slots=True)
class CohesivityTenant:
    """Result of creating an ephemeral tenant."""

    tenant_id: str
    management_key: str
    application_key: str
    base_url: str
    expires_at: str | None = None
    runtime_profile: str | None = None


@dataclass(slots=True)
class CohesivityProvisionResult:
    """Result of provisioning a resource."""

    resource: str
    success: bool
    edge_url: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CohesivityStatusResult:
    """Tenant status and provisioned resources."""

    tenant_id: str
    status: str
    resources: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


class CohesivityToolkit(Toolkit):
    """Toolkit that gives agents on-demand backend infrastructure.

    Three tools let an agent create a backend, provision resources, and check
    their status, all without requiring pre-existing accounts or API keys:

      - ``cohesivity_create_tenant``: create an ephemeral 72-hour tenant and
        receive management and application keys
      - ``cohesivity_provision``: provision one or more resources (postgres,
        redis, hosting, inbox, object-storage, and others)
      - ``cohesivity_status``: check what is provisioned and the tenant
        lifecycle

    Pass a ``management_key`` to skip tenant creation when working with an
    existing tenant. When omitted, the toolkit reads
    ``COHESIVITY_MANAGEMENT_KEY`` from the environment. When neither is set,
    the agent can call ``cohesivity_create_tenant`` to obtain one.

    After provisioning, the agent uses each resource's edge URL directly
    (e.g. ``POST /edge/postgres`` for SQL queries). See the
    `Cohesivity docs <https://cohesivity.ai/docs>`_ for the full edge API
    reference.

    Example::

        from ag2 import Agent
        from ag2.config import AnthropicConfig
        from ag2.extensions.tools import CohesivityToolkit

        agent = Agent(
            "backend-agent",
            config=AnthropicConfig(model="claude-sonnet-5"),
            tools=[CohesivityToolkit()],
        )
    """

    __slots__ = ("_attribution", "_base_url", "_management_key", "_timeout")

    def __init__(
        self,
        management_key: str | None = None,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        attribution: str = _DEFAULT_ATTRIBUTION,
        timeout: float = 30.0,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        self._management_key = management_key or os.environ.get("COHESIVITY_MANAGEMENT_KEY")
        self._base_url = base_url.rstrip("/")
        self._attribution = attribution
        self._timeout = timeout

        super().__init__(
            self.create_tenant(),
            self.provision(),
            self.status(),
            name="cohesivity_toolkit",
            middleware=middleware,
        )

    def _require_key(self) -> str:
        key = self._management_key
        if not key:
            raise RuntimeError(
                "No management key available. Call cohesivity_create_tenant first, "
                "or pass management_key to CohesivityToolkit()."
            )
        return key

    def _ua(self) -> str:
        return _user_agent(self._attribution)

    def create_tenant(
        self,
        *,
        name: str = "cohesivity_create_tenant",
        description: str = (
            "Create an ephemeral Cohesivity tenant with a 72-hour lifetime. "
            "Returns management and application keys for provisioning and "
            "using resources. No account or API keys required."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Create the ``cohesivity_create_tenant`` tool."""
        toolkit = self

        @tool(name=name, description=description, middleware=middleware)
        async def cohesivity_create_tenant(
            ctx: Context,
        ) -> ToolResult:
            """Create an ephemeral Cohesivity tenant."""
            async with httpx.AsyncClient(
                base_url=toolkit._base_url,
                headers={"User-Agent": toolkit._ua()},
                timeout=toolkit._timeout,
            ) as client:
                response = await client.post("/api/genesis")
                response.raise_for_status()

            parsed = _parse_dotfile(response.text)
            management_key = parsed.get("coh_management_key", "")
            toolkit._management_key = management_key

            return ToolResult(
                CohesivityTenant(
                    tenant_id=parsed.get("tenant_id", ""),
                    management_key=management_key,
                    application_key=parsed.get("coh_application_key", ""),
                    base_url=toolkit._base_url,
                    expires_at=parsed.get("expires_at"),
                    runtime_profile=parsed.get("runtime_profile"),
                )
            )

        return cohesivity_create_tenant

    def provision(
        self,
        *,
        name: str = "cohesivity_provision",
        description: str = (
            "Provision a backend resource on Cohesivity. Available resources "
            "include: postgres, redis, object-storage, vector-database, inbox, "
            "railway-hosting, cloudflare-workers, realtime, social-login, "
            "openai-api, ai-gateway, deepgram-api, exa-api, steel-browser, "
            "openweather-api, google-geocoding-api."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Create the ``cohesivity_provision`` tool."""
        toolkit = self

        @tool(name=name, description=description, middleware=middleware)
        async def cohesivity_provision(
            resource: Annotated[
                str,
                Field(description='The resource to provision, e.g. "postgres", "redis", "railway-hosting".'),
            ],
            ctx: Context,
        ) -> ToolResult:
            """Provision a backend resource."""
            async with httpx.AsyncClient(
                base_url=toolkit._base_url,
                headers={
                    "Authorization": f"Bearer {toolkit._require_key()}",
                    "User-Agent": toolkit._ua(),
                },
                timeout=toolkit._timeout,
            ) as client:
                response = await client.post(f"/api/resources/{resource}")
                response.raise_for_status()

            raw: dict[str, Any] = response.json()
            return ToolResult(
                CohesivityProvisionResult(
                    resource=resource,
                    success=raw.get("success", True),
                    edge_url=raw.get("edge_url"),
                    details={k: v for k, v in raw.items() if k not in ("success", "edge_url")},
                )
            )

        return cohesivity_provision

    def status(
        self,
        *,
        name: str = "cohesivity_status",
        description: str = (
            "Check the status of a Cohesivity tenant, including provisioned resources, lifecycle, and remaining quota."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Create the ``cohesivity_status`` tool."""
        toolkit = self

        @tool(name=name, description=description, middleware=middleware)
        async def cohesivity_status(
            ctx: Context,
        ) -> ToolResult:
            """Check tenant status and provisioned resources."""
            async with httpx.AsyncClient(
                base_url=toolkit._base_url,
                headers={
                    "Authorization": f"Bearer {toolkit._require_key()}",
                    "User-Agent": toolkit._ua(),
                },
                timeout=toolkit._timeout,
            ) as client:
                response = await client.get("/api/status")
                response.raise_for_status()

            raw: dict[str, Any] = response.json()
            return ToolResult(
                CohesivityStatusResult(
                    tenant_id=raw.get("tenant_id", ""),
                    status=raw.get("status", ""),
                    resources=raw.get("resources", {}),
                    details={k: v for k, v in raw.items() if k not in ("tenant_id", "status", "resources")},
                )
            )

        return cohesivity_status
