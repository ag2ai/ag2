# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Configure reusable boxd machines as sandboxes for AG2 tools."""

import threading
from collections.abc import AsyncGenerator, Hashable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from boxd import AsyncBoxd

from ag2.annotations import Variable
from ag2.tools.builtin._resolve import resolve_variable

from .sandbox import BoxdSandbox, check_env_names

if TYPE_CHECKING:
    from ag2.context import ConversationContext


@dataclass(slots=True)
class BoxdResources:
    """Size of a boxd machine.

    A field left as ``None`` defers to the organization's default size. boxd
    offers fixed sizes (1 vCPU / 4G, 2 / 8G, 4 / 16G, 8 / 32G), up to the
    organization's quota.

    Attributes:
        vcpu: Virtual CPUs.
        memory: Memory in bytes, or a size string such as ``"8G"``.
        disk: Disk size in bytes, or a size string such as ``"100G"``.
    """

    vcpu: int | None = None
    memory: int | str | None = None
    disk: int | str | None = None


class BoxdEnvironment:
    """:class:`SandboxFactory` for :class:`BoxdSandbox`.

    boxd credentials and machine parameters may be deferred with
    :class:`~ag2.annotations.Variable`. Machines are cached by their resolved
    parameters and remain available until :meth:`aclose`, preserving files
    across tool calls.

    ``BOXD_API_KEY``, ``BOXD_BASE_URL`` and ``BOXD_REGION`` are read by the SDK
    when their constructor arguments are omitted.
    """

    def __init__(
        self,
        *,
        api_key: "str | Variable | None" = None,  # pragma: allowlist secret
        base_url: "str | Variable | None" = None,
        region: "str | Variable | None" = None,
        name: "str | Variable | None" = None,
        image: "str | Variable | None" = None,
        env_vars: "dict[str, str] | Variable | None" = None,
        labels: "dict[str, str] | Variable | None" = None,
        resources: BoxdResources | None = None,
        isolated: bool = True,
        timeout: float = 60,
        delete_after: int | None = 3600,
        workdir: str = "/home/boxd",
    ) -> None:
        """Configure the machines this factory hands out.

        Every ``Variable``-accepting argument is resolved per :meth:`open`
        call, so one factory can serve several tenants.

        Args:
            api_key: boxd API key. Falls back to ``BOXD_API_KEY``.
            base_url: boxd API endpoint. Falls back to ``BOXD_BASE_URL``, then
                to ``boxd.sh``.
            region: Region to create machines in, such as ``"fra"``. Falls
                back to ``BOXD_REGION``, then to the endpoint's own region.
            name: Machine name. Names are unique within an organization, so
                the default ``None`` lets boxd pick a fresh one.
            image: Guest image. ``None`` uses boxd's default Ubuntu 24.04
                image, which runs as user ``boxd`` and ships Python, Node,
                git and Docker.
            env_vars: Environment variables set on every command the sandbox
                runs. Per-call variables take precedence.
            labels: Labels added to the machine, on top of ``created-by=ag2``,
                which every machine this extension creates carries so it can
                be listed. A ``created-by`` key here is overridden.
            resources: Machine size; ``None`` uses the organization's default.
            isolated: Create the machine isolated: no in-machine boxd
                credential, metadata endpoint or private network peers, so
                code run by the agent cannot reach your boxd account. Keep it
                on for model-written code.
            timeout: Default per-command timeout in seconds, also the budget
                for waiting on a new machine to become ready. Must be > 0.
            delete_after: Seconds after which boxd deletes the machine on its
                own. It is the backstop that reclaims the machine if client
                cleanup never runs. ``None`` disables it; otherwise must be > 0.
            workdir: Machine-side directory commands and relative paths
                resolve against. Created on first use if missing.

        Raises:
            ValueError: If ``timeout`` or ``delete_after`` is not positive.
        """
        if timeout <= 0:
            raise ValueError("`timeout` must be greater than 0 seconds.")
        if delete_after is not None and delete_after <= 0:
            raise ValueError("`delete_after` must be greater than 0 seconds, or None to disable it.")

        self._api_key = api_key
        self._base_url = base_url
        self._region = region
        self._name = name
        self._image = image
        self._env_vars = env_vars
        self._labels = labels
        self._resources = resources
        self._isolated = isolated
        self._timeout = timeout
        self._delete_after = delete_after
        self._workdir = workdir

        self._cache: dict[Hashable, BoxdSandbox] = {}
        # Sandboxes an `aclose` failed to delete, retried by the next `aclose`.
        self._undeleted: list[BoxdSandbox] = []
        self._cache_lock = threading.Lock()

    @property
    def workdir(self) -> PurePosixPath:
        """Machine-side working directory commands and file writes resolve against."""
        return PurePosixPath(self._workdir)

    @asynccontextmanager
    async def open(
        self,
        context: "ConversationContext | None" = None,
    ) -> AsyncGenerator[BoxdSandbox]:
        """Yield the sandbox matching the resolved parameters.

        Sandboxes are cached by those parameters, so repeated calls reuse one
        machine and files survive across tool calls. The factory keeps
        ownership: leaving this scope does not delete the machine, only
        :meth:`aclose` does.

        Args:
            context: Conversation context used to resolve ``Variable``
                parameters. ``None`` requires every parameter to be concrete.

        Yields:
            The ready sandbox for this parameter set.

        Raises:
            RuntimeError: If a parameter is a ``Variable`` but no ``context``
                was supplied to resolve it.
            KeyError: If a ``Variable`` names a key absent from the context.
        """
        api_key = resolve_variable(self._api_key, context, param_name="api_key") if context else self._api_key
        base_url = resolve_variable(self._base_url, context, param_name="base_url") if context else self._base_url
        region = resolve_variable(self._region, context, param_name="region") if context else self._region
        name = resolve_variable(self._name, context, param_name="name") if context else self._name
        image = resolve_variable(self._image, context, param_name="image") if context else self._image
        env_vars = (
            resolve_variable(self._env_vars, context, param_name="env_vars") if context else self._env_vars
        ) or {}
        labels = (resolve_variable(self._labels, context, param_name="labels") if context else self._labels) or {}

        unresolved = (api_key, base_url, region, name, image, env_vars, labels)
        if any(isinstance(value, Variable) for value in unresolved):
            raise RuntimeError(
                "boxd parameters given as Variable but no Context is available to resolve them. "
                "Variables are only resolvable when a sandbox tool is invoked through an Agent."
            )
        assert api_key is None or isinstance(api_key, str)
        assert base_url is None or isinstance(base_url, str)
        assert region is None or isinstance(region, str)
        assert name is None or isinstance(name, str)
        assert image is None or isinstance(image, str)
        assert isinstance(env_vars, dict)
        assert isinstance(labels, dict)

        # Checked before a client exists, so a bad name leaves nothing to release.
        check_env_names(env_vars)

        resources = self._resources or BoxdResources()
        all_labels = {**labels, "created-by": "ag2"}
        key: Hashable = (
            api_key,
            base_url,
            region,
            name,
            image,
            tuple(sorted(env_vars.items())),
            tuple(sorted(all_labels.items())),
            resources.vcpu,
            resources.memory,
            resources.disk,
            self._isolated,
            self._timeout,
            self._delete_after,
            self._workdir,
        )

        with self._cache_lock:
            sandbox = self._cache.get(key)
            if sandbox is None or sandbox.closed:
                client = AsyncBoxd(api_key=api_key, base_url=base_url, region=region)
                sandbox = BoxdSandbox(
                    client=client,
                    create_options={
                        "name": name,
                        "image": image,
                        "labels": all_labels,
                        "vcpu": resources.vcpu,
                        "memory": resources.memory,
                        "disk": resources.disk,
                        "isolated": self._isolated,
                        "delete_after": self._delete_after,
                    },
                    timeout=self._timeout,
                    workdir=self._workdir,
                    env=env_vars,
                    api_key=api_key,
                )
                self._cache[key] = sandbox

        # A failed setup is not closed here: the entry is shared, and closing it
        # would delete a machine another caller is still creating or using. The
        # sandbox deletes its own failed setup, and retries a failed delete
        # before it creates a replacement.
        await sandbox.__aenter__()
        # The factory owns the lifecycle so cached state survives this scope.
        yield sandbox

    async def aclose(self) -> None:
        """Delete every cached machine. Safe to call multiple times; a call
        retries the machines a previous one failed to delete."""
        with self._cache_lock:
            sandboxes = [*self._cache.values(), *self._undeleted]
            self._cache.clear()
            self._undeleted = []
        try:
            for sandbox in sandboxes:
                await sandbox.aclose()
        finally:
            # Also on cancellation: whatever is not deleted yet stays for the next call.
            with self._cache_lock:
                self._undeleted.extend(sandbox for sandbox in sandboxes if sandbox.machine_id is not None)

    async def __aenter__(self) -> "BoxdEnvironment":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def __deepcopy__(self, memo: dict[int, Any]) -> "BoxdEnvironment":
        return self
