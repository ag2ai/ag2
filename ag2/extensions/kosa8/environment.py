# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Configure reusable kosa8 Cloud sandboxes for AG2 tools."""

import json
import os
import threading
from collections.abc import AsyncGenerator, Hashable, Mapping
from contextlib import asynccontextmanager
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

import httpx

from ag2.annotations import Variable
from ag2.tools.builtin._resolve import resolve_variable
from ag2.tools.code import CodeLanguage
from ag2.tools.sandbox import CodeAdapter, LanguageRunner

from .sandbox import DEFAULT_WORKDIR, Kosa8Sandbox, check_env_names

if TYPE_CHECKING:
    from ag2.context import ConversationContext

DEFAULT_BASE_URL = "https://api.kosa8.com"

#: kosa8 Cloud takes a ttl_s from 60 seconds to 90 days.
MIN_TTL_S = 60
MAX_TTL_S = 7_776_000


def make_client(api_key: str | None, base_url: str | None) -> httpx.AsyncClient:
    """Build the API client, reading ``KOSA8_API_KEY`` and ``KOSA8_API_URL``
    for what is not given.

    Raises:
        ValueError: If there is no API key, or the address is not HTTPS (plain
            HTTP is accepted for localhost only, since the key travels in every
            request).
    """
    key = (api_key if api_key is not None else os.environ.get("KOSA8_API_KEY", "")).strip()
    if not key:
        raise ValueError("No kosa8 API key: pass `api_key` or set KOSA8_API_KEY.")
    url = httpx.URL((base_url or os.environ.get("KOSA8_API_URL") or DEFAULT_BASE_URL).rstrip("/"))
    local = url.host in ("localhost", "127.0.0.1", "::1")
    if url.scheme != "https" and not (url.scheme == "http" and local):
        raise ValueError(f"The kosa8 API address must use https: {url}")
    return httpx.AsyncClient(
        base_url=url,
        headers={
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "User-Agent": "ag2-kosa8",
        },
        timeout=httpx.Timeout(60.0),
    )


class Kosa8Environment:
    """:class:`SandboxFactory` for :class:`Kosa8Sandbox`.

    kosa8 credentials and sandbox parameters may be deferred with
    :class:`~ag2.annotations.Variable`. Sandboxes are cached by their resolved
    parameters and remain available until :meth:`aclose`, preserving files
    across tool calls.

    ``KOSA8_API_KEY`` and ``KOSA8_API_URL`` are read when their constructor
    arguments are omitted.
    """

    def __init__(
        self,
        *,
        api_key: "str | Variable | None" = None,  # pragma: allowlist secret
        base_url: "str | Variable | None" = None,
        template: "str | Variable" = "python",
        region: "str | Variable | None" = None,
        name: "str | Variable | None" = None,
        env_vars: "dict[str, str] | Variable | None" = None,
        labels: "dict[str, str] | Variable | None" = None,
        egress: Mapping[str, Any] | None = None,
        persistent: bool = False,
        timeout: float = 60,
        start_timeout: float = 300,
        ttl_s: int | None = 3600,
        workdir: str = DEFAULT_WORKDIR,
    ) -> None:
        """Configure the sandboxes this factory hands out.

        Every ``Variable``-accepting argument is resolved per :meth:`open`
        call, so one factory can serve several tenants.

        Args:
            api_key: kosa8 API key (``k8c_live_...``). Falls back to
                ``KOSA8_API_KEY``.
            base_url: kosa8 API endpoint. Falls back to ``KOSA8_API_URL``,
                then to ``https://api.kosa8.com``. Point it at your own
                deployment to run sandboxes on your own hosts.
            template: Template key, such as ``"python"`` (Python 3.12 on
                Alpine, the default), ``"node"`` or ``"base"`` (Alpine,
                busybox ``sh`` only). ``GET /v1/templates`` lists them.
            region: Region to create sandboxes in. ``None`` lets kosa8 pick.
            name: Sandbox name, for finding it in the dashboard.
            env_vars: Environment variables set on every command the sandbox
                runs. Per-call variables take precedence.
            labels: Labels added to the sandbox, on top of ``created-by=ag2``,
                which every sandbox this extension creates carries so it can
                be listed. A ``created-by`` key here is overridden.
            egress: Network policy applied before the first command, in the
                API's shape: ``{"internet": False, "allow": ["pypi.org"]}``.
                ``None`` keeps the organization's default.
            persistent: Run every command in one long-lived container, so
                packages installed outside the workdir and background
                processes last between commands. Plan-limited. By default
                each command gets a fresh container and only the workdir
                carries over.
            timeout: Default per-command timeout in seconds. Up to 90 s a
                command is buffered; longer timeouts are streamed. Must be > 0.
            start_timeout: How long to wait for a sandbox that boots cold.
            ttl_s: Seconds after which kosa8 deletes the sandbox on its own,
                from 60 to 90 days. It is the backstop that reclaims the
                sandbox if client cleanup never runs. ``None`` disables it.
            workdir: Sandbox-side directory commands and relative paths
                resolve against. Only ``/workspace`` (the default) and paths
                under it outlive a command when ``persistent`` is off.

        Raises:
            ValueError: If ``timeout`` is not positive or ``ttl_s`` is out of range.
        """
        if timeout <= 0:
            raise ValueError("`timeout` must be greater than 0 seconds.")
        if start_timeout <= 0:
            raise ValueError("`start_timeout` must be greater than 0 seconds.")
        if ttl_s is not None and not MIN_TTL_S <= ttl_s <= MAX_TTL_S:
            raise ValueError(f"`ttl_s` must be from {MIN_TTL_S} to {MAX_TTL_S} seconds, or None to disable it.")
        if not PurePosixPath(workdir).is_absolute():
            raise ValueError(f"`workdir` must be an absolute path: {workdir}")

        self._api_key = api_key
        self._base_url = base_url
        self._template = template
        self._region = region
        self._name = name
        self._env_vars = env_vars
        self._labels = labels
        self._egress = dict(egress) if egress is not None else None
        self._persistent = persistent
        self._timeout = timeout
        self._start_timeout = start_timeout
        self._ttl_s = ttl_s
        self._workdir = workdir

        self._cache: dict[Hashable, Kosa8Sandbox] = {}
        # Sandboxes an `aclose` failed to delete, retried by the next `aclose`.
        self._undeleted: list[Kosa8Sandbox] = []
        self._cache_lock = threading.Lock()

    @property
    def workdir(self) -> PurePosixPath:
        """Sandbox-side working directory commands and file writes resolve against."""
        return PurePosixPath(self._workdir)

    def code_environment(self, *, languages: tuple[CodeLanguage, ...] = ("python", "bash")) -> CodeAdapter:
        """A :class:`~ag2.tools.sandbox.CodeAdapter` wired for kosa8's Alpine templates.

        The ``python``, ``node`` and ``base`` templates are Alpine images with
        busybox ``sh`` and no ``bash``, so ``bash`` snippets run with ``sh -c``.
        Pass the result straight to :class:`~ag2.tools.SandboxCodeTool`::

            env = Kosa8Environment()
            tool = SandboxCodeTool(env.code_environment())
        """
        return CodeAdapter(
            self,
            languages=languages,
            runners={"bash": LanguageRunner(inline_argv=("sh", "-c"))},
        )

    @asynccontextmanager
    async def open(
        self,
        context: "ConversationContext | None" = None,
    ) -> AsyncGenerator[Kosa8Sandbox]:
        """Yield the sandbox matching the resolved parameters.

        Sandboxes are cached by those parameters, so repeated calls reuse one
        sandbox and files survive across tool calls. The factory keeps
        ownership: leaving this scope does not delete the sandbox, only
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
            ValueError: If no API key is given or set in ``KOSA8_API_KEY``.
        """
        api_key = resolve_variable(self._api_key, context, param_name="api_key") if context else self._api_key
        base_url = resolve_variable(self._base_url, context, param_name="base_url") if context else self._base_url
        template = resolve_variable(self._template, context, param_name="template") if context else self._template
        region = resolve_variable(self._region, context, param_name="region") if context else self._region
        name = resolve_variable(self._name, context, param_name="name") if context else self._name
        env_vars = (
            resolve_variable(self._env_vars, context, param_name="env_vars") if context else self._env_vars
        ) or {}
        labels = (resolve_variable(self._labels, context, param_name="labels") if context else self._labels) or {}

        unresolved = (api_key, base_url, template, region, name, env_vars, labels)
        if any(isinstance(value, Variable) for value in unresolved):
            raise RuntimeError(
                "kosa8 parameters given as Variable but no Context is available to resolve them. "
                "Variables are only resolvable when a sandbox tool is invoked through an Agent."
            )
        assert api_key is None or isinstance(api_key, str)
        assert base_url is None or isinstance(base_url, str)
        assert isinstance(template, str)
        assert region is None or isinstance(region, str)
        assert name is None or isinstance(name, str)
        assert isinstance(env_vars, dict)
        assert isinstance(labels, dict)

        # Checked before a client exists, so a bad name leaves nothing to release.
        check_env_names(env_vars)

        all_labels = {**labels, "created-by": "ag2"}
        create_options: dict[str, Any] = {"template": template}
        if region is not None:
            create_options["region"] = region
        if name is not None:
            create_options["name"] = name
        if self._persistent:
            create_options["persistent"] = True
        if self._ttl_s is not None:
            create_options["ttl_s"] = self._ttl_s

        key: Hashable = (
            api_key,
            base_url,
            tuple(sorted(create_options.items())),
            tuple(sorted(env_vars.items())),
            tuple(sorted(all_labels.items())),
            json.dumps(self._egress, sort_keys=True),
            self._timeout,
            self._start_timeout,
            self._workdir,
        )

        with self._cache_lock:
            sandbox = self._cache.get(key)
            if sandbox is None or sandbox.closed:
                sandbox = Kosa8Sandbox(
                    client=make_client(api_key, base_url),
                    create_options=create_options,
                    labels=all_labels,
                    egress=self._egress,
                    timeout=self._timeout,
                    start_timeout=self._start_timeout,
                    workdir=self._workdir,
                    env=env_vars,
                )
                self._cache[key] = sandbox

        # A failed setup is not closed here: the entry is shared, and closing it
        # would delete a sandbox another caller is still creating or using. The
        # sandbox deletes its own failed setup, and retries a failed delete
        # before it creates a replacement.
        await sandbox.__aenter__()
        # The factory owns the lifecycle so cached state survives this scope.
        yield sandbox

    async def aclose(self) -> None:
        """Delete every cached sandbox. Safe to call multiple times; a call
        retries the sandboxes a previous one failed to delete."""
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
                self._undeleted.extend(sandbox for sandbox in sandboxes if sandbox.sandbox_id is not None)

    async def __aenter__(self) -> "Kosa8Environment":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def __deepcopy__(self, memo: dict[int, Any]) -> "Kosa8Environment":
        return self
