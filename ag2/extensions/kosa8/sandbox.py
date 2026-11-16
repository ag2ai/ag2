# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Adapt the kosa8 Cloud REST API to AG2's sandbox protocol.

kosa8 Cloud is reached over plain HTTPS with ``httpx`` (already an AG2
dependency), so this extension needs no extra package.
"""

import asyncio
import atexit
import base64
import json
import logging
import math
import re
import time
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

import httpx

from ag2.annotations import Variable
from ag2.tools.sandbox import ExecResult, SandboxBase

logger = logging.getLogger(__name__)

#: The one directory every command of a sandbox shares (and that survives sleep).
DEFAULT_WORKDIR = "/workspace"

#: kosa8 Cloud holds a buffered exec to 90 s; longer timeouts are streamed.
BUFFERED_EXEC_LIMIT_S = 90

#: The files API takes at most 8 MiB per write.
MAX_FILE_API_BYTES = 8 * 1024 * 1024

# One exec argument is at most 8192 characters: 6000 bytes are 8000 in base64.
_EXEC_WRITE_CHUNK = 6000

# Environment variable names must be plain shell identifiers.
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

# The exec API has no working-directory field, so commands enter it first.
_IN_WORKDIR = ("sh", "-c", 'cd -- "$1" || exit 1; shift; exec "$@"', "ag2")

_WRITE_FIRST = 'mkdir -p -- "$(dirname -- "$1")" && printf %s "$2" | base64 -d > "$1"'
_WRITE_MORE = 'printf %s "$2" | base64 -d >> "$1"'

# How often a sandbox that boots cold is polled.
_POLL_INTERVAL_S = 1.0

# States a booting sandbox does not leave on its own.
_SETTLED = frozenset({"running", "error", "deleted", "stopped", "host_removed", "host_lost", "sleeping", "paused"})


class Kosa8APIError(Exception):
    """An error answered by kosa8 Cloud.

    ``code`` is the API's stable error code, such as ``payment_required``,
    ``quota_exceeded`` or ``exec_timeout``.
    """

    def __init__(self, message: str, *, status: int, code: str | None = None, body: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.body = body


def check_env_names(env: Mapping[str, str] | None) -> None:
    """Raise ``ValueError`` unless every name in ``env`` is a shell identifier."""
    for key in env or {}:
        if not _ENV_NAME.match(key):
            raise ValueError(f"Invalid environment variable name for a kosa8 sandbox: {key!r}")


def _api_error(response: httpx.Response) -> Kosa8APIError:
    try:
        body: Any = response.json()
    except ValueError:
        body = response.text
    code = None
    message = ""
    if isinstance(body, dict):
        code = body.get("error") if isinstance(body.get("error"), str) else None
        message = str(body.get("message") or body.get("error") or "")
    elif body:
        message = str(body)[:500]
    return Kosa8APIError(
        message or f"kosa8 Cloud answered HTTP {response.status_code}",
        status=response.status_code,
        code=code,
        body=body,
    )


async def _call(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    *,
    json_body: Any = None,
    content: bytes | None = None,
    params: dict[str, str] | None = None,
    timeout: float | None = None,
) -> Any:
    """One API call; the parsed JSON answer, or :class:`Kosa8APIError`."""
    headers = {"Content-Type": "application/octet-stream"} if content is not None else None
    response = await client.request(
        method,
        path,
        json=json_body,
        content=content,
        params=params,
        headers=headers,
        timeout=timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT,
    )
    if response.status_code >= 400:
        raise _api_error(response)
    if not response.content:
        return None
    try:
        return response.json()
    except ValueError:
        return None


def _env_list(env: Mapping[str, str]) -> list[str]:
    return [f"{key}={value}" for key, value in env.items()]


def _joined(stdout: str, stderr: str) -> str:
    return (stdout + stderr).strip()


def _in_workspace(path: PurePosixPath) -> bool:
    return path.is_relative_to(DEFAULT_WORKDIR)


class Kosa8Sandbox(SandboxBase):
    """Sandbox backed by a kosa8 Cloud sandbox (a microVM).

    Creation is lazy and failure-atomic after the API returns a sandbox ID:
    errors while it starts, and cancellation, delete the new sandbox. A delete
    that fails keeps the sandbox handle, so :meth:`aclose`, the next creation
    attempt and the atexit fallback can retry it. A ``ttl_s`` in
    ``create_options`` remains the server-side backstop.

    Each command runs in a fresh container of the sandbox's template image
    (unless the sandbox is ``persistent``); the workdir, ``/workspace`` by
    default, is the disk they all share.

    Use one instance from one event loop at a time; creation and close are
    serialized per loop.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        create_options: dict[str, Any],
        labels: dict[str, str] | None = None,
        egress: Mapping[str, Any] | None = None,
        timeout: float = 60,
        start_timeout: float = 300,
        workdir: str = DEFAULT_WORKDIR,
        env: dict[str, str] | None = None,
    ) -> None:
        for name, value in (("client", client), ("create_options", create_options)):
            if isinstance(value, Variable):
                raise TypeError(
                    f"Kosa8Sandbox.{name} must be a concrete value; got Variable. "
                    "Wrap with Kosa8Environment to resolve Variables from a Context."
                )
        if timeout <= 0:
            raise ValueError("`timeout` must be greater than 0 seconds.")
        if start_timeout <= 0:
            raise ValueError("`start_timeout` must be greater than 0 seconds.")
        if not PurePosixPath(workdir).is_absolute():
            raise ValueError(f"`workdir` must be an absolute path: {workdir}")
        check_env_names(env)

        self._client: httpx.AsyncClient | None = client
        self._create_options = create_options
        self._labels = dict(labels or {})
        self._egress = dict(egress) if egress is not None else None
        self._default_timeout = timeout
        self._start_timeout = start_timeout
        self._workdir = PurePosixPath(workdir)
        self._env = dict(env or {})
        self._sandbox_id: str | None = None
        self._ready = False
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self._atexit_registered = False
        self._files_api = True
        # Captured up front: the atexit fallback runs after `aclose` has
        # dropped the async client, and has to build a fresh sync one.
        self._sync_base_url = str(client.base_url)
        self._sync_headers = dict(client.headers)

    @property
    def workdir(self) -> PurePosixPath:
        return self._workdir

    @property
    def host_workdir(self) -> Path | None:
        return None

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def sandbox_id(self) -> str | None:
        """ID of the kosa8 sandbox (``sbx_...``), once it has been created."""
        return self._sandbox_id

    def _creation_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    async def __aenter__(self) -> "Kosa8Sandbox":
        await self._ensure_sandbox()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def __deepcopy__(self, memo: dict[int, Any]) -> "Kosa8Sandbox":
        return self

    async def exec(
        self,
        argv: list[str],
        *,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        if not argv:
            return ExecResult(output="", exit_code=2)
        check_env_names(env)
        merged_env = {**self._env, **(env or {})}

        client, sandbox_id = await self._ensure_sandbox()
        exec_timeout = timeout if timeout is not None else self._default_timeout
        # The API takes whole seconds, at least one.
        timeout_s = max(1, math.ceil(exec_timeout))
        body: dict[str, Any] = {"cmd": [*_IN_WORKDIR, str(self._workdir), *argv], "timeout_s": timeout_s}
        if merged_env:
            body["env"] = _env_list(merged_env)
        try:
            if timeout_s <= BUFFERED_EXEC_LIMIT_S:
                answer = await _call(
                    client,
                    "POST",
                    f"{self._path(sandbox_id)}/exec",
                    json_body=body,
                    timeout=timeout_s + 30,
                )
                output = _joined(answer.get("stdout") or "", answer.get("stderr") or "")
                exit_code = answer.get("exit")
            else:
                output, exit_code = await self._exec_streamed(client, sandbox_id, body)
        except Kosa8APIError as e:
            if e.code == "exec_timeout":
                partial = ""
                if isinstance(e.body, dict):
                    partial = _joined(str(e.body.get("stdout") or ""), str(e.body.get("stderr") or ""))
                note = f"kosa8 execution timed out after {exec_timeout}s"
                return ExecResult(output=f"{partial}\n{note}" if partial else note, exit_code=124)
            return ExecResult(output=f"kosa8 error: {e}", exit_code=1)
        except httpx.TimeoutException:
            return ExecResult(output=f"kosa8 execution timed out after {exec_timeout}s", exit_code=124)
        except httpx.HTTPError as e:
            return ExecResult(output=f"kosa8 error: {e}", exit_code=1)

        # Commands run through a POSIX shell, so the exit code already follows
        # POSIX: 127 not found, 126 not executable, 128+N for signal N.
        if not isinstance(exit_code, int):
            note = "kosa8 execution ended without an exit status"
            return ExecResult(output=f"{output}\n{note}" if output else note, exit_code=1)
        return ExecResult(output=output, exit_code=exit_code)

    async def _exec_streamed(
        self,
        client: httpx.AsyncClient,
        sandbox_id: str,
        body: dict[str, Any],
    ) -> tuple[str, int | None]:
        """Run a command past the buffered limit as a stream of NDJSON frames:
        ``{stream, data: base64}`` while it prints (empty ones are heartbeats),
        then ``{done: true, exit}`` or ``{done: true, error, code}``."""
        chunks: list[bytes] = []
        done: dict[str, Any] = {}
        # The host sends a heartbeat frame every 15 s while the command is quiet.
        read_timeout = httpx.Timeout(30.0, read=60.0)
        async with client.stream(
            "POST",
            f"{self._path(sandbox_id)}/exec",
            json={**body, "stream": True},
            timeout=read_timeout,
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                raise _api_error(response)
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                frame = json.loads(line)
                if frame.get("done"):
                    done = frame
                    break
                if frame.get("data"):
                    chunks.append(base64.b64decode(frame["data"]))
        output = b"".join(chunks).decode("utf-8", errors="replace").strip()
        if done.get("code") == "exec_timeout":
            raise Kosa8APIError(
                str(done.get("error") or "exec timed out"), status=504, code="exec_timeout", body={"stdout": output}
            )
        if done.get("error"):
            message = f"kosa8 error: {done['error']}"
            return (f"{output}\n{message}" if output else message), 1
        return output, done.get("exit")

    async def put_file(self, path: PurePosixPath, content: bytes) -> None:
        if path.is_absolute():
            raise ValueError(f"Absolute paths are not allowed in put_file: {path}")
        client, sandbox_id = await self._ensure_sandbox()
        target = self._workdir / path
        # The files API reaches /workspace only, one file of at most 8 MiB.
        if self._files_api and _in_workspace(target) and len(content) <= MAX_FILE_API_BYTES:
            try:
                if target.parent != PurePosixPath(DEFAULT_WORKDIR):
                    await _call(
                        client,
                        "POST",
                        f"{self._path(sandbox_id)}/files/mkdir",
                        json_body={"path": str(target.parent)},
                    )
                await _call(
                    client,
                    "PUT",
                    f"{self._path(sandbox_id)}/files",
                    params={"path": str(target)},
                    content=content,
                )
                return
            except Kosa8APIError as e:
                if e.status != 501:
                    raise
                # A host without the files API: write through exec from now on.
                self._files_api = False
        await self._write_through_exec(target, content)

    async def _write_through_exec(self, target: PurePosixPath, content: bytes) -> None:
        offsets = range(0, len(content), _EXEC_WRITE_CHUNK) if content else range(1)
        for offset in offsets:
            chunk = base64.b64encode(content[offset : offset + _EXEC_WRITE_CHUNK]).decode("ascii")
            script = _WRITE_FIRST if offset == 0 else _WRITE_MORE
            result = await self.exec(["sh", "-c", script, "ag2", str(target), chunk])
            if result.exit_code != 0:
                raise RuntimeError(f"Could not write {target} in kosa8 sandbox: {result.output}")

    @staticmethod
    def _path(sandbox_id: str) -> str:
        return f"/v1/sandboxes/{quote(sandbox_id, safe='')}"

    async def _ensure_sandbox(self) -> tuple[httpx.AsyncClient, str]:
        if self._closed:
            raise RuntimeError("Kosa8Sandbox has been closed.")
        if self._client is not None and self._sandbox_id is not None and self._ready:
            return self._client, self._sandbox_id

        async with self._creation_lock():
            if self._closed:
                raise RuntimeError("Kosa8Sandbox has been closed.")
            if self._client is None:
                raise RuntimeError("Kosa8Sandbox client has been closed.")
            client = self._client
            if self._sandbox_id is not None and self._ready:
                return client, self._sandbox_id
            # A sandbox left behind by a failed setup must be gone before a new one exists.
            if self._sandbox_id is not None and not await self._delete():
                raise RuntimeError("A kosa8 sandbox from a failed setup could not be deleted; retry later.")

            answer = await _call(client, "POST", "/v1/sandboxes", json_body=self._create_options)
            info = answer["sandbox"]
            sandbox_id = str(info["id"])
            self._sandbox_id = sandbox_id
            self._register_atexit()
            try:
                await self._wait_until_running(client, sandbox_id, info)
                if self._labels:
                    await _call(client, "PUT", f"{self._path(sandbox_id)}/labels", json_body={"labels": self._labels})
                if self._egress is not None:
                    await _call(client, "PUT", f"{self._path(sandbox_id)}/egress", json_body=self._egress)
                if self._workdir != PurePosixPath(DEFAULT_WORKDIR):
                    # Commands `cd` into the workdir first, and fail if it is missing.
                    prepared = await _call(
                        client,
                        "POST",
                        f"{self._path(sandbox_id)}/exec",
                        json_body={"cmd": ["mkdir", "-p", str(self._workdir)]},
                        timeout=BUFFERED_EXEC_LIMIT_S + 30,
                    )
                    if prepared.get("exit") != 0:
                        detail = _joined(prepared.get("stdout") or "", prepared.get("stderr") or "")
                        raise RuntimeError(f"Could not create workdir {self._workdir} in kosa8 sandbox: {detail}")
            except BaseException:
                # On failure the handle stays for `aclose`, the next attempt
                # and the atexit fallback to retry.
                await asyncio.shield(self._delete())
                raise

            self._ready = True
            logger.info("kosa8 sandbox created (id=%s, template=%s)", sandbox_id, info.get("template"))
            return client, sandbox_id

    async def _wait_until_running(self, client: httpx.AsyncClient, sandbox_id: str, info: dict[str, Any]) -> None:
        # A template sandbox answers `running`; a sized one boots cold and
        # answers `creating` until its host reports it running.
        deadline = time.monotonic() + self._start_timeout
        while info.get("state") not in _SETTLED:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"kosa8 sandbox {sandbox_id} still {info.get('state')} after {self._start_timeout}s")
            await asyncio.sleep(_POLL_INTERVAL_S)
            info = (await _call(client, "GET", self._path(sandbox_id)))["sandbox"]
        if info.get("state") != "running":
            reason = info.get("last_error") or f"it is {info.get('state')}"
            raise RuntimeError(f"kosa8 sandbox {sandbox_id} did not start: {reason}")

    async def aclose(self) -> None:
        """Delete the sandbox and release the client.

        Safe to call repeatedly. Waits for a creation in progress on the same
        event loop, so a sandbox that is still being created is deleted too.
        A delete that fails keeps both the sandbox handle and the client so a
        later call can retry; the atexit fallback stays armed for the case
        where no retry comes.
        """
        self._closed = True
        async with self._creation_lock():
            if self._sandbox_id is not None and not await self._delete():
                return
            if self._client is not None:
                try:
                    await self._client.aclose()
                except Exception as e:
                    logger.debug("Suppressed exception during kosa8 client close: %s", e)
                self._client = None

    async def _delete(self) -> bool:
        """Delete the sandbox this instance owns; whether it is gone afterwards."""
        sandbox_id = self._sandbox_id
        assert sandbox_id is not None and self._client is not None
        try:
            await _call(self._client, "DELETE", self._path(sandbox_id))
        except Kosa8APIError as e:
            if e.status not in (404, 410):  # 404/410: already gone, e.g. by its ttl
                logger.debug("Suppressed exception during kosa8 sandbox delete: %s", e)
                return False
        except Exception as e:
            logger.debug("Suppressed exception during kosa8 sandbox delete: %s", e)
            return False
        # A shielded delete can finish after a later call already replaced the handle.
        if self._sandbox_id == sandbox_id:
            self._sandbox_id = None
            self._ready = False
            self._unregister_atexit()
        return True

    def _register_atexit(self) -> None:
        if not self._atexit_registered:
            atexit.register(self._atexit_close)
            self._atexit_registered = True

    def _unregister_atexit(self) -> None:
        if self._atexit_registered:
            atexit.unregister(self._atexit_close)
            self._atexit_registered = False

    def _atexit_close(self) -> None:
        if self._sandbox_id is None:
            return
        try:
            with httpx.Client(base_url=self._sync_base_url, headers=self._sync_headers, timeout=30) as client:
                client.delete(self._path(self._sandbox_id))
        except Exception as e:
            logger.debug("Suppressed exception during atexit kosa8 sandbox cleanup: %s", e)
