# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol, runtime_checkable


@dataclass(slots=True)
class ExecResult:
    """Outcome of a single :meth:`Sandbox.exec` call.

    ``output`` carries the combined ``stdout + stderr`` (already trimmed
    by the backend, optionally truncated to ``max_output`` characters).
    ``exit_code`` follows POSIX conventions (``0`` = success, ``124`` =
    timeout, ``127`` = command not found).
    """

    output: str
    exit_code: int


@runtime_checkable
class Sandbox(Protocol):
    """Low-level execution backend.

    A ``Sandbox`` runs an ``argv`` list inside a working directory and
    returns combined output with an exit code. It is the shared primitive
    on top of which the higher-level adapters are built:

    - :class:`~ag2.tools.sandbox.adapter.ShellAdapter` —
      one ``run(command)`` per shell command, applied filters, sync facade.
    - :class:`~ag2.tools.sandbox.adapter.CodeAdapter` —
      one ``run(code, language)`` per code snippet, language matrix.

    Implementations target local subprocesses, Docker containers, Daytona
    sandboxes, SSH, or anything else. Adding a new backend = one new
    ``Sandbox`` class. Both shell and code semantics come for free via the
    adapters.

    Sandbox is async-only. Construction is cheap; backend resources are
    acquired on ``__aenter__`` (or first call when ``__aenter__`` is a
    no-op, as for :class:`LocalSandbox`) and released on ``__aexit__``.
    """

    @property
    def workdir(self) -> PurePosixPath:
        """Working directory in which ``argv`` is executed, as seen *inside*
        the sandbox.

        POSIX even on a Windows host so the same path is meaningful across
        backends.
        """
        ...

    @property
    def host_workdir(self) -> Path | None:
        """Host-side view of :attr:`workdir`, when one exists.

        Returns ``None`` for backends whose workdir is not visible on the
        host filesystem (e.g. remote Daytona sandboxes).
        """
        ...

    async def __aenter__(self) -> "Sandbox": ...

    async def __aexit__(self, *exc: object) -> None: ...

    async def exec(
        self,
        argv: list[str],
        *,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        """Execute ``argv`` and return its output.

        Args:
            argv: Command and arguments. Always a normal argv list — no
                  shell parsing happens on this layer. Callers that need
                  shell features pass ``["sh", "-c", cmd]`` explicitly
                  (the :class:`ShellAdapter` does this for them).
            env: Extra environment variables merged into the process
                 environment. ``None`` inherits the parent environment as-is.
            timeout: Per-call timeout in seconds. ``None`` uses the
                     backend's default. On timeout, the implementation
                     must terminate the process and return
                     ``exit_code=124``.
        """
        ...

    async def put_file(self, path: PurePosixPath, content: bytes) -> None:
        """Write ``content`` to ``path`` inside the sandbox.

        ``path`` is relative to :attr:`workdir` (absolute paths are
        rejected to keep writes confined to the workdir).

        Optional capability. Backends without filesystem write support
        raise :class:`NotImplementedError`.
        """
        ...

    async def remove_file(self, path: PurePosixPath) -> None:
        """Delete ``path`` inside the sandbox. Best-effort; never raises if
        the file is already gone.

        ``path`` is relative to :attr:`workdir`. Used by the
        :class:`~ag2.tools.sandbox.adapter.CodeAdapter` to clean up
        the temp files it writes for file-mode languages.
        """
        ...

    async def aclose(self) -> None:
        """Release backend resources.

        Equivalent to ``__aexit__``. Safe to call multiple times.
        """
        ...


class SandboxBase(ABC):
    """ABC with default implementations for the optional :class:`Sandbox`
    methods.

    Concrete backends subclass :class:`SandboxBase` and override
    :meth:`exec`, :attr:`workdir`, :attr:`host_workdir`. They opt into
    streaming and file IO by overriding the relevant methods.
    """

    @property
    @abstractmethod
    def workdir(self) -> PurePosixPath: ...

    @property
    @abstractmethod
    def host_workdir(self) -> Path | None: ...

    @abstractmethod
    async def exec(
        self,
        argv: list[str],
        *,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecResult: ...

    async def put_file(self, path: PurePosixPath, content: bytes) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not support put_file. Override the method on the subclass."
        )

    async def remove_file(self, path: PurePosixPath) -> None:
        """Default cleanup: ``rm -f`` inside the sandbox.

        Works on any POSIX backend that can ``exec``. Backends with a native
        delete API (local filesystem, Daytona ``fs.delete_file``) should
        override for directness. Never raises on a missing file.
        """
        if path.is_absolute():
            raise ValueError(f"Absolute paths are not allowed in remove_file: {path}")
        target = self.workdir / path
        await self.exec(["rm", "-f", str(target)])

    async def aclose(self) -> None:
        """Default cleanup: nothing.

        Backends that hold resources (containers, remote sandboxes)
        override this method.
        """

    async def __aenter__(self) -> "SandboxBase":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
