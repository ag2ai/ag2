# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

ChangeCallback = Callable[[str], Awaitable[None]]


class ChangeSubscription(Protocol):
    """Handle returned by :meth:`KnowledgeStore.on_change`.

    Closing the subscription stops delivery of change notifications. This
    is filesystem-level reactivity for the backing store — not to be
    confused with ``autogen.beta.watch.Watch``, which is the event- and
    time-pattern trigger system used by the framework-core ``Scheduler``.
    """

    async def close(self) -> None:
        """Stop receiving change notifications."""
        ...


class NoopChangeSubscription:
    """Sentinel returned by backends that cannot observe changes efficiently.

    The hub falls back to polling when it sees a
    :class:`NoopChangeSubscription`.
    """

    async def close(self) -> None:
        return None


@runtime_checkable
class KnowledgeStore(Protocol):
    """Virtual path-based store for actor knowledge.

    Provides filesystem semantics over any storage backend.
    Paths use Unix conventions: /dir/subdir/file.txt
    Directories are implicit -- writing /a/b/c.txt implies /a/ and /a/b/ exist.
    Listing returns immediate children. Directory entries end with '/'.

    The network layer uses the same protocol to back the hub's virtual file
    system; see :mod:`autogen.beta.network`. Three methods beyond the basic
    CRUD are required for WAL-backed sessions: ``append`` and ``read_range``
    are mandatory, while ``on_change`` is optional (backends that cannot
    observe changes efficiently return a :class:`NoopChangeSubscription`
    and callers fall back to polling).
    """

    async def read(self, path: str) -> str | None:
        """Read content at path. Returns None if not found."""
        ...

    async def write(self, path: str, content: str) -> None:
        """Write content to path. Creates parent directories implicitly."""
        ...

    async def list(self, path: str = "/") -> list[str]:
        """List immediate children of a directory path.

        Returns relative names. Directories end with '/'.
        Example: list("/log/") might return ["stream-abc.jsonl", "stream-def.jsonl"]
        """
        ...

    async def delete(self, path: str) -> None:
        """Delete entry at path. No-op if not found."""
        ...

    async def exists(self, path: str) -> bool:
        """Check if path exists."""
        ...

    async def append(self, path: str, content: str) -> int:
        """Atomically append ``content`` to the file at ``path``.

        Creates the file (and its parents) if it does not exist. Returns the
        byte offset at which ``content`` was written, so callers can record a
        cursor for later ``read_range`` calls.
        """
        ...

    async def read_range(self, path: str, start: int, end: int | None = None) -> str:
        """Read the byte slice ``[start, end)`` of the file at ``path``.

        ``end`` of ``None`` means "up to the current end of file". Returns an
        empty string if the file does not exist. Slices are returned as UTF-8
        text; callers that append multi-byte content must align offsets to
        character boundaries themselves.
        """
        ...

    async def on_change(self, path: str, callback: ChangeCallback) -> ChangeSubscription:
        """Subscribe to change notifications at ``path``.

        Backends that can observe changes efficiently invoke
        ``callback(path)`` whenever a file under ``path`` changes. Backends
        that cannot must return :class:`NoopChangeSubscription`; the hub
        then polls on a short interval instead.
        """
        ...


def _normalize(path: str) -> str:
    """Normalize path: ensure leading /, collapse //, strip trailing /."""
    if not path.startswith("/"):
        path = "/" + path
    while "//" in path:
        path = path.replace("//", "/")
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return path
