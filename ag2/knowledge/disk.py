# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import os
import shutil
from pathlib import Path
from typing import Any

from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver

from .base import ChangeCallback, ChangeSubscription, NoopChangeSubscription, _normalize


class _DiskChangeHandler:
    """watchdog ``FileSystemEventHandler`` that bridges to an async callback.

    The watchdog observer runs in a background thread and calls this
    handler synchronously on file events. We translate the physical
    path back into its store-relative virtual form and schedule the
    async ``callback`` on the main event loop via
    :func:`asyncio.run_coroutine_threadsafe`.

    Directory events are ignored — only file-level writes / creations /
    deletions / moves are delivered, matching the ``MemoryKnowledgeStore``
    contract where every change is observed as a file path.
    """

    def __init__(
        self,
        *,
        root: Path,
        virtual_prefix: str,
        loop: asyncio.AbstractEventLoop,
        callback: ChangeCallback,
    ) -> None:
        self._root = root
        self._virtual_prefix = virtual_prefix
        self._loop = loop
        self._callback = callback

    def _virtual_path_for(self, src_path: str) -> str | None:
        try:
            rel = Path(src_path).resolve().relative_to(self._root)
        except ValueError:
            return None
        virtual = "/" + str(rel).replace("\\", "/")
        if self._virtual_prefix != "/":
            prefix = self._virtual_prefix.rstrip("/") + "/"
            if virtual != self._virtual_prefix and not virtual.startswith(prefix):
                return None
        return virtual

    def _dispatch(self, src_path: str) -> None:
        virtual = self._virtual_path_for(src_path)
        if virtual is None:
            return
        coro = self._callback(virtual)
        try:
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        except RuntimeError:
            coro.close()

    def on_modified(self, event: Any) -> None:
        if getattr(event, "is_directory", False):
            return
        self._dispatch(event.src_path)

    def on_created(self, event: Any) -> None:
        if getattr(event, "is_directory", False):
            return
        self._dispatch(event.src_path)

    def on_deleted(self, event: Any) -> None:
        if getattr(event, "is_directory", False):
            return
        self._dispatch(event.src_path)

    def on_moved(self, event: Any) -> None:
        if getattr(event, "is_directory", False):
            return
        dest = getattr(event, "dest_path", None) or event.src_path
        self._dispatch(dest)

    def dispatch(self, event: Any) -> None:
        """watchdog's entry point. Delegates to the per-type hooks above."""
        event_type = getattr(event, "event_type", "")
        if event_type == "modified":
            self.on_modified(event)
        elif event_type == "created":
            self.on_created(event)
        elif event_type == "deleted":
            self.on_deleted(event)
        elif event_type == "moved":
            self.on_moved(event)


class _DiskChangeSubscription:
    """Handle returned by :meth:`DiskKnowledgeStore.on_change`.

    Wraps the live watchdog ``Observer`` and ensures ``close()`` stops
    and joins the background thread so the subscription never leaks a
    daemon thread past the caller's lifetime.
    """

    def __init__(self, observer: Any) -> None:
        self._observer = observer
        self._closed = False

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._shutdown)

    def _shutdown(self) -> None:
        with contextlib.suppress(Exception):  # pragma: no cover — watchdog internals
            self._observer.unschedule_all()
        with contextlib.suppress(Exception):  # pragma: no cover
            self._observer.stop()
        with contextlib.suppress(Exception):  # pragma: no cover
            self._observer.join(timeout=2.0)


class DiskKnowledgeStore:
    """Persistent KnowledgeStore backed by the local filesystem.

    Maps virtual paths directly to real files under a root directory.
    Directories are created on write. Supports macOS and Linux.
    Not supported on Windows (filenames may contain characters that
    are illegal on NTFS such as ``:``, ``?``, ``*``, ``<``, ``>``).

    Example::

        store = DiskKnowledgeStore("/tmp/my-agent")
        await store.write("/artifacts/report.md", "# Report")
        # Creates /tmp/my-agent/artifacts/report.md on disk
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root)

    def _resolve(self, path: str) -> Path:
        """Map virtual path to real filesystem path."""
        normalized = _normalize(path).lstrip("/")
        resolved = (self._root / normalized).resolve() if normalized else self._root.resolve()
        if not str(resolved).startswith(str(self._root.resolve())):
            raise ValueError(f"Path traversal blocked: {path}")
        return resolved

    async def read(self, path: str) -> str | None:
        target = self._resolve(path)
        if not target.is_file():
            return None
        return target.read_text(encoding="utf-8")

    async def write(self, path: str, content: str) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    async def list(self, path: str = "/") -> list[str]:
        target = self._resolve(path)
        if not target.is_dir():
            return []
        children: list[str] = []
        for entry in sorted(target.iterdir()):
            if entry.is_dir():
                children.append(entry.name + "/")
            else:
                children.append(entry.name)
        return children

    async def delete(self, path: str) -> None:
        target = self._resolve(path)
        if target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)

    async def exists(self, path: str) -> bool:
        return self._resolve(path).exists()

    async def append(self, path: str, content: str) -> int:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = content.encode("utf-8")
        with target.open("ab") as fh:
            offset = fh.tell()
            fh.write(payload)
        return offset

    async def read_range(self, path: str, start: int, end: int | None = None) -> str:
        target = self._resolve(path)
        if not target.is_file():
            return ""
        with target.open("rb") as fh:
            fh.seek(start)
            if end is None:
                data = fh.read()
            else:
                span = max(0, end - start)
                data = fh.read(span)
        return data.decode("utf-8", errors="strict")

    async def on_change(self, path: str, callback: ChangeCallback) -> ChangeSubscription:
        """Subscribe to filesystem change notifications under ``path``.

        Uses the ``watchdog`` library to dispatch platform-native events
        (inotify on Linux, FSEvents on macOS, ReadDirectoryChangesW on
        Windows). Falls back to :class:`PollingObserver` if the native
        backend cannot be initialized, and to :class:`NoopChangeSubscription`
        if ``watchdog`` is not installed at all.
        """

        virtual_path = _normalize(path)
        physical_target = self._resolve(virtual_path)
        physical_target.mkdir(parents=True, exist_ok=True)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return NoopChangeSubscription()

        handler = _DiskChangeHandler(
            root=self._root.resolve(),
            virtual_prefix=virtual_path,
            loop=loop,
            callback=callback,
        )

        observer: Any
        try:
            observer = Observer()
            observer.schedule(handler, str(physical_target), recursive=True)
            observer.start()
        except Exception:
            observer = PollingObserver()
            observer.schedule(handler, str(physical_target), recursive=True)
            observer.start()

        return _DiskChangeSubscription(observer)
