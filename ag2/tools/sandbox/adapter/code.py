# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from ag2.tools.code.environment.base import CodeLanguage, CodeRunResult
from ag2.tools.sandbox.base import Sandbox
from ag2.tools.sandbox.factory import SandboxFactory, SingletonFactory

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ag2.context import ConversationContext


@dataclass(slots=True, frozen=True)
class LanguageRunner:
    """How a language maps to an ``argv`` invocation.

    Exactly one of two modes:

    - **Inline** — ``inline_argv`` is provided. The adapter calls
      ``sandbox.exec([*inline_argv, code])``. Best for languages whose
      interpreter accepts ``-c`` (python, bash).
    - **File** — ``file_extension`` and ``file_runner_argv`` are
      provided. The adapter writes ``code`` to a temp file via
      ``sandbox.put_file`` and then ``sandbox.exec([*file_runner_argv,
      <path>])``. Best for languages without a robust ``-c`` (node,
      ts-node).
    """

    inline_argv: tuple[str, ...] | None = None
    file_extension: str | None = None
    file_runner_argv: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        inline = self.inline_argv is not None
        file_mode = self.file_extension is not None or self.file_runner_argv is not None
        if inline == file_mode:
            raise ValueError(
                "LanguageRunner must be either inline (inline_argv) or file-based "
                "(file_extension + file_runner_argv), not both or neither."
            )
        if file_mode and (self.file_extension is None or self.file_runner_argv is None):
            raise ValueError("File-mode LanguageRunner requires both file_extension and file_runner_argv.")


DEFAULT_RUNNERS: dict[CodeLanguage, LanguageRunner] = {
    "python": LanguageRunner(inline_argv=("python", "-c")),
    "bash": LanguageRunner(inline_argv=("bash", "-c")),
    "javascript": LanguageRunner(file_extension="js", file_runner_argv=("node",)),
    "typescript": LanguageRunner(file_extension="ts", file_runner_argv=("ts-node",)),
}


class CodeAdapter:
    """One :class:`~ag2.tools.code.CodeEnvironment` implementation
    that works on every :class:`Sandbox`.

    The language matrix (python/bash → inline; js/ts → file) lives here
    once. Backends only need to implement
    :meth:`Sandbox.exec` (always) plus :meth:`Sandbox.put_file` (only for
    file-mode languages).

    Args:
        sandbox: Either a long-lived :class:`Sandbox` or a
                 :class:`SandboxFactory` (opened per :meth:`run`).
        languages: Languages this adapter is willing to accept; surfaced
                   as :attr:`supported_languages`.
        runners: Override or extend :data:`DEFAULT_RUNNERS`. Use this
                 hook to add new languages or tweak interpreters
                 (``ts-node`` vs ``tsx``, etc.).
        timeout: Per-call timeout in seconds. ``None`` lets the backend
                 pick its default.
    """

    def __init__(
        self,
        sandbox: "Sandbox | SandboxFactory",
        *,
        languages: tuple[CodeLanguage, ...] = ("python", "bash"),
        runners: dict[CodeLanguage, LanguageRunner] | None = None,
        timeout: float | None = None,
    ) -> None:
        merged = dict(DEFAULT_RUNNERS)
        if runners:
            merged.update(runners)

        for lang in languages:
            if lang not in merged:
                raise ValueError(f"No LanguageRunner registered for language {lang!r}.")

        self._factory: SandboxFactory = sandbox if isinstance(sandbox, SandboxFactory) else SingletonFactory(sandbox)
        self._languages: tuple[CodeLanguage, ...] = tuple(languages)
        self._runners = merged
        self._timeout = timeout

    @property
    def supported_languages(self) -> tuple[CodeLanguage, ...]:
        return self._languages

    async def run(
        self,
        code: str,
        language: CodeLanguage,
        *,
        context: "ConversationContext | None" = None,
    ) -> CodeRunResult:
        if language not in self._languages:
            return CodeRunResult(
                output=f"Language {language!r} is not enabled. Available: {list(self._languages)}",
                exit_code=2,
            )

        runner = self._runners[language]

        async with self._open(context) as sandbox:
            if runner.inline_argv is not None:
                result = await sandbox.exec(
                    [*runner.inline_argv, code],
                    timeout=self._timeout,
                )
                return CodeRunResult(output=result.output, exit_code=result.exit_code)

            assert runner.file_extension is not None and runner.file_runner_argv is not None
            filename = f"ag2_{uuid.uuid4().hex}.{runner.file_extension}"
            path = PurePosixPath(filename)
            await sandbox.put_file(path, code.encode("utf-8"))
            target = sandbox.workdir / path
            try:
                result = await sandbox.exec(
                    [*runner.file_runner_argv, str(target)],
                    timeout=self._timeout,
                )
            finally:
                # Best-effort cleanup so persistent sandboxes (e.g. a
                # reused container or a LocalSandbox over a fixed workdir)
                # do not accumulate temp scripts. Never fails the run.
                with suppress(Exception):
                    await sandbox.remove_file(path)
            return CodeRunResult(output=result.output, exit_code=result.exit_code)

    @asynccontextmanager
    async def _open(
        self,
        context: "ConversationContext | None",
    ) -> "AsyncIterator[Sandbox]":
        async with self._factory.open(context) as sb:
            yield sb
