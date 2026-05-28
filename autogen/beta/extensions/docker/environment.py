# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64
import os
import uuid
from typing import TYPE_CHECKING

from autogen.beta.annotations import Variable
from autogen.beta.tools.code import CodeEnvironment, CodeLanguage, CodeRunResult
from autogen.beta.tools.sandbox import CodeAdapter, LanguageRunner
from autogen.beta.tools.sandbox.adapter.code import DEFAULT_RUNNERS

from .factory import DockerSandboxFactory

if TYPE_CHECKING:
    from autogen.beta.context import ConversationContext


class DockerCodeEnvironment(CodeEnvironment):
    """Deprecated alias — use ``CodeAdapter(DockerSandboxFactory(...))``.

    Retains the v1 ``base64-shell-hack`` for ``javascript``/``typescript``
    because :class:`DockerSandbox` does support ``put_file`` natively but
    this façade is kept stable for callers built against v1 (they expect
    the historical ``exec_run`` argv shape). New code should drop this
    class and use :class:`CodeAdapter` directly.
    """

    def __init__(
        self,
        *,
        image: "str | Variable" = "python:3.12-slim",
        env_vars: "dict[str, str] | Variable | None" = None,
        timeout: int = 60,
        network_mode: "str | Variable" = "none",
        mem_limit: str | None = "512m",
        cpu_quota: int | None = None,
        user: str | None = None,
        auto_remove: bool = True,
        languages: tuple[CodeLanguage, ...] = ("python", "bash"),
        host_path: str | os.PathLike[str] | None = None,
        workdir: str = "/workspace",
    ) -> None:
        if timeout < 1:
            raise ValueError("`timeout` must be >= 1 second.")

        self._factory = DockerSandboxFactory(
            image=image,
            env_vars=env_vars,
            timeout=timeout,
            network_mode=network_mode,
            mem_limit=mem_limit,
            cpu_quota=cpu_quota,
            user=user,
            auto_remove=auto_remove,
            host_path=host_path,
            workdir=workdir,
        )
        self._timeout = timeout
        self._languages: tuple[CodeLanguage, ...] = tuple(languages)
        self._workdir = workdir

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

        runner = DEFAULT_RUNNERS[language]
        if runner.inline_argv is not None:
            cmd: list[str] = [*runner.inline_argv, code]
        else:
            assert runner.file_extension is not None and runner.file_runner_argv is not None
            file_runner = runner.file_runner_argv[0]
            script_path = f"{self._workdir}/ag2_{uuid.uuid4().hex}.{runner.file_extension}"
            encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
            cmd = [
                "sh",
                "-c",
                f"echo {encoded} | base64 -d > {script_path} && {file_runner} {script_path}; "
                f"rc=$?; rm -f {script_path}; exit $rc",
            ]

        async with self._factory.open(context) as sandbox:
            result = await sandbox.exec(cmd, timeout=self._timeout)
        return CodeRunResult(output=result.output, exit_code=result.exit_code)

    async def aclose(self) -> None:
        """No-op: factory opens / closes a container per :meth:`run`."""

    async def __aenter__(self) -> "DockerCodeEnvironment":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


__all__ = ("CodeAdapter", "DockerCodeEnvironment", "LanguageRunner")
