# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import fnmatch
import shlex
from pathlib import Path
from typing import Protocol, runtime_checkable

# Commands that only read state and never modify the filesystem.
# Used when ``LocalShellEnvironment.readonly=True`` and no explicit ``allowed``
# list is provided.  This is a best-effort list — commands like ``echo`` can
# still redirect output (``echo x > file``), because ``shell=True`` processing
# happens inside the OS shell after our prefix check.
READONLY_COMMANDS: tuple[str, ...] = (
    "cat",
    "head",
    "tail",
    "ls",
    "ll",
    "la",
    "grep",
    "egrep",
    "fgrep",
    "find",
    "wc",
    "du",
    "df",
    "diff",
    "stat",
    "file",
    "which",
    "pwd",
    "echo",
    "env",
    "printenv",
    "sort",
    "uniq",
    "cut",
    "git log",
    "git diff",
    "git status",
    "git show",
    "git branch",
)


def matches(pattern: str, command: str) -> bool:
    """Return True if *command* starts with *pattern* as a whole word or prefix.

    ``"git"`` matches ``"git status"`` and ``"git"`` but not ``"gitconfig"``.
    ``"uv run"`` matches ``"uv run pytest"`` but not ``"uv add requests"``.
    """
    stripped = command.strip()
    if not stripped.startswith(pattern):
        return False
    rest = stripped[len(pattern) :]
    return rest == "" or rest[0] == " "


def check_ignore(command: str, workdir: Path, patterns: list[str]) -> str | None:
    """Return ``"Access denied: <path>"`` if any literal path in *command* matches *patterns*.

    Tokens are extracted via :func:`shlex.split` to handle quoted paths. Each
    token is resolved relative to *workdir* and checked against each pattern.

    Returns ``None`` if no pattern matches.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    resolved_workdir = workdir.resolve()

    for token in tokens:
        try:
            resolved = (workdir / token).resolve()
        except Exception:
            continue

        try:
            rel = str(resolved.relative_to(resolved_workdir)).replace("\\", "/")
        except ValueError:
            return f"Access denied: {resolved}"

        for pattern in patterns:
            if any(c in pattern for c in ("*", "?", "[")):
                if fnmatch.fnmatch(rel, pattern):
                    return f"Access denied: {resolved}"
                if pattern.startswith("**/") and fnmatch.fnmatch(resolved.name, pattern[3:]):
                    return f"Access denied: {resolved}"
                if fnmatch.fnmatch(resolved.name, pattern):
                    return f"Access denied: {resolved}"
            else:
                if resolved.name == pattern or rel == pattern or rel.startswith(pattern + "/"):
                    return f"Access denied: {resolved}"

    return None


@runtime_checkable
class ShellEnvironment(Protocol):
    @property
    def workdir(self) -> Path: ...

    def run(self, command: str) -> str: ...
