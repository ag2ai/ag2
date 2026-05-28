# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Command-level filters shared by :class:`ShellAdapter` and the legacy
:class:`ShellEnvironment` aliases.

These helpers used to live on :mod:`autogen.beta.tools.shell.environment.base`,
but the adapter that needs them sits inside the sandbox package — keeping them
here lets the adapter import them without triggering ``shell`` package
initialisation (which would re-enter sandbox and deadlock).
"""

import fnmatch
import shlex
from pathlib import Path

# Commands that only read state and never modify the filesystem.
# Used when ``ShellAdapter.readonly=True`` and no explicit ``allowed``
# list is provided. Best-effort: ``echo`` can still redirect output
# (``echo x > file``) because shell processing happens in the OS shell
# after our prefix check.
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
