# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Command-level filters used by :class:`ShellAdapter`.

The adapter implements the ``allowed`` / ``blocked`` / ``ignore`` / ``readonly``
shell surface on top of any :class:`~ag2.tools.sandbox.base.Sandbox`.
These helpers live in the sandbox package so the adapter can import them without
triggering ``shell`` package initialisation (which would re-enter sandbox and
deadlock).
"""

import fnmatch
import posixpath
import shlex
from pathlib import Path, PurePath, PurePosixPath

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


# Shell metacharacters that let a single allowed head-command spawn or
# redirect to other commands, bypassing the allow-list. Blocked while a
# restricted ``allowed`` set is active (see :meth:`ShellAdapter._filter`).
_SHELL_OPERATORS: tuple[str, ...] = (">", ">>", "|", ";", "&&", "||", "`", "$(")


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


def contains_shell_operator(command: str) -> bool:
    """Return True if *command* contains shell operators that could bypass
    the allowed-command whitelist (redirection, pipes, chaining, backtick
    or ``$(...)`` command substitution).
    """
    return any(op in command for op in _SHELL_OPERATORS)


def check_ignore(command: str, workdir: "Path | PurePath", patterns: list[str]) -> str | None:
    """Return ``"Access denied: <path>"`` if any literal path in *command* matches *patterns*.

    Tokens are extracted via :func:`shlex.split` to handle quoted paths. Each
    token is resolved relative to *workdir* and checked against each pattern.
    Returns ``None`` if no pattern matches.

    *workdir* may be a host :class:`~pathlib.Path` (local backend) or a
    :class:`~pathlib.PurePosixPath` (remote/container backend). For a host path
    tokens are resolved against the real filesystem (symlinks included); for a
    pure path they are normalised lexically (``posixpath.normpath``) so the
    filter works on remote backends without touching the host filesystem.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    host_backed = isinstance(workdir, Path)
    if host_backed:
        resolved_workdir: PurePath = workdir.resolve()
    else:
        resolved_workdir = PurePosixPath(posixpath.normpath(str(workdir)))

    for token in tokens:
        if host_backed:
            try:
                resolved: PurePath = (workdir / token).resolve()
            except Exception:
                continue
        else:
            resolved = PurePosixPath(posixpath.normpath(posixpath.join(str(workdir), token)))

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
