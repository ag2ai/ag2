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
from collections.abc import Sequence
from pathlib import Path, PurePath, PurePosixPath

# What ``readonly=True`` allows when no ``allowed`` list is given: commands with
# no option to write a file or run another program, so the guarantee can be
# checked by reading this list. ``find`` (``-exec``), ``file`` (``-C``) and
# ``git`` (``--output``, programs named in ``.git/config``) are left out on
# purpose; a user who needs them lists them in ``allowed``.
READONLY_COMMANDS: tuple[str, ...] = (
    "cat",
    "head",
    "tail",
    "ls",
    "grep",
    "egrep",
    "fgrep",
    "wc",
    "du",
    "df",
    "diff",
    "stat",
    "which",
    "pwd",
    "echo",
    "printenv",
    "cut",
)


# Shell syntax a model may try in restricted mode. Restricted mode runs the
# parsed argv without a shell, so none of it takes effect; rejecting it up
# front gives the model a clear error instead of a confusing one from the
# command (``cat: '>': No such file``).
_SHELL_OPERATORS: tuple[str, ...] = (">", ">>", "<", "|", ";", "&", "&&", "||", "\n", "\r", "`", "$(")


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


def matches_argv(pattern: str, argv: Sequence[str]) -> bool:
    """Return True if ``argv`` starts with the words of ``pattern``.

    ``"git log"`` matches ``["git", "log", "-5"]`` but not ``["git", "-c", "x", "log"]``.
    """
    words = pattern.split()
    return list(argv[: len(words)]) == words


def split_command(command: str) -> list[str] | None:
    """Split ``command`` into argv with POSIX shell quoting, or return None if the quotes do not balance."""
    try:
        return shlex.split(command)
    except ValueError:
        return None


def contains_shell_operator(command: str) -> bool:
    """Return True if *command* contains any of ``_SHELL_OPERATORS`` outside quotes and escapes."""
    return any(op in _unquoted(command) for op in _SHELL_OPERATORS)


def _unquoted(command: str) -> str:
    """Return *command* with quoted and backslash-escaped text removed."""
    kept: list[str] = []
    quote = ""
    chars = iter(command)
    for char in chars:
        if quote:
            if char == quote:
                quote = ""
            elif char == "\\" and quote == '"':
                next(chars, "")
        elif char in "'\"":
            quote = char
        elif char == "\\":
            next(chars, "")
        else:
            kept.append(char)
    return "".join(kept)


def check_ignore(command: str, workdir: "Path | PurePath", patterns: list[str]) -> str | None:
    """Deny wildcard arguments and literal paths that leave *workdir* or match *patterns*.

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
        # Windows runtimes can expand wildcards even without a shell. Check
        # parsed tokens so quotes/escapes cannot hide a pattern from the filter.
        if any(char in token for char in "*?[{"):
            return f"Access denied (wildcard arguments are not supported with ignore): {token!r}"
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
