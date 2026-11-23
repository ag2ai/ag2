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
import os
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


def check_ignore(
    command: str,
    workdir: "Path | PurePath",
    patterns: list[str],
    *,
    max_entries: int = 20_000,
) -> str | None:
    """Return ``"Access denied: <path>"`` if any literal path in *command* leaves *workdir* or matches *patterns*.

    Tokens are extracted via :func:`shlex.split` to handle quoted paths. Each
    token is resolved relative to *workdir* and checked against each pattern;
    a glob token is checked against the files it matches on a host path.
    Returns ``None`` if no pattern matches.

    For ``grep`` / ``diff`` every file under a directory operand is checked as
    well. Anything the filter cannot inspect is refused: a path that does not
    resolve, a directory that cannot be listed, a ``**`` wildcard, or more than
    *max_entries* wildcard matches and walked entries in total.

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

    expanded_tokens = [candidate for token in tokens for candidate in _embedded_paths(token)]

    if not isinstance(workdir, Path):
        resolved_workdir = PurePosixPath(posixpath.normpath(str(workdir)))
        for token in expanded_tokens:
            resolved = PurePosixPath(posixpath.normpath(posixpath.join(str(workdir), token)))
            denied = _denied(resolved, resolved_workdir, patterns)
            if denied is not None:
                return denied
        if tokens and PurePosixPath(tokens[0]).name in _DIRECTORY_READERS:
            return (
                f"Access denied: {tokens[0]} can read files inside a directory, which cannot be checked on this backend"
            )
        return None

    try:
        return _check_host_paths(tokens, expanded_tokens, workdir, patterns, _Budget(max_entries))
    except _UncheckableError as exc:
        return f"Access denied: {exc}"


class _UncheckableError(Exception):
    """A path the filter cannot inspect; the command is refused rather than let through."""


class _Budget:
    """Caps the filesystem entries one command may make the filter look at (wildcard matches and walked entries)."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def spend(self) -> None:
        self.used += 1
        if self.used > self.limit:
            raise _UncheckableError(f"more than {self.limit} files to check against ignore; name a smaller directory")


def _check_host_paths(
    tokens: list[str],
    expanded_tokens: list[str],
    workdir: Path,
    patterns: list[str],
    budget: _Budget,
) -> str | None:
    resolved_workdir = _resolve(workdir)
    # Each word with the paths its wildcard matches, relative to workdir; expanded once and reused below.
    words = [
        (token, [PurePath(token), *_glob_matches(workdir, resolved_workdir, token, budget)])
        for token in expanded_tokens
    ]

    for _, paths in words:
        for path in paths:
            denied = _denied(_resolve(workdir / path), resolved_workdir, patterns)
            if denied is not None:
                return denied

    if not tokens or PurePosixPath(tokens[0]).name not in _DIRECTORY_READERS:
        return None
    return _check_directory_operands(tokens, words[1:], workdir, resolved_workdir, patterns, budget)


def _resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise _UncheckableError(f"cannot resolve {path!r}") from exc


# Programs in READONLY_COMMANDS that open files inside a directory operand on
# their own: ``grep -r``, ``diff -r``, and ``diff -N dir1 dir2`` even without
# recursion. Matching a directory word against the patterns is not enough, so
# every file under such an operand is checked.
_DIRECTORY_READERS = frozenset({"grep", "egrep", "fgrep", "diff"})
_GREP_FAMILY = frozenset({"grep", "egrep", "fgrep"})

# grep options whose value may be the next word. Treating an option as taking a
# value when it does not only makes the check stricter, so long names match by
# prefix (getopt accepts unambiguous abbreviations).
_GREP_SHORT_WITH_VALUE = "efmABCdD"
_GREP_LONG_WITH_VALUE = (
    "regexp",
    "file",
    "max-count",
    "after-context",
    "before-context",
    "context",
    "directories",
    "devices",
    "label",
    "include",
    "exclude",
    "exclude-from",
    "exclude-dir",
    "binary-files",
    "group-separator",
)


def _check_directory_operands(
    tokens: list[str],
    words: list[tuple[str, list[PurePath]]],
    workdir: Path,
    resolved_workdir: Path,
    patterns: list[str],
    budget: _Budget,
) -> str | None:
    # Each directory is walked under the path it was named by (an alias such as
    # a symlink may carry its own ignore rule) as well as its real location.
    starts: list[tuple[PurePath, Path]] = [
        (PurePath(os.path.normpath(resolved_workdir / path)), _resolve(workdir / path))
        for _, paths in words
        for path in paths
        if _is_dir(workdir / path)
    ]

    if PurePosixPath(tokens[0]).name in _GREP_FAMILY and not any(
        _exists(workdir / operand) for operand in _grep_file_operands(tokens[1:])
    ):
        # With no file operand a recursive grep searches the working directory.
        starts.append((resolved_workdir, resolved_workdir))

    return _ignored_inside(starts, resolved_workdir, patterns, budget)


def _grep_file_operands(args: list[str]) -> list[str]:
    """Return the words grep takes as files (best effort, erring towards none).

    The first positional word is the pattern unless ``-e`` / ``-f`` gives one.
    When this guesses wrong it returns fewer operands, which makes the caller
    check the whole working directory.
    """
    positionals: list[str] = []
    explicit_pattern = False
    options_ended = False
    i = 0
    while i < len(args):
        arg = args[i]
        i += 1
        if options_ended or arg == "-" or not arg.startswith("-"):
            positionals.append(arg)
        elif arg == "--":
            options_ended = True
        elif arg.startswith("--"):
            name, eq, _ = arg[2:].partition("=")
            explicit_pattern = explicit_pattern or name in ("regexp", "file")
            if not eq and any(option.startswith(name) for option in _GREP_LONG_WITH_VALUE):
                i += 1
        else:
            for pos, letter in enumerate(arg[1:], start=1):
                explicit_pattern = explicit_pattern or letter in "ef"
                if letter in _GREP_SHORT_WITH_VALUE:
                    if pos == len(arg) - 1:
                        i += 1
                    break
    return positionals if explicit_pattern else positionals[1:]


def _ignored_inside(
    starts: list[tuple[PurePath, Path]],
    resolved_workdir: Path,
    patterns: list[str],
    budget: _Budget,
) -> str | None:
    """Return ``"Access denied: <path>"`` for the first file under the *starts* directories that may not be read.

    Every entry is checked both by the path it is reached under (keeping the
    alias a directory was entered by) and by its real target, and symlinks are
    followed as ``grep -R`` and ``diff -r`` follow them. Entries are read one at
    a time and each is charged to *budget*, so a large tree is refused without
    being listed in full. A directory that cannot be listed or resolved raises
    :class:`_UncheckableError`. A symlink back into one of its own ancestors is not
    entered again (``grep`` and ``diff`` skip such loops too).
    """
    pending = [(lexical, real, frozenset({real})) for lexical, real in starts]
    while pending:
        lexical, real, ancestors = pending.pop()
        try:
            with os.scandir(real) as entries:
                for entry in entries:
                    budget.spend()
                    child = lexical / entry.name
                    target = _resolve(real / entry.name)
                    for candidate in (child, target):
                        denied = _denied(candidate, resolved_workdir, patterns)
                        if denied is not None:
                            return denied
                    if entry.is_dir() and target not in ancestors:
                        pending.append((child, target, ancestors | {target}))
        except OSError as exc:
            raise _UncheckableError(f"cannot list {lexical}") from exc
    return None


def _is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _embedded_paths(token: str) -> list[str]:
    """Return *token* plus the paths a program may read out of it.

    A word can carry a path inside it: after ``=`` (``of=.env``,
    ``--output=.env``) or glued to a short option (``-o.env``, ``-ro.env``).
    Each tail is checked as well as the whole word, so the filter errs towards
    refusing (``-ofoo.env`` is refused when ``.env`` is ignored).
    """
    found = [token]
    if "=" in token:
        found.extend(part for part in token.split("=") if part)
    if token.startswith("-") and not token.startswith("--"):
        found.extend(token[i:] for i in range(2, len(token)))
    return found


def _glob_matches(workdir: Path, resolved_workdir: Path, token: str, budget: _Budget) -> list[PurePath]:
    """Return the paths, relative to *workdir*, that a wildcard *token* matches.

    The command runs as argv, but a program may expand a wildcard itself (MSYS
    tools do on Windows), so a pattern naming an ignored file, or a directory
    holding one, must be refused. The pattern is expanded one path segment at a
    time and every directory entry read on the way, matching or not, is charged
    to *budget*. A pattern that cannot be expanded here raises
    :class:`_UncheckableError`: ``**``, whose scan the budget cannot bound, and a
    pattern crossing a directory that cannot be listed.
    """
    if not any(c in token for c in "*?["):
        return []
    pattern = PurePath(token)
    if pattern.is_absolute():
        base = next((b for b in (workdir, resolved_workdir) if pattern.is_relative_to(b)), None)
        if base is None:
            return []  # outside the workdir: the word itself is refused
        pattern = pattern.relative_to(base)
    if "**" in token:
        raise _UncheckableError(f"cannot check recursive wildcard {token!r}")
    found = [PurePath()]
    for part in pattern.parts:
        if any(c in part for c in "*?["):
            found = [match for prefix in found for match in _matching_entries(workdir, prefix, part, budget)]
        else:
            found = [prefix / part for prefix in found]
    return found


def _matching_entries(workdir: Path, prefix: PurePath, part: str, budget: _Budget) -> list[PurePath]:
    """Return ``prefix / name`` for each entry of the *prefix* directory whose name matches the wildcard *part*."""
    matched: list[PurePath] = []
    try:
        with os.scandir(workdir / prefix) as entries:
            for entry in entries:
                budget.spend()
                if fnmatch.fnmatch(entry.name, part):
                    matched.append(prefix / entry.name)
    except (FileNotFoundError, NotADirectoryError):
        return []  # nothing there for the wildcard to match
    except (OSError, ValueError) as exc:
        raise _UncheckableError(f"cannot list {workdir / prefix}") from exc
    return matched


def _denied(resolved: PurePath, resolved_workdir: PurePath, patterns: list[str]) -> str | None:
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
        elif resolved.name == pattern or rel == pattern or rel.startswith(pattern + "/"):
            return f"Access denied: {resolved}"

    return None
