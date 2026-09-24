# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Type-level guarantees are tested through the checker's own output.

Each fixture under ``fixtures/`` carries its expectations inline — ``# N: <note>`` and ``# E: <error>``
on the line they belong to — so a form the user guide teaches and the type it is
expected to have are read together. Hooking the DSL plugin's internals instead
would be testing mypy's API rather than ours.

mypy runs as a subprocess, under the repository's own ``[tool.mypy]`` settings,
because that is the configuration the guarantee is about.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mypy")

REPO_ROOT = Path(__file__).parents[2]
FIXTURES = sorted(p for p in (Path(__file__).parent / "fixtures").glob("*.py") if p.name != "__init__.py")

_EXPECTATION = re.compile(r"#\s(?P<kind>[NE]):\s(?P<message>.+?)\s*$")
_REPORTED = re.compile(r"^(?P<path>.+?):(?P<line>\d+): (?P<kind>note|error): (?P<message>.+?)\s*$")

_KINDS = {"N": "note", "E": "error"}


def _expected(fixture: Path) -> set[tuple[int, str, str]]:
    out = set()
    for lineno, line in enumerate(fixture.read_text().splitlines(), start=1):
        match = _EXPECTATION.search(line)
        if match:
            out.add((lineno, _KINDS[match["kind"]], match["message"]))
    return out


def _reported(fixture: Path) -> set[tuple[int, str, str]]:
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--no-error-summary", str(fixture)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # ``install_types`` makes mypy keep stdout for its own prompts and report
    # diagnostics on stderr, so both streams are read.
    out = set()
    for line in (result.stdout + result.stderr).splitlines():
        match = _REPORTED.match(line)
        assert match, f"unparsable mypy output: {line!r}"
        out.add((int(match["line"]), match["kind"], match["message"]))
    return out


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.stem)
def test_checker_agrees_with_the_fixture(fixture: Path) -> None:
    """Every form a fixture expects to check does, and every form it expects rejected is."""
    assert _reported(fixture) == _expected(fixture)
