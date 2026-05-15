# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Regression: PythonEnvironment._write_to_file must pin UTF-8.

Agent-generated scripts routinely embed non-ASCII content — string literals
in CJK / emoji / smart quotes from copy-pasted documentation, identifiers
under PEP 3131. When `open(path, "w")` runs on a host whose
`locale.getpreferredencoding(False)` is not UTF-8 (e.g. cp1252 on a default
Windows install), the first non-cp1252 character raises
`UnicodeEncodeError` mid-write. The script is then partially written, the
subprocess runs an invalid file, and the agent sees a syntax error or
truncated output instead of the intended behavior.

Pinning `encoding="utf-8"` on the open call removes the OS-locale
dependency entirely. This source-level guard prevents the kwarg from
silently regressing.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_python_environment_write_to_file_pins_utf8() -> None:
    source = (REPO_ROOT / "autogen" / "environments" / "python_environment.py").read_text(encoding="utf-8")
    assert 'open(script_path, "w", encoding="utf-8") as f' in source, (
        "PythonEnvironment._write_to_file must pin encoding='utf-8' on its "
        "open() call so non-cp1252 script content (CJK string literals, emoji, "
        "smart quotes, PEP 3131 identifiers) does not crash on Windows."
    )
