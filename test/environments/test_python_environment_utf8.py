# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_python_environment_write_to_file_pins_utf8() -> None:
    source = (REPO_ROOT / "autogen" / "environments" / "python_environment.py").read_text(encoding="utf-8")
    assert 'open(script_path, "w", encoding="utf-8") as f' in source, (
        "PythonEnvironment._write_to_file must pin encoding='utf-8' on its "
        "open() call so non-cp1252 script content (CJK string literals, emoji, "
        "smart quotes, PEP 3131 identifiers) does not crash on Windows."
    )
