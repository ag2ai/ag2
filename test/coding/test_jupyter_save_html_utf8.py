# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_embedded_save_html_pins_utf8_encoding_in_source() -> None:
    source = (REPO_ROOT / "autogen" / "coding" / "jupyter" / "embedded_ipython_code_executor.py").read_text(
        encoding="utf-8"
    )
    assert 'open(path, "w", encoding="utf-8") as f' in source, (
        "EmbeddedIPythonCodeExecutor._save_html must pin encoding='utf-8' on its "
        "open() call so non-cp1252 cell output (emoji, CJK, smart quotes) does "
        "not crash on Windows."
    )


def test_jupyter_save_html_pins_utf8_encoding_in_source() -> None:
    source = (REPO_ROOT / "autogen" / "coding" / "jupyter" / "jupyter_code_executor.py").read_text(encoding="utf-8")
    assert 'open(path, "w", encoding="utf-8") as f' in source, (
        "JupyterCodeExecutor._save_html must pin encoding='utf-8' on its "
        "open() call so non-cp1252 cell output (emoji, CJK, smart quotes) does "
        "not crash on Windows."
    )
