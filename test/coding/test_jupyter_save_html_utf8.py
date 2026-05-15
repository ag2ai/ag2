# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Regression: ``_save_html`` on the jupyter code executors must pin UTF-8.

The default ``open(path, "w")`` honors ``locale.getpreferredencoding(False)``,
which on Windows commonly resolves to ``cp1252``. Jupyter cells regularly
return HTML rendered output containing characters outside the cp1252 range
(emoji, CJK, mathematical symbols), and writing them with the platform default
raises ``UnicodeEncodeError: 'charmap' codec can't encode character ...``,
turning a successful cell into a code-execution failure for the agent.

These tests pin the encoding kwarg on the concrete ``open`` call sites so the
behavior is portable across OS locale defaults. They use ``Path.read_text`` on
the source file rather than importing the module, so the regression check runs
on every CI lane regardless of whether the optional ``jupyter-executor`` extras
are installed.
"""

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
