# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Regression: MCP proxy file writes must pin UTF-8.

`autogen/mcp/mcp_proxy/mcp_proxy.py` rewrites the generated `main.py` (post-
fastapi-codegen patch) and saves the rendered server configuration template.
Both call sites previously used the bare locale default — on Windows that
resolves to `cp1252` and any non-cp1252 glyph in an OpenAPI spec
(internationalized model names, smart quotes in descriptions, emoji in
example payloads) raised `UnicodeEncodeError` mid-write, killing the proxy
generation step.

This source-level check guards the kwarg so the bug cannot silently regress.
Runs without optional MCP extras installed.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_mcp_proxy_file_writes_pin_utf8() -> None:
    source = (REPO_ROOT / "autogen" / "mcp" / "mcp_proxy" / "mcp_proxy.py").read_text(encoding="utf-8")
    must_contain = (
        'main_path.open("r", encoding="utf-8") as f',
        'main_path.open("w", encoding="utf-8") as f',
        'open(output_file, "w", encoding="utf-8") as f',
        'Path(config_file).open("r", encoding="utf-8") as f',
    )
    missing = [pattern for pattern in must_contain if pattern not in source]
    assert not missing, (
        f"MCP proxy main_path / output_file / config_file IO must pin encoding='utf-8'; missing patterns: {missing!r}"
    )
