# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Table-driven validation at the ``A2UIServer`` seam.

Each case in ``validation_corpus.json`` is a model reply for one protocol version
(optionally with a custom catalog) and the outcome a user observes: its messages
streamed as frames, or a retry whose prompt names the expected error.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from .seam import a2ui_reply, serve_turn

_CORPUS = json.loads((Path(__file__).parent / "validation_corpus.json").read_text(encoding="utf-8"))
_CATALOGS: dict[str, dict[str, Any]] = _CORPUS["catalogs"]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _CORPUS["cases"], ids=[case["id"] for case in _CORPUS["cases"]])
async def test_corpus_case(case: dict[str, Any]) -> None:
    catalog = _CATALOGS[case["catalog"]] if "catalog" in case else None

    turn = await serve_turn(
        a2ui_reply(case["messages"]),
        "Plain text.",
        protocol_version=case["version"],
        custom_catalog=catalog,
    )

    if case["expect"] == "frames":
        assert turn.feedback == []
        assert turn.ui == case["messages"]
    else:
        assert len(turn.feedback) == 1
        assert f"- {case['error']}" in turn.feedback[0]
        assert turn.ui == []
