# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for human pairwise comparators (offline export/import + inline)."""

import json

import pytest

from ag2.eval import InMemoryTraceSource, TraceRef, evaluate_pairwise
from ag2.eval.scorers import export_pairwise_cases, human_labels, human_pairwise
from ag2.eval.trace import Trace
from ag2.events import ModelMessage, ModelResponse


def _trace(answer: str) -> Trace:
    return Trace(events=[ModelResponse(message=ModelMessage(answer))], exception=None, duration_ms=0)


def _src(label: str, task_ids) -> InMemoryTraceSource:
    return InMemoryTraceSource([(TraceRef(f"{label}-{t}", task_id=t), _trace(label)) for t in task_ids])


@pytest.mark.asyncio()
async def test_offline_export_label_import_roundtrip(tmp_path) -> None:
    src_a, src_b = _src("A", ("t1", "t2")), _src("B", ("t1", "t2"))
    manifest = tmp_path / "cases.jsonl"
    await export_pairwise_cases(src_a, src_b, criteria=["correctness"], out=str(manifest), seed=0)

    # simulate a labeler who always prefers variant A (de-blind via first_variant)
    labeled = tmp_path / "labeled.jsonl"
    out_lines = []
    for raw in manifest.read_text().splitlines():
        rec = json.loads(raw)
        rec["preferred"] = "1" if rec["first_variant"] == "a" else "2"  # the slot holding A
        out_lines.append(json.dumps(rec))
    labeled.write_text("\n".join(out_lines) + "\n")

    comp = human_labels(str(labeled), criterion="correctness", key="correctness@human")
    result = await evaluate_pairwise(src_a, src_b, comparators=[comp], variant_a="A", variant_b="B", store_dir=tmp_path)

    assert result.tally("correctness@human") == (2, 0, 0)  # A wins both


@pytest.mark.asyncio()
async def test_inline_human_prefers_response_one(tmp_path) -> None:
    src_a, src_b = _src("A", ("t1",)), _src("B", ("t1",))
    seen: dict[str, str] = {}

    def ask(task, response_1, response_2):  # always prefer Response 1
        seen["r1"], seen["r2"] = response_1, response_2
        return "1"

    comp = human_pairwise(key="h", ask=ask, seed=0)
    result = await evaluate_pairwise(src_a, src_b, comparators=[comp], store_dir=tmp_path)

    a, b, t = result.tally("h")
    assert (a + b, t) == (1, 0)  # one decisive winner (which variant depends on the blinded order), no tie
    assert {seen["r1"], seen["r2"]} == {"A", "B"}


@pytest.mark.asyncio()
async def test_human_labels_missing_label_is_tie(tmp_path) -> None:
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps({"task_id": "t1", "criterion": "c", "first_variant": "a"}) + "\n")
    comp = human_labels(str(manifest), criterion="c", key="c")

    result = await evaluate_pairwise(_src("A", ("t1",)), _src("B", ("t1",)), comparators=[comp], store_dir=tmp_path)
    assert result.tally("c") == (0, 0, 1)  # no 'preferred' -> tie
