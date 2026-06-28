# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Public-API tests for :class:`Suite`, :meth:`Suite.from_list`, and :meth:`Suite.from_jsonl`."""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel

from ag2.eval import Suite, Task


class _RefModel(BaseModel):
    answer: str
    confidence: float = 1.0


@dataclass
class _RefDataclass:
    answer: str


class TestFromList:
    def test_minimal_task(self) -> None:
        suite = Suite.from_list([{"inputs": {"input": "hi"}}])

        assert len(suite) == 1
        [task] = list(suite)
        assert task == Task(task_id="task-0000", inputs={"input": "hi"})

    def test_auto_generated_task_ids_are_zero_padded_and_sequential(self) -> None:
        suite = Suite.from_list([
            {"inputs": {"input": "first"}},
            {"inputs": {"input": "second"}},
            {"inputs": {"input": "third"}},
        ])

        assert tuple(t.task_id for t in suite) == ("task-0000", "task-0001", "task-0002")

    def test_explicit_task_id_is_preserved(self) -> None:
        suite = Suite.from_list([
            {"task_id": "weather-001", "inputs": {"input": "Tokyo?"}},
        ])

        [task] = list(suite)
        assert task.task_id == "weather-001"

    def test_full_payload_round_trips(self) -> None:
        suite = Suite.from_list([
            {
                "task_id": "weather-001",
                "inputs": {"input": "Tokyo?"},
                "reference_outputs": {"city": "Tokyo"},
                "tags": ["happy-path"],
                "metadata": {"difficulty": "easy"},
            }
        ])

        [task] = list(suite)
        assert task == Task(
            task_id="weather-001",
            inputs={"input": "Tokyo?"},
            reference_outputs={"city": "Tokyo"},
            tags=("happy-path",),
            metadata={"difficulty": "easy"},
        )

    def test_missing_inputs_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="missing required field 'inputs'"):
            Suite.from_list([{"task_id": "x"}])

    def test_inputs_not_a_dict_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="must be a dict"):
            Suite.from_list([{"inputs": "what is the weather?"}])

    def test_default_name_and_source_are_inline(self) -> None:
        suite = Suite.from_list([{"inputs": {"input": "hi"}}])

        assert suite.name == "inline"
        assert suite.source == "inline"

    def test_custom_name_overrides_default(self) -> None:
        suite = Suite.from_list([{"inputs": {"input": "hi"}}], name="weather-mini")

        assert suite.name == "weather-mini"
        assert suite.source == "inline"


class TestFromJsonl:
    def test_loads_one_task_per_line(self, tmp_path: Path) -> None:
        path = tmp_path / "dataset.jsonl"
        path.write_text(
            "\n".join([
                json.dumps({"task_id": "a", "inputs": {"input": "1"}}),
                json.dumps({"task_id": "b", "inputs": {"input": "2"}}),
            ]),
            encoding="utf-8",
        )

        suite = Suite.from_jsonl(path)

        assert tuple(t.task_id for t in suite) == ("a", "b")

    def test_name_is_filename_stem_and_source_is_path(self, tmp_path: Path) -> None:
        path = tmp_path / "weather.jsonl"
        path.write_text(json.dumps({"inputs": {"input": "hi"}}) + "\n", encoding="utf-8")

        suite = Suite.from_jsonl(path)

        assert suite.name == "weather"
        assert suite.source == str(path)

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "dataset.jsonl"
        path.write_text(
            "\n".join([
                json.dumps({"inputs": {"input": "1"}}),
                "",
                "   ",
                json.dumps({"inputs": {"input": "2"}}),
            ]),
            encoding="utf-8",
        )

        suite = Suite.from_jsonl(path)

        assert len(suite) == 2

    def test_accepts_str_and_path_inputs(self, tmp_path: Path) -> None:
        path = tmp_path / "dataset.jsonl"
        path.write_text(json.dumps({"inputs": {"input": "hi"}}) + "\n", encoding="utf-8")

        assert len(Suite.from_jsonl(str(path))) == 1
        assert len(Suite.from_jsonl(path)) == 1

    def test_invalid_json_reports_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "dataset.jsonl"
        path.write_text(
            "\n".join([
                json.dumps({"inputs": {"input": "ok"}}),
                "{not json",
            ]),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="line 2"):
            Suite.from_jsonl(path)

    def test_non_object_json_line_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "dataset.jsonl"
        path.write_text("[1, 2, 3]\n", encoding="utf-8")

        with pytest.raises(ValueError, match="must be a JSON object"):
            Suite.from_jsonl(path)


def test_suite_iterates_in_dataset_order() -> None:
    suite = Suite.from_list([{"inputs": {"input": str(i)}} for i in range(3)])

    assert [t.inputs["input"] for t in suite] == ["0", "1", "2"]


def test_suite_tasks_is_a_tuple() -> None:
    suite = Suite.from_list([{"inputs": {"input": "hi"}}])

    assert isinstance(suite.tasks, tuple)
    assert len(suite.tasks) == len(suite)


class TestReferenceOutputsCoercion:
    """reference_outputs is normalised to a plain dict; non-mapping values raise."""

    def test_pydantic_model_is_coerced_to_dict(self) -> None:
        task = Task(task_id="t", inputs={"input": "q"}, reference_outputs=_RefModel(answer="Paris"))

        assert task.reference_outputs == {"answer": "Paris", "confidence": 1.0}

    def test_dataclass_instance_is_coerced_to_dict(self) -> None:
        task = Task(task_id="t", inputs={"input": "q"}, reference_outputs=_RefDataclass(answer="Paris"))

        assert task.reference_outputs == {"answer": "Paris"}

    def test_dict_is_kept(self) -> None:
        task = Task(task_id="t", inputs={"input": "q"}, reference_outputs={"answer": "Paris"})

        assert task.reference_outputs == {"answer": "Paris"}

    def test_none_is_kept(self) -> None:
        assert Task(task_id="t", inputs={"input": "q"}).reference_outputs is None

    def test_non_mapping_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="reference_outputs must be a dict"):
            Task(task_id="t", inputs={"input": "q"}, reference_outputs=["Paris"])

    def test_coercion_applies_through_from_list(self) -> None:
        suite = Suite.from_list([{"inputs": {"input": "q"}, "reference_outputs": _RefModel(answer="Paris")}])

        [task] = list(suite)
        assert task.reference_outputs == {"answer": "Paris", "confidence": 1.0}
