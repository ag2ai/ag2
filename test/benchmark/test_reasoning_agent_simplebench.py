# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import json
import os
import sys

import pytest
from conftest import reason, skip_openai

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from benchmark.simple_bench.run_simple_bench import run_benchmark

DATASET_PATH = "benchmark/simple_bench/simplebench_public.json"


@pytest.fixture
def load_dataset():
    # Ensure the dataset file exists
    if not os.path.exists(DATASET_PATH):
        pytest.fail(f"Dataset file not found at {DATASET_PATH}")

    # Load the dataset
    with open(DATASET_PATH, "r") as f:
        return json.load(f)


@pytest.mark.skipif(
    skip_openai,
    reason=reason,
)
def test_reasoning_agent_simplebench(load_dataset):
    eval_data = load_dataset.get("eval_data")
    assert eval_data is not None, "No eval_list found in dataset"

    run_benchmark(eval_data)

    results_file = next(
        (f for f in os.listdir(".") if f.startswith("results-") and f.endswith(".json")),
        None,
    )
    assert results_file is not None, "Results file not created."

    with open(results_file, "r") as f:
        results = json.load(f)

    correct_results = 0
    for result in results:
        if str(result["answer"]) == str(result["generated_answer"]):
            correct_results += 1

    assert len(results) == len(eval_data), "Results length mismatch."
    print(f"Correct results: {correct_results}/{len(results)}")
    assert correct_results >= len(results) * 0.3, "Less than 30% correct results."
