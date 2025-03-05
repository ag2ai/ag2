# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# !/usr/bin/env python3 -m pytest

import json

import pytest

from autogen.agentchat.contrib.agent_eval.agent_eval import generate_criteria, quantify_criteria
from autogen.agentchat.contrib.agent_eval.criterion import Criterion
from autogen.agentchat.contrib.agent_eval.task import Task
from autogen.import_utils import run_for_optional_imports

from ....conftest import Credentials


def remove_ground_truth(test_case: str):
    test_details = json.loads(test_case)
    # need to remove the ground truth from the test details
    correctness = test_details.pop("is_correct", None)
    test_details.pop("correct_ans", None)
    test_details.pop("check_result", None)
    return str(test_details), correctness


@pytest.fixture
def task() -> Task:
    success_str = open("test/test_files/agenteval-in-out/samples/sample_math_response_successful.txt").read()  # noqa: SIM115
    response_successful = remove_ground_truth(success_str)[0]
    failed_str = open("test/test_files/agenteval-in-out/samples/sample_math_response_failed.txt").read()  # noqa: SIM115
    response_failed = remove_ground_truth(failed_str)[0]
    task = Task(**{
        "name": "Math problem solving",
        "description": "Given any question, the system needs to solve the problem as consisely and accurately as possible",
        "successful_response": response_successful,
        "failed_response": response_failed,
    })
    return task


@run_for_optional_imports("openai", "openai")
def test_generate_criteria(credentials_azure: Credentials, task: Task):
    criteria = generate_criteria(task=task, llm_config={"config_list": credentials_azure.config_list})
    assert criteria
    assert len(criteria) > 0
    assert criteria[0].description
    assert criteria[0].name
    assert criteria[0].accepted_values


@run_for_optional_imports("openai", "openai")
@run_for_optional_imports("openai", "openai")
def test_quantify_criteria(credentials_azure: Credentials, task: Task):
    criteria_file = "test/test_files/agenteval-in-out/samples/sample_math_criteria.json"
    criteria = open(criteria_file).read()  # noqa: SIM115
    criteria = Criterion.parse_json_str(criteria)

    test_case = open("test/test_files/agenteval-in-out/samples/sample_test_case.json").read()  # noqa: SIM115
    test_case, ground_truth = remove_ground_truth(test_case)

    quantified = quantify_criteria(
        llm_config={"config_list": credentials_azure.config_list},
        criteria=criteria,
        task=task,
        test_case=test_case,
        ground_truth=ground_truth,
    )
    assert quantified
    assert quantified["actual_success"]
    assert quantified["estimated_performance"]
