# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import json
import sqlite3
import uuid
from collections.abc import Generator
from typing import Any, Optional

import pytest
from _pytest.mark import ParameterSet

import autogen
import autogen.runtime_logging

from ..conftest import Credentials, credentials_all_llms, suppress_gemini_resource_exhausted

TEACHER_MESSAGE = """
    You are roleplaying a math teacher, and your job is to help your students with linear algebra.
    Keep your explanations short.
"""

STUDENT_MESSAGE = """
    You are roleplaying a high school student struggling with linear algebra.
    Regardless how well the teacher explains things to you, you just don't quite get it.
    Keep your questions short.
"""

CHAT_COMPLETIONS_QUERY = """SELECT id, invocation_id, client_id, wrapper_id, session_id,
    request, response, is_cached, cost, start_time, end_time FROM chat_completions;"""

AGENTS_QUERY = "SELECT id, agent_id, wrapper_id, session_id, name, class, init_args, timestamp FROM agents"

OAI_CLIENTS_QUERY = "SELECT id, client_id, wrapper_id, session_id, class, init_args, timestamp FROM oai_clients"

OAI_WRAPPERS_QUERY = "SELECT id, wrapper_id, session_id, init_args, timestamp FROM oai_wrappers"

EVENTS_QUERY = (
    "SELECT source_id, source_name, event_name, agent_module, agent_class_name, json_state, timestamp FROM events"
)


@pytest.fixture(scope="function")
def db_connection() -> Generator[Optional[sqlite3.Connection], Any, None]:
    autogen.runtime_logging.start(config={"dbname": ":memory:"})
    con = autogen.runtime_logging.get_connection()
    con.row_factory = sqlite3.Row
    yield con

    autogen.runtime_logging.stop()


def _test_two_agents_logging(
    credentials: Credentials, db_connection: Generator[Optional[sqlite3.Connection], Any, None], row_classes: list[str]
) -> None:
    cur = db_connection.cursor()

    teacher = autogen.AssistantAgent(
        "teacher",
        system_message=TEACHER_MESSAGE,
        is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
        llm_config=credentials.llm_config,
        max_consecutive_auto_reply=2,
    )

    student = autogen.AssistantAgent(
        "student",
        system_message=STUDENT_MESSAGE,
        is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
        llm_config=credentials.llm_config,
        max_consecutive_auto_reply=1,
    )

    student.initiate_chat(
        teacher,
        message="Can you explain the difference between eigenvalues and singular values again?",
    )

    # Verify log completions table
    cur.execute(CHAT_COMPLETIONS_QUERY)
    rows = cur.fetchall()

    assert len(rows) >= 3  # some config may fail
    session_id = rows[0]["session_id"]

    for idx, row in enumerate(rows):
        assert row["invocation_id"] and str(uuid.UUID(row["invocation_id"], version=4)) == row["invocation_id"], (
            "invocation id is not valid uuid"
        )
        assert row["client_id"], "client id is empty"
        assert row["wrapper_id"], "wrapper id is empty"
        assert row["session_id"] and row["session_id"] == session_id

        request = json.loads(row["request"])
        first_request_message = request["messages"][0]["content"]
        first_request_role = request["messages"][0]["role"]

        # some config may fail
        if idx == 0 or idx == len(rows) - 1:
            assert first_request_message == TEACHER_MESSAGE
        elif idx == 1 and len(rows) == 3:
            assert first_request_message == STUDENT_MESSAGE
        else:
            assert first_request_message in (TEACHER_MESSAGE, STUDENT_MESSAGE)
        assert first_request_role == "system"

        response = json.loads(row["response"])

        if "response" in response:  # config failed or response was empty
            assert response["response"] is None or "error_code" in response["response"]
        else:
            assert "choices" in response and len(response["choices"]) > 0

        assert row["cost"] >= 0.0
        assert row["start_time"], "start timestamp is empty"
        assert row["end_time"], "end timestamp is empty"

    # Verify agents table
    cur.execute(AGENTS_QUERY)
    rows = cur.fetchall()

    assert len(rows) == 2

    session_id = rows[0]["session_id"]
    for idx, row in enumerate(rows):
        assert row["wrapper_id"], "wrapper id is empty"
        assert row["session_id"] and row["session_id"] == session_id

        agent = json.loads(row["init_args"])
        if idx == 0:
            assert row["name"] == "teacher"
            assert agent["name"] == "teacher"
            agent["system_message"] == TEACHER_MESSAGE
        elif idx == 1:
            assert row["name"] == "student"
            assert agent["name"] == "student"
            agent["system_message"] = STUDENT_MESSAGE

        assert "api_key" not in row["init_args"]
        assert row["timestamp"], "timestamp is empty"

    # Verify oai client table
    cur.execute(OAI_CLIENTS_QUERY)
    rows = cur.fetchall()

    assert len(rows) == len(credentials.config_list) * 2  # two agents

    session_id = rows[0]["session_id"]
    for row in rows:
        assert row["client_id"], "client id is empty"
        assert row["wrapper_id"], "wrapper id is empty"
        assert row["session_id"] and row["session_id"] == session_id
        assert row["class"] in row_classes
        init_args = json.loads(row["init_args"])
        if row["class"] == "AzureOpenAI":
            assert "api_version" in init_args
        assert row["timestamp"], "timestamp is empty"

    # Verify oai wrapper table
    cur.execute(OAI_WRAPPERS_QUERY)
    rows = cur.fetchall()

    session_id = rows[0]["session_id"]

    for row in rows:
        assert row["wrapper_id"], "wrapper id is empty"
        assert row["session_id"] and row["session_id"] == session_id
        init_args = json.loads(row["init_args"])
        assert "config_list" in init_args
        assert len(init_args["config_list"]) > 0
        assert row["timestamp"], "timestamp is empty"


@pytest.mark.parametrize("credentials_fixture", credentials_all_llms)
@suppress_gemini_resource_exhausted
def test_two_agents_logging(
    credentials_fixture: ParameterSet,
    request: pytest.FixtureRequest,
    db_connection: Generator[Optional[sqlite3.Connection], Any, None],
) -> None:
    credentials = request.getfixturevalue(credentials_fixture)
    # Determine the client classes based on the markers applied to the current test
    applied_markers = [mark.name for mark in request.node.iter_markers()]
    if "gemini" in applied_markers:
        row_classes = ["GeminiClient"]
    elif "anthropic" in applied_markers:
        row_classes = ["AnthropicClient"]
    elif "openai" in applied_markers:
        row_classes = ["AzureOpenAI", "OpenAI"]
    else:
        raise ValueError("Unknown client class")

    _test_two_agents_logging(credentials, db_connection, row_classes)


def _test_groupchat_logging(credentials: Credentials, credentials2: Credentials, db_connection):
    cur = db_connection.cursor()

    teacher = autogen.AssistantAgent(
        "teacher",
        system_message=TEACHER_MESSAGE,
        is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
        llm_config=credentials.llm_config,
        max_consecutive_auto_reply=2,
    )

    student = autogen.AssistantAgent(
        "student",
        system_message=STUDENT_MESSAGE,
        is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
        llm_config=credentials2.llm_config,
        max_consecutive_auto_reply=1,
    )

    groupchat = autogen.GroupChat(
        agents=[teacher, student], messages=[], max_round=3, speaker_selection_method="round_robin"
    )

    group_chat_manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=credentials2.llm_config)

    student.initiate_chat(
        group_chat_manager,
        message="Can you explain the difference between eigenvalues and singular values again?",
    )

    # Verify chat_completions message
    cur.execute(CHAT_COMPLETIONS_QUERY)
    rows = cur.fetchall()
    assert len(rows) >= 2  # some config may fail

    # Verify group chat manager agent
    cur.execute(AGENTS_QUERY)
    rows = cur.fetchall()
    assert len(rows) == 3

    chat_manager_query = "SELECT agent_id, name, class, init_args FROM agents WHERE name = 'chat_manager'"
    cur.execute(chat_manager_query)
    rows = cur.fetchall()
    assert len(rows) == 1

    # Verify oai clients
    cur.execute(OAI_CLIENTS_QUERY)
    rows = cur.fetchall()
    assert len(rows) == len(credentials2.config_list) * 2 + len(credentials.config_list)  # two agents and chat manager

    # Verify oai wrappers
    cur.execute(OAI_WRAPPERS_QUERY)
    rows = cur.fetchall()
    assert len(rows) == 3

    # Verify events
    cur.execute(EVENTS_QUERY)
    rows = cur.fetchall()
    json_state = json.loads(rows[0]["json_state"])
    assert rows[0]["event_name"] == "received_message"
    assert json_state["message"] == "Can you explain the difference between eigenvalues and singular values again?"
    assert len(rows) == 15

    # Verify schema version
    version_query = "SELECT id, version_number from version"
    cur.execute(version_query)
    rows = cur.fetchall()
    assert len(rows) == 1
    assert rows[0]["id"] == 1 and rows[0]["version_number"] == 1


@pytest.mark.parametrize("credentials_from_test_param", credentials_all_llms, indirect=True)
@suppress_gemini_resource_exhausted
def test_groupchat_logging(
    credentials_from_test_param: Credentials,
    db_connection: Generator[Optional[sqlite3.Connection], Any, None],
) -> None:
    _test_groupchat_logging(credentials_from_test_param, credentials_from_test_param, db_connection)
