# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import json
import sqlite3
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest

import autogen.runtime_logging
from autogen.logger.file_logger import FileLogger
from autogen.logger.logger_utils import get_current_ts

SENTINEL_API_KEY = "sk-secret-key-12345"
REDACTED = "***REDACTED***"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    req: dict[str, Any] = {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "gpt-4o",
        "api_key": SENTINEL_API_KEY,
    }
    if extra:
        req.update(extra)
    return req


def _dummy_fn(x: int) -> int:
    return x * 2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_connection():
    autogen.runtime_logging.start(config={"dbname": ":memory:"})
    con = autogen.runtime_logging.get_connection()
    con.row_factory = sqlite3.Row
    yield con
    autogen.runtime_logging.stop()


@pytest.fixture()
def file_logger(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    logger = FileLogger(config={"filename": "test.log"})
    logger.start()
    yield logger
    logger.stop()


# ---------------------------------------------------------------------------
# Bug #8: file_logger.log_chat_completion uses to_dict(request) without _redact
# ---------------------------------------------------------------------------


def test_file_logger_chat_completion_redacts_api_key(file_logger, tmp_path):
    """Bug #8: api_key in request must not appear in the log file."""
    request = _make_request()
    file_logger.log_chat_completion(
        invocation_id=uuid.uuid4(),
        client_id=1,
        wrapper_id=2,
        source="TestAgent",
        request=request,
        response="some response",
        is_cached=0,
        cost=0.01,
        start_time=get_current_ts(),
    )

    log_file = tmp_path / "autogen_logs" / "test.log"
    log_content = log_file.read_text()

    assert SENTINEL_API_KEY not in log_content, (
        f"api_key '{SENTINEL_API_KEY}' was written in plaintext to the log file (Bug #8)"
    )
    assert REDACTED in log_content, "Expected '***REDACTED***' marker in log output"


# ---------------------------------------------------------------------------
# Bug #18: sqlite_logger.log_event does not call _redact on kwargs
# ---------------------------------------------------------------------------


def test_sqlite_logger_event_redacts_api_key(db_connection):
    """Bug #18: api_key in event kwargs must be redacted in the events table."""
    autogen.runtime_logging.log_event(
        source="TestAgent",
        name="test_event",
        api_key=SENTINEL_API_KEY,
        other_field="safe_value",
    )

    cur = db_connection.cursor()
    row = cur.execute("SELECT json_state FROM events ORDER BY rowid DESC LIMIT 1").fetchone()
    assert row is not None

    state = json.loads(row["json_state"])
    assert state.get("api_key") != SENTINEL_API_KEY, (
        f"api_key '{SENTINEL_API_KEY}' stored in plaintext in events table (Bug #18)"
    )
    assert state.get("api_key") == REDACTED, "Expected '***REDACTED***' in events.json_state"


# ---------------------------------------------------------------------------
# Bug #19: sqlite_logger.log_function_use does not redact args/returns
# ---------------------------------------------------------------------------


def test_sqlite_logger_function_use_redacts_api_key(db_connection):
    """Bug #19: api_key in function args/returns must be redacted in function_calls table."""
    source = "TestAgent"
    args = {"api_key": SENTINEL_API_KEY, "param": "value"}
    returns = {"api_key": SENTINEL_API_KEY, "result": 42}

    autogen.runtime_logging.log_function_use(
        agent=source,
        function=_dummy_fn,
        args=args,
        returns=returns,
    )

    cur = db_connection.cursor()
    row = cur.execute("SELECT args, returns FROM function_calls ORDER BY rowid DESC LIMIT 1").fetchone()
    assert row is not None

    logged_args = json.loads(row["args"])
    logged_returns = json.loads(row["returns"])

    assert logged_args.get("api_key") != SENTINEL_API_KEY, (
        f"api_key '{SENTINEL_API_KEY}' in args stored in plaintext (Bug #19)"
    )
    assert logged_args.get("api_key") == REDACTED

    assert logged_returns.get("api_key") != SENTINEL_API_KEY, (
        f"api_key '{SENTINEL_API_KEY}' in returns stored in plaintext (Bug #19)"
    )
    assert logged_returns.get("api_key") == REDACTED


# ---------------------------------------------------------------------------
# Bug: sqlite_logger.log_chat_completion uses json.dumps(to_dict(request)) without redaction
# ---------------------------------------------------------------------------


def test_sqlite_logger_chat_completion_redacts_api_key(db_connection):
    """sqlite_logger.log_chat_completion must redact api_key in the request column."""
    request = _make_request()

    source = MagicMock()
    source.name = "TestAgent"

    autogen.runtime_logging.log_chat_completion(
        invocation_id=str(uuid.uuid4()),
        client_id=1,
        wrapper_id=2,
        agent=source,
        request=request,
        response="some response",
        is_cached=0,
        cost=0.01,
        start_time=get_current_ts(),
    )

    cur = db_connection.cursor()
    row = cur.execute("SELECT request FROM chat_completions ORDER BY rowid DESC LIMIT 1").fetchone()
    assert row is not None

    stored_request = json.loads(row["request"])
    assert stored_request.get("api_key") != SENTINEL_API_KEY, (
        f"api_key '{SENTINEL_API_KEY}' stored in plaintext in chat_completions.request"
    )
    assert stored_request.get("api_key") == REDACTED


# ---------------------------------------------------------------------------
# Baseline: file_logger.log_event already redacts (was correct before the bugs)
# ---------------------------------------------------------------------------


def test_file_logger_event_already_redacts(file_logger, tmp_path):
    """file_logger.log_event already applies _redact -- verify the baseline."""
    file_logger.log_event(
        source="TestAgent",
        name="test_event",
        api_key=SENTINEL_API_KEY,
        safe_field="safe_value",
    )

    log_file = tmp_path / "autogen_logs" / "test.log"
    log_content = log_file.read_text()

    assert SENTINEL_API_KEY not in log_content, (
        "file_logger.log_event baseline broken -- api_key should already be redacted"
    )
    assert REDACTED in log_content


# ---------------------------------------------------------------------------
# Baseline: sqlite_logger.log_new_wrapper uses get_sensitive_exclude_keys (was correct)
# ---------------------------------------------------------------------------


def test_sqlite_logger_new_wrapper_already_excludes(db_connection):
    """sqlite_logger.log_new_wrapper uses get_sensitive_exclude_keys -- verify baseline."""
    wrapper = MagicMock()
    wrapper.__class__.__name__ = "OpenAIWrapper"

    init_args = {"llm_config": {"config_list": [{"model": "gpt-4o", "api_key": SENTINEL_API_KEY}]}}

    autogen.runtime_logging.log_new_wrapper(wrapper=wrapper, init_args=init_args)

    cur = db_connection.cursor()
    row = cur.execute("SELECT init_args FROM oai_wrappers ORDER BY rowid DESC LIMIT 1").fetchone()
    assert row is not None

    stored = json.loads(row["init_args"])
    stored_str = json.dumps(stored)
    assert SENTINEL_API_KEY not in stored_str, (
        "sqlite_logger.log_new_wrapper baseline broken -- api_key should be excluded"
    )


# ---------------------------------------------------------------------------
# Adversarial: nested api_key in request (redact must recurse)
# ---------------------------------------------------------------------------


def test_file_logger_chat_completion_redacts_nested_api_key(file_logger, tmp_path):
    """Nested api_key inside config_list must be redacted."""
    request = {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "gpt-4o",
        "llm_config": {"config_list": [{"model": "gpt-4o", "api_key": SENTINEL_API_KEY}]},
    }
    file_logger.log_chat_completion(
        invocation_id=uuid.uuid4(),
        client_id=1,
        wrapper_id=2,
        source="TestAgent",
        request=request,
        response="ok",
        is_cached=0,
        cost=0.01,
        start_time=get_current_ts(),
    )

    log_content = (tmp_path / "autogen_logs" / "test.log").read_text()
    assert SENTINEL_API_KEY not in log_content, "Nested api_key leaked in log file"


def test_sqlite_logger_event_redacts_nested_api_key(db_connection):
    """Nested api_key in event kwargs must be redacted."""
    autogen.runtime_logging.log_event(
        source="TestAgent",
        name="test_event",
        config={"api_key": SENTINEL_API_KEY, "model": "gpt-4o"},
    )

    cur = db_connection.cursor()
    row = cur.execute("SELECT json_state FROM events ORDER BY rowid DESC LIMIT 1").fetchone()
    assert row is not None
    assert SENTINEL_API_KEY not in row["json_state"], "Nested api_key leaked in events table"


# ---------------------------------------------------------------------------
# Adversarial: api_key inside a list of dicts
# ---------------------------------------------------------------------------


def test_sqlite_logger_event_redacts_api_key_in_list(db_connection):
    """api_key inside a list of dicts must be redacted."""
    autogen.runtime_logging.log_event(
        source="TestAgent",
        name="test_event",
        config_list=[{"model": "gpt-4o", "api_key": SENTINEL_API_KEY}],
    )

    cur = db_connection.cursor()
    row = cur.execute("SELECT json_state FROM events ORDER BY rowid DESC LIMIT 1").fetchone()
    assert row is not None
    assert SENTINEL_API_KEY not in row["json_state"], "api_key in list item leaked in events table"


# ---------------------------------------------------------------------------
# Bug: file_logger.log_new_agent did not redact init_args
# ---------------------------------------------------------------------------


def test_file_logger_new_agent_redacts_api_key(file_logger, tmp_path):
    """file_logger.log_new_agent must redact api_key in init_args."""
    agent = MagicMock()
    agent.name = "TestAgent"
    agent.client = None

    init_args = {"llm_config": {"api_key": SENTINEL_API_KEY, "model": "gpt-4o"}}
    file_logger.log_new_agent(agent=agent, init_args=init_args)

    log_content = (tmp_path / "autogen_logs" / "test.log").read_text()
    assert SENTINEL_API_KEY not in log_content, "api_key in init_args leaked via file_logger.log_new_agent"


# ---------------------------------------------------------------------------
# Adversarial: non-dict args/returns in log_function_use (boundary)
# ---------------------------------------------------------------------------


def test_sqlite_logger_function_use_non_dict_args_no_crash(db_connection):
    """Non-dict args (str, list, None) must not crash and must not leak secrets."""
    for args_val in ["some string", [1, 2, 3], None, 42]:
        autogen.runtime_logging.log_function_use(
            agent="TestAgent",
            function=_dummy_fn,
            args=args_val,
            returns=args_val,
        )

    cur = db_connection.cursor()
    rows = cur.execute("SELECT args, returns FROM function_calls").fetchall()
    assert len(rows) == 4


def test_file_logger_function_use_non_dict_args_no_crash(file_logger, tmp_path):
    """Non-dict args must not crash file_logger.log_function_use."""
    for args_val in ["some string", [1, 2, 3], None]:
        file_logger.log_function_use(
            source="TestAgent",
            function=_dummy_fn,
            args=args_val,
            returns=args_val,
        )

    log_content = (tmp_path / "autogen_logs" / "test.log").read_text()
    assert len(log_content) > 0


# ---------------------------------------------------------------------------
# Adversarial: key name case variations (redact uses k.lower())
# ---------------------------------------------------------------------------


def test_redact_handles_case_variations(db_connection):
    """api_key in various cases must all be redacted (redact uses .lower())."""
    autogen.runtime_logging.log_event(
        source="TestAgent",
        name="case_test",
        api_key=SENTINEL_API_KEY,
        Api_Key=SENTINEL_API_KEY,
        API_KEY=SENTINEL_API_KEY,
    )

    cur = db_connection.cursor()
    row = cur.execute("SELECT json_state FROM events ORDER BY rowid DESC LIMIT 1").fetchone()
    assert row is not None
    assert SENTINEL_API_KEY not in row["json_state"], "Case-variant api_key leaked"


# ---------------------------------------------------------------------------
# Adversarial: empty dict and empty kwargs
# ---------------------------------------------------------------------------


def test_sqlite_logger_event_empty_kwargs_no_crash(db_connection):
    """Empty kwargs must not crash log_event."""
    autogen.runtime_logging.log_event(source="TestAgent", name="empty_event")

    cur = db_connection.cursor()
    row = cur.execute("SELECT json_state FROM events ORDER BY rowid DESC LIMIT 1").fetchone()
    assert row is not None
    state = json.loads(row["json_state"])
    assert state == {}
