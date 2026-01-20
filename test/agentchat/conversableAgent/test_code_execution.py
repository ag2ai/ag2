# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from autogen.agentchat import ConversableAgent
from autogen.coding.base import CodeBlock, CodeResult
from autogen.events.agent_events import ExecuteCodeBlockEvent, GenerateCodeExecutionReplyEvent


@pytest.fixture
def agent_with_executor():
    """Create an agent with code executor configured."""
    agent = ConversableAgent(
        name="test_agent",
        llm_config=False,
        code_execution_config={
            "executor": "commandline-local",
            "last_n_messages": "auto",
            "use_docker": False,
        },
    )
    return agent


@pytest.fixture
def agent_with_legacy_config():
    """Create an agent with legacy code execution config."""
    agent = ConversableAgent(
        name="test_agent",
        llm_config=False,
        code_execution_config={"last_n_messages": 3, "use_docker": False},
    )
    return agent


def test__generate_code_execution_reply_using_executor_config_validation(agent_with_executor):
    """Test that config parameter raises ValueError."""
    with pytest.raises(ValueError, match="config is not supported"):
        agent_with_executor._generate_code_execution_reply_using_executor(
            messages=[], sender=None, config={"test": "value"}
        )


def test__generate_code_execution_reply_using_executor_config_false(agent_with_executor):
    """Test that False config returns False, None."""
    agent_with_executor._code_execution_config = False
    result = agent_with_executor._generate_code_execution_reply_using_executor()
    assert result == (False, None)


def test__generate_code_execution_reply_using_executor_no_code_blocks(agent_with_executor):
    """Test when no code blocks are found."""
    messages = [{"content": "no code here", "role": "user"}]
    mock_extractor = MagicMock()
    mock_extractor.extract_code_blocks.return_value = []
    agent_with_executor._code_executor.code_extractor = mock_extractor

    result = agent_with_executor._generate_code_execution_reply_using_executor(messages=messages)
    assert result == (False, None)


def test__generate_code_execution_reply_using_executor_with_code_blocks(agent_with_executor):
    """Test successful code execution with code blocks."""
    messages = [{"content": "```python\nprint('hello')\n```", "role": "user"}]
    code_blocks = [CodeBlock(code="print('hello')", language="python")]
    code_result = CodeResult(exit_code=0, output="hello")

    mock_extractor = MagicMock()
    mock_extractor.extract_code_blocks.return_value = code_blocks
    mock_executor = MagicMock()
    mock_executor.execute_code_blocks.return_value = code_result
    agent_with_executor._code_executor.code_extractor = mock_extractor
    agent_with_executor._code_executor.execute_code_blocks = mock_executor.execute_code_blocks

    mock_iostream = MagicMock()
    with patch("autogen.agentchat.conversableAgent.code_execution.IOStream.get_default", return_value=mock_iostream):
        result = agent_with_executor._generate_code_execution_reply_using_executor(messages=messages)

    assert result == (True, "exitcode: 0 (execution succeeded)\nCode output: hello")
    mock_iostream.send.assert_called_once()
    assert isinstance(mock_iostream.send.call_args[0][0], GenerateCodeExecutionReplyEvent)


def test__generate_code_execution_reply_using_executor_failed_execution(agent_with_executor):
    """Test code execution with non-zero exit code."""
    messages = [{"content": "```python\nraise Exception('error')\n```", "role": "user"}]
    code_blocks = [CodeBlock(code="raise Exception('error')", language="python")]
    code_result = CodeResult(exit_code=1, output="Traceback...")

    mock_extractor = MagicMock()
    mock_extractor.extract_code_blocks.return_value = code_blocks
    mock_executor = MagicMock()
    mock_executor.execute_code_blocks.return_value = code_result
    agent_with_executor._code_executor.code_extractor = mock_extractor
    agent_with_executor._code_executor.execute_code_blocks = mock_executor.execute_code_blocks

    mock_iostream = MagicMock()
    with patch("autogen.agentchat.conversableAgent.code_execution.IOStream.get_default", return_value=mock_iostream):
        result = agent_with_executor._generate_code_execution_reply_using_executor(messages=messages)

    assert result == (True, "exitcode: 1 (execution failed)\nCode output: Traceback...")
    assert "execution failed" in result[1]


def test__generate_code_execution_reply_using_executor_last_n_messages(agent_with_executor):
    """Test last_n_messages parameter."""
    agent_with_executor._code_execution_config = {"last_n_messages": 1, "use_docker": False}
    messages = [
        {"content": "old message", "role": "user"},
        {"content": "```python\nprint('new')\n```", "role": "user"},
    ]

    code_blocks = [CodeBlock(code="print('new')", language="python")]
    code_result = CodeResult(exit_code=0, output="new")

    mock_extractor = MagicMock()
    mock_extractor.extract_code_blocks.return_value = code_blocks
    mock_executor = MagicMock()
    mock_executor.execute_code_blocks.return_value = code_result
    agent_with_executor._code_executor.code_extractor = mock_extractor
    agent_with_executor._code_executor.execute_code_blocks = mock_executor.execute_code_blocks

    mock_iostream = MagicMock()
    with patch("autogen.agentchat.conversableAgent.code_execution.IOStream.get_default", return_value=mock_iostream):
        result = agent_with_executor._generate_code_execution_reply_using_executor(messages=messages)

    assert result[0] is True
    # Should only scan last 1 message
    assert mock_extractor.extract_code_blocks.call_count >= 1


def test__generate_code_execution_reply_using_executor_auto_mode(agent_with_executor):
    """Test auto mode for last_n_messages."""
    agent_with_executor._code_execution_config = {"last_n_messages": "auto", "use_docker": False}
    messages = [
        {"content": "```python\nprint('test')\n```", "role": "user"},
        {"content": "another user message", "role": "user"},
    ]

    code_blocks = [CodeBlock(code="print('test')", language="python")]
    code_result = CodeResult(exit_code=0, output="test")

    mock_extractor = MagicMock()
    mock_extractor.extract_code_blocks.return_value = code_blocks
    mock_executor = MagicMock()
    mock_executor.execute_code_blocks.return_value = code_result
    agent_with_executor._code_executor.code_extractor = mock_extractor
    agent_with_executor._code_executor.execute_code_blocks = mock_executor.execute_code_blocks

    mock_iostream = MagicMock()
    with patch("autogen.agentchat.conversableAgent.code_execution.IOStream.get_default", return_value=mock_iostream):
        result = agent_with_executor._generate_code_execution_reply_using_executor(messages=messages)

    assert result[0] is True


def test__generate_code_execution_reply_using_executor_invalid_last_n_messages(agent_with_executor):
    """Test invalid last_n_messages raises ValueError."""
    agent_with_executor._code_execution_config = {"last_n_messages": -1, "use_docker": False}
    messages = [{"content": "test", "role": "user"}]

    with pytest.raises(ValueError, match="last_n_messages must be either"):
        agent_with_executor._generate_code_execution_reply_using_executor(messages=messages)


def test_execute_code_blocks_python(agent_with_legacy_config):
    """Test executing Python code blocks."""
    code_blocks = [("python", "print('hello world')")]

    with patch.object(agent_with_legacy_config, "run_code", return_value=(0, "hello world", None)):
        mock_iostream = MagicMock()
        with patch(
            "autogen.agentchat.conversableAgent.code_execution.IOStream.get_default", return_value=mock_iostream
        ):
            exitcode, logs = agent_with_legacy_config.execute_code_blocks(code_blocks)

    assert exitcode == 0
    assert "hello world" in logs
    mock_iostream.send.assert_called_once()
    assert isinstance(mock_iostream.send.call_args[0][0], ExecuteCodeBlockEvent)


def test_execute_code_blocks_bash(agent_with_legacy_config):
    """Test executing bash/shell code blocks."""
    code_blocks = [("bash", "echo 'test'")]

    with patch.object(agent_with_legacy_config, "run_code", return_value=(0, "test", None)):
        mock_iostream = MagicMock()
        with patch(
            "autogen.agentchat.conversableAgent.code_execution.IOStream.get_default", return_value=mock_iostream
        ):
            exitcode, logs = agent_with_legacy_config.execute_code_blocks(code_blocks)

    assert exitcode == 0
    assert "test" in logs


def test_execute_code_blocks_unknown_language(agent_with_legacy_config):
    """Test executing code blocks with unknown language."""
    code_blocks = [("unknown_lang", "some code")]

    exitcode, logs = agent_with_legacy_config.execute_code_blocks(code_blocks)

    assert exitcode == 1
    assert "unknown language" in logs


def test_execute_code_blocks_multiple_blocks(agent_with_legacy_config):
    """Test executing multiple code blocks."""
    code_blocks = [
        ("python", "print('first')"),
        ("python", "print('second')"),
    ]

    with patch.object(agent_with_legacy_config, "run_code", side_effect=[(0, "first", None), (0, "second", None)]):
        mock_iostream = MagicMock()
        with patch(
            "autogen.agentchat.conversableAgent.code_execution.IOStream.get_default", return_value=mock_iostream
        ):
            exitcode, logs = agent_with_legacy_config.execute_code_blocks(code_blocks)

    assert exitcode == 0
    assert "first" in logs
    assert "second" in logs
    assert mock_iostream.send.call_count == 2


def test_execute_code_blocks_early_exit_on_error(agent_with_legacy_config):
    """Test that execution stops early on non-zero exit code."""
    code_blocks = [
        ("python", "raise Exception('error')"),
        ("python", "print('never executed')"),
    ]

    with patch.object(agent_with_legacy_config, "run_code", return_value=(1, "error occurred", None)):
        exitcode, logs = agent_with_legacy_config.execute_code_blocks(code_blocks)

    assert exitcode == 1
    assert "error occurred" in logs
    assert "never executed" not in logs


def test_execute_code_blocks_infer_language(agent_with_legacy_config):
    """Test language inference when language is not provided."""
    code_blocks = [("", "print('hello')")]  # Empty language

    with patch("autogen.agentchat.conversableAgent.code_execution.infer_lang", return_value="python"):
        with patch.object(agent_with_legacy_config, "run_code", return_value=(0, "hello", None)):
            exitcode, logs = agent_with_legacy_config.execute_code_blocks(code_blocks)

    assert exitcode == 0


def test_execute_code_blocks_filename_extraction(agent_with_legacy_config):
    """Test filename extraction from code block."""
    code_blocks = [("python", "# filename: test.py\nprint('hello')")]

    with patch.object(agent_with_legacy_config, "run_code", return_value=(0, "hello", None)) as mock_run:
        agent_with_legacy_config.execute_code_blocks(code_blocks)

    # Check that filename was passed to run_code
    call_kwargs = mock_run.call_args[1]
    assert call_kwargs.get("filename") == "test.py"


def test_execute_code_blocks_docker_image_update(agent_with_legacy_config):
    """Test that docker image is updated in config when returned."""
    code_blocks = [("python", "print('test')")]
    docker_image = "python:3.9"

    with patch.object(agent_with_legacy_config, "run_code", return_value=(0, "test", docker_image)):
        agent_with_legacy_config.execute_code_blocks(code_blocks)

    assert agent_with_legacy_config._code_execution_config.get("use_docker") == docker_image


def test_execute_code_blocks_empty_content(agent_with_legacy_config):
    """Test handling empty code blocks."""
    code_blocks = [("python", "")]

    with patch.object(agent_with_legacy_config, "run_code", return_value=(0, "", None)):
        exitcode, logs = agent_with_legacy_config.execute_code_blocks(code_blocks)

    assert exitcode == 0


def test_run_code_delegates_to_execute_code(agent_with_legacy_config):
    """Test that run_code delegates to execute_code from code_utils."""
    code = "print('test')"
    kwargs = {"lang": "python", "use_docker": False}

    with patch(
        "autogen.agentchat.conversableAgent.code_execution.execute_code", return_value=(0, "test", None)
    ) as mock_execute:
        result = agent_with_legacy_config.run_code(code, **kwargs)

    assert result == (0, "test", None)
    mock_execute.assert_called_once_with(code, **kwargs)


def test_run_code_passes_kwargs(agent_with_legacy_config):
    """Test that run_code passes all kwargs to execute_code."""
    code = "print('test')"
    kwargs = {"lang": "python", "use_docker": False, "timeout": 30, "work_dir": "/tmp"}

    with patch(
        "autogen.agentchat.conversableAgent.code_execution.execute_code", return_value=(0, "test", None)
    ) as mock_execute:
        agent_with_legacy_config.run_code(code, **kwargs)

    call_kwargs = mock_execute.call_args[1]
    assert call_kwargs["lang"] == "python"
    assert call_kwargs["use_docker"] is False
    assert call_kwargs["timeout"] == 30
    assert call_kwargs["work_dir"] == "/tmp"


def test_run_code_return_format(agent_with_legacy_config):
    """Test that run_code returns correct format (exitcode, logs, image)."""
    code = "print('test')"

    with patch(
        "autogen.agentchat.conversableAgent.code_execution.execute_code", return_value=(0, "test output", "python:3.9")
    ):
        exitcode, logs, image = agent_with_legacy_config.run_code(code)

    assert isinstance(exitcode, int)
    assert isinstance(logs, str)
    assert image == "python:3.9"


def test__generate_code_execution_reply_using_executor_empty_content(agent_with_executor):
    """Test handling messages with empty content."""
    messages = [{"content": "", "role": "user"}]

    mock_extractor = MagicMock()
    mock_extractor.extract_code_blocks.return_value = []
    agent_with_executor._code_executor.code_extractor = mock_extractor

    result = agent_with_executor._generate_code_execution_reply_using_executor(messages=messages)
    assert result == (False, None)


def test__generate_code_execution_reply_using_executor_messages_from_sender(agent_with_executor):
    """Test that messages are retrieved from sender when not provided."""
    sender = ConversableAgent(name="sender", llm_config=False)
    agent_with_executor._oai_messages[sender] = [{"content": "```python\nprint('test')\n```", "role": "user"}]

    code_blocks = [CodeBlock(code="print('test')", language="python")]
    code_result = CodeResult(exit_code=0, output="test")

    mock_extractor = MagicMock()
    mock_extractor.extract_code_blocks.return_value = code_blocks
    mock_executor = MagicMock()
    mock_executor.execute_code_blocks.return_value = code_result
    agent_with_executor._code_executor.code_extractor = mock_extractor
    agent_with_executor._code_executor.execute_code_blocks = mock_executor.execute_code_blocks

    mock_iostream = MagicMock()
    with patch("autogen.agentchat.conversableAgent.code_execution.IOStream.get_default", return_value=mock_iostream):
        result = agent_with_executor._generate_code_execution_reply_using_executor(sender=sender, messages=None)

    assert result[0] is True
