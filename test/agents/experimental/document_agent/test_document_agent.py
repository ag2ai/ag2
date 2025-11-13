# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autogen.agents.experimental.document_agent.document_agent import (
    DEFAULT_SYSTEM_MESSAGE,
    DocAgent,
    DocumentTask,
    DocumentTriageAgent,
)
from autogen.agents.experimental.document_agent.document_utils import Ingest, Query, QueryType
from autogen.import_utils import run_for_optional_imports, skip_on_missing_imports

from ....conftest import Credentials


class TestDocumentTask:
    """Test DocumentTask model."""

    def test_document_task_init(self) -> None:
        """Test DocumentTask initialization."""
        ingestions = [Ingest(path_or_url="test.pdf")]
        queries = [Query(query_type=QueryType.RAG_QUERY, query="What is this about?")]

        task = DocumentTask(ingestions=ingestions, queries=queries)

        assert len(task.ingestions) == 1
        assert task.ingestions[0].path_or_url == "test.pdf"
        assert len(task.queries) == 1
        assert task.queries[0].query == "What is this about?"
        assert task.queries[0].query_type == QueryType.RAG_QUERY

    def test_document_task_empty_lists(self) -> None:
        """Test DocumentTask with empty lists."""
        task = DocumentTask(ingestions=[], queries=[])

        assert len(task.ingestions) == 0
        assert len(task.queries) == 0

    def test_document_task_serialization(self) -> None:
        """Test DocumentTask serialization."""
        ingestions = [Ingest(path_or_url="test.pdf")]
        queries = [Query(query_type=QueryType.RAG_QUERY, query="What is this about?")]

        task = DocumentTask(ingestions=ingestions, queries=queries)

        # Test that it can be serialized to dict
        task_dict = task.model_dump()
        assert "ingestions" in task_dict
        assert "queries" in task_dict

        # Test that it can be reconstructed from dict
        reconstructed = DocumentTask(**task_dict)
        assert len(reconstructed.ingestions) == 1
        assert reconstructed.ingestions[0].path_or_url == "test.pdf"


class TestDocumentTriageAgent:
    """Test DocumentTriageAgent class."""

    @run_for_optional_imports(["openai"], "openai")
    def test_document_triage_agent_init(self, credentials_gpt_4o_mini: Credentials) -> None:
        """Test DocumentTriageAgent initialization."""
        llm_config = credentials_gpt_4o_mini.llm_config
        triage_agent = DocumentTriageAgent(llm_config)

        assert triage_agent.name == "DocumentTriageAgent"
        assert triage_agent.llm_config["response_format"] == DocumentTask  # type: ignore [index]
        assert triage_agent.human_input_mode == "NEVER"
        assert "document triage agent" in triage_agent.system_message.lower()

    @run_for_optional_imports(["openai"], "openai")
    def test_document_triage_agent_init_none_config(self) -> None:
        """Test DocumentTriageAgent initialization with None config."""
        with patch("autogen.llm_config.LLMConfig.get_current_llm_config") as mock_get_config:
            mock_get_config.return_value = {"config_list": [{"model": "gpt-4o-mini", "api_key": "test"}]}
            triage_agent = DocumentTriageAgent(None)

            assert triage_agent.name == "DocumentTriageAgent"
            assert triage_agent.llm_config["response_format"] == DocumentTask  # type: ignore [index]

    @run_for_optional_imports(["openai"], "openai")
    def test_document_triage_agent_system_message(self, credentials_gpt_4o_mini: Credentials) -> None:
        """Test DocumentTriageAgent system message content."""
        llm_config = credentials_gpt_4o_mini.llm_config
        triage_agent = DocumentTriageAgent(llm_config)

        system_message = triage_agent.system_message
        assert "document triage agent" in system_message.lower()
        assert "documenttask" in system_message.lower()
        assert "ingestions" in system_message.lower()
        assert "queries" in system_message.lower()


class TestDocAgent:
    """Test DocAgent class."""

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_document_agent_init(self, credentials_gpt_4o_mini: Credentials, tmp_path: Path) -> None:
        """Test DocAgent initialization."""
        llm_config = credentials_gpt_4o_mini.llm_config
        document_agent = DocAgent(llm_config=llm_config, parsed_docs_path=tmp_path)

        # Check that the correct agents are initialized (not the old ones)
        assert hasattr(document_agent, "_task_manager_agent")
        assert hasattr(document_agent, "_triage_agent")
        assert hasattr(document_agent, "_summary_agent")

        # Check that old agents are NOT present
        assert not hasattr(document_agent, "_data_ingestion_agent")
        assert not hasattr(document_agent, "_query_agent")
        assert not hasattr(document_agent, "_error_agent")

        # Check initialization values
        assert document_agent.name == "DocAgent"
        assert document_agent.human_input_mode == "NEVER"
        assert document_agent.documents_ingested == []

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_document_agent_init_defaults(self, credentials_gpt_4o_mini: Credentials) -> None:
        """Test DocAgent initialization with default values."""
        llm_config = credentials_gpt_4o_mini.llm_config
        document_agent = DocAgent(llm_config=llm_config)

        assert document_agent.name == "DocAgent"
        assert document_agent.system_message == DEFAULT_SYSTEM_MESSAGE
        assert document_agent.human_input_mode == "NEVER"

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_document_agent_init_custom_name(self, credentials_gpt_4o_mini: Credentials, tmp_path: Path) -> None:
        """Test DocAgent initialization with custom name."""
        llm_config = credentials_gpt_4o_mini.llm_config
        custom_name = "CustomDocAgent"
        document_agent = DocAgent(name=custom_name, llm_config=llm_config, parsed_docs_path=tmp_path)

        assert document_agent.name == custom_name

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_document_agent_init_custom_system_message(
        self, credentials_gpt_4o_mini: Credentials, tmp_path: Path
    ) -> None:
        """Test DocAgent initialization with custom system message."""
        llm_config = credentials_gpt_4o_mini.llm_config
        custom_message = "Custom system message for testing."
        document_agent = DocAgent(llm_config=llm_config, system_message=custom_message, parsed_docs_path=tmp_path)

        assert document_agent.system_message == custom_message

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_document_agent_init_with_custom_query_engine(
        self, credentials_gpt_4o_mini: Credentials, tmp_path: Path
    ) -> None:
        """Test DocAgent initialization with custom query engine."""
        llm_config = credentials_gpt_4o_mini.llm_config
        mock_query_engine = MagicMock()

        document_agent = DocAgent(llm_config=llm_config, parsed_docs_path=tmp_path, query_engine=mock_query_engine)

        assert document_agent._task_manager_agent.query_engine == mock_query_engine

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_get_document_input_message_valid(self, credentials_gpt_4o_mini: Credentials, tmp_path: Path) -> None:
        """Test _get_document_input_message with valid messages."""
        llm_config = credentials_gpt_4o_mini.llm_config
        with patch("autogen.agents.experimental.document_agent.chroma_query_engine.VectorChromaQueryEngine"):
            document_agent = DocAgent(llm_config=llm_config, parsed_docs_path=tmp_path)

            messages = [{"content": "Test message", "role": "user"}]
            result = document_agent._get_document_input_message(messages)

            assert result == "Test message"

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_get_document_input_message_none(self, credentials_gpt_4o_mini: Credentials, tmp_path: Path) -> None:
        """Test _get_document_input_message with None messages."""
        llm_config = credentials_gpt_4o_mini.llm_config
        with patch("autogen.agents.experimental.document_agent.chroma_query_engine.VectorChromaQueryEngine"):
            document_agent = DocAgent(llm_config=llm_config, parsed_docs_path=tmp_path)

            result = document_agent._get_document_input_message(None)

            assert result == ""

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_get_document_input_message_empty_list(self, credentials_gpt_4o_mini: Credentials, tmp_path: Path) -> None:
        """Test _get_document_input_message with empty messages list."""
        llm_config = credentials_gpt_4o_mini.llm_config
        with patch("autogen.agents.experimental.document_agent.chroma_query_engine.VectorChromaQueryEngine"):
            document_agent = DocAgent(llm_config=llm_config, parsed_docs_path=tmp_path)

            result = document_agent._get_document_input_message([])

            assert result == ""

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_get_document_input_message_invalid_format(
        self, credentials_gpt_4o_mini: Credentials, tmp_path: Path
    ) -> None:
        """Test _get_document_input_message with invalid message format."""
        llm_config = credentials_gpt_4o_mini.llm_config
        with patch("autogen.agents.experimental.document_agent.chroma_query_engine.VectorChromaQueryEngine"):
            document_agent = DocAgent(llm_config=llm_config, parsed_docs_path=tmp_path)

            with pytest.raises(NotImplementedError, match="Invalid messages format"):
                document_agent._get_document_input_message([{"invalid": "format"}])

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    @patch("autogen.agents.experimental.document_agent.document_agent.initiate_group_chat")
    def test_generate_inner_group_chat_reply_basic(
        self, mock_initiate_group_chat: MagicMock, credentials_gpt_4o_mini: Credentials, tmp_path: Path
    ) -> None:
        """Test generate_inner_group_chat_reply basic functionality."""
        llm_config = credentials_gpt_4o_mini.llm_config

        with patch("autogen.agents.experimental.document_agent.chroma_query_engine.VectorChromaQueryEngine"):
            document_agent = DocAgent(llm_config=llm_config, parsed_docs_path=tmp_path)

            # Mock the group chat result
            mock_result = MagicMock()
            mock_result.summary = "Test summary"
            mock_initiate_group_chat.return_value = (mock_result, MagicMock(), MagicMock())

            messages = [{"content": "Test message", "role": "user"}]
            success, reply = document_agent.generate_inner_group_chat_reply(messages=messages)

            assert success is True
            assert reply == "Test summary"
            mock_initiate_group_chat.assert_called_once()

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    @patch("autogen.agents.experimental.document_agent.document_agent.initiate_group_chat")
    def test_generate_inner_group_chat_reply_with_triage_output(
        self, mock_initiate_group_chat: MagicMock, credentials_gpt_4o_mini: Credentials, tmp_path: Path
    ) -> None:
        """Test generate_inner_group_chat_reply with triage agent output."""
        llm_config = credentials_gpt_4o_mini.llm_config

        with patch("autogen.agents.experimental.document_agent.chroma_query_engine.VectorChromaQueryEngine"):
            document_agent = DocAgent(llm_config=llm_config, parsed_docs_path=tmp_path)

            # Mock the group chat result
            mock_result = MagicMock()
            mock_result.summary = "Test summary"
            mock_initiate_group_chat.return_value = (mock_result, MagicMock(), MagicMock())

            # Create a triage agent output message
            triage_output = {
                "ingestions": [{"path_or_url": "test.pdf"}],
                "queries": [{"query_type": "RAG_QUERY", "query": "What is this about?"}],
            }
            messages = [{"name": "DocumentTriageAgent", "content": json.dumps(triage_output), "role": "assistant"}]

            success, reply = document_agent.generate_inner_group_chat_reply(messages=messages)

            assert success is True
            assert reply == "Test summary"
            mock_initiate_group_chat.assert_called_once()

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_generate_inner_group_chat_reply_invalid_json(
        self, credentials_gpt_4o_mini: Credentials, tmp_path: Path
    ) -> None:
        """Test generate_inner_group_chat_reply with invalid JSON in triage output."""
        llm_config = credentials_gpt_4o_mini.llm_config

        with (
            patch("autogen.agents.experimental.document_agent.chroma_query_engine.VectorChromaQueryEngine"),
            patch(
                "autogen.agents.experimental.document_agent.document_agent.initiate_group_chat"
            ) as mock_initiate_group_chat,
        ):
            document_agent = DocAgent(llm_config=llm_config, parsed_docs_path=tmp_path)

            # Mock the group chat result
            mock_result = MagicMock()
            mock_result.summary = "Test summary"
            mock_initiate_group_chat.return_value = (mock_result, MagicMock(), MagicMock())
            # Create a triage agent output message with invalid JSON
            messages = [{"name": "DocumentTriageAgent", "content": "invalid json content", "role": "assistant"}]
            success, reply = document_agent.generate_inner_group_chat_reply(messages=messages)
            assert success is True
            assert reply == "Test summary"
            mock_initiate_group_chat.assert_called_once()

    @pytest.mark.openai
    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_context_variables_initialization(self, credentials_gpt_4o_mini: Credentials, tmp_path: Path) -> None:
        """Test that context variables are properly initialized."""
        llm_config = credentials_gpt_4o_mini.llm_config

        with (
            patch("autogen.agents.experimental.document_agent.chroma_query_engine.VectorChromaQueryEngine"),
            patch(
                "autogen.agents.experimental.document_agent.document_agent.initiate_group_chat"
            ) as mock_initiate_group_chat,
        ):
            document_agent = DocAgent(llm_config=llm_config, parsed_docs_path=tmp_path)

            # Mock the group chat result
            mock_result = MagicMock()
            mock_result.summary = "Test summary"
            mock_initiate_group_chat.return_value = (mock_result, MagicMock(), MagicMock())
            messages = [{"content": "Test message", "role": "user"}]
            document_agent.generate_inner_group_chat_reply(messages=messages)
            # Check that context variables were passed to initiate_group_chat
            call_args = mock_initiate_group_chat.call_args
            pattern = call_args[1]["pattern"]
            context_variables = pattern.context_variables
            assert "CompletedTaskCount" in context_variables
            assert "DocumentsToIngest" in context_variables
            assert "DocumentsIngested" in context_variables
            assert "QueriesToRun" in context_variables
            assert "QueryResults" in context_variables
            assert context_variables["CompletedTaskCount"] == 0
            assert context_variables["DocumentsToIngest"] == []
            assert context_variables["DocumentsIngested"] == []
            assert context_variables["QueriesToRun"] == []
            assert context_variables["QueryResults"] == []
