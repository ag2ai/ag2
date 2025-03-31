# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from autogen.agents.experimental.document_agent.document_agent import (
    DocAgent,
    DocumentTask,
    DocumentTriageAgent,
)
from autogen.import_utils import run_for_optional_imports, skip_on_missing_imports

from ....conftest import Credentials


@run_for_optional_imports(["openai"], "openai")
def test_document_triage_agent_init(credentials_gpt_4o_mini: Credentials) -> None:
    llm_config = credentials_gpt_4o_mini.llm_config
    triage_agent = DocumentTriageAgent(llm_config)
    assert triage_agent.llm_config["response_format"] == DocumentTask  # type: ignore [index]


@pytest.mark.openai
@skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
def test_document_agent_init(credentials_gpt_4o_mini: Credentials, tmp_path: Path) -> None:
    llm_config = credentials_gpt_4o_mini.llm_config
    document_agent = DocAgent(llm_config=llm_config, parsed_docs_path=tmp_path)

    assert hasattr(document_agent, "_task_manager_agent")
    assert hasattr(document_agent, "_triage_agent")
    assert hasattr(document_agent, "_data_ingestion_agent")
    assert hasattr(document_agent, "_query_agent")
    assert hasattr(document_agent, "_error_agent")
    assert hasattr(document_agent, "_summary_agent")
