# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen import ConversableAgent
from autogen.agents.experimental import DocAgent
from autogen.import_utils import require_optional_import
from test.conftest import Credentials


@require_optional_import("openai", "openai")
@pytest.mark.skip(
    reason="This test is failing due to the fact that the document_agent is not able to ingest the document"
)
def test_issue_1324(credentials_gpt_4o_mini: Credentials, monkeypatch: pytest.MonkeyPatch) -> None:
    llm_config = credentials_gpt_4o_mini.llm_config

    monkeypatch.setenv("OPENAI_API_KEY", credentials_gpt_4o_mini.api_key)

    document_agent = DocAgent(name="document_agent", llm_config=llm_config, collection_name="toast_report")

    user_proxy = ConversableAgent(
        name="user_proxy",
        human_input_mode="NEVER",
    )

    summary = user_proxy.initiate_chat(
        recipient=document_agent,
        message="could you ingest ../test/agentchat/contrib/graph_rag/Toast_financial_report.pdf? What is the fiscal year 2024 financial summary?",
        max_turns=10,
        summary_method="reflection_with_llm",
    )

    assert "error" not in summary.summary.lower()
    assert "fail" not in summary.summary.lower()
