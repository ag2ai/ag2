# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen import ConversableAgent
from autogen.agents.experimental import DocAgent
from autogen.import_utils import require_optional_import
from test.conftest import Credentials


@require_optional_import("openai", "openai")
def test_issue_1324(credentials_gpt_4o_mini: Credentials, monkeypatch: pytest.MonkeyPatch) -> None:
    llm_config = credentials_gpt_4o_mini.llm_config

    monkeypatch.setenv("OPENAI_API_KEY", credentials_gpt_4o_mini.api_key)

    document_agent = DocAgent(name="document_agent", llm_config=llm_config, collection_name="toast_report")

    user_proxy = ConversableAgent(
        name="user_proxy",
        human_input_mode="ALWAYS",
    )

    user_proxy.initiate_chat(
        recipient=document_agent,
        message="could you ingest ../test/agentchat/contrib/graph_rag/Toast_financial_report.pdf? What is the fiscal year 2024 financial summary?",
    )
