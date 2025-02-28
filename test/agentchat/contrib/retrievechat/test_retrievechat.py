# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# !/usr/bin/env python3 -m pytest

import sys

import pytest

from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import (
    RetrieveUserProxyAgent,
)
from autogen.import_utils import optional_import_block, run_for_optional_imports

from ....conftest import Credentials, reason

with optional_import_block() as result:
    import chromadb
    from chromadb.utils import embedding_functions as ef


reason = "do not run on MacOS or windows OR dependency is not installed OR " + reason


@run_for_optional_imports("openai", "openai")
@pytest.mark.skipif(
    sys.platform in ["darwin", "win32"],
    reason=reason,
)
@run_for_optional_imports(["chromadb", "IPython", "openai"], "retrievechat")
def test_retrievechat(credentials_gpt_4o_mini: Credentials):
    conversations = {}
    # autogen.ChatCompletion.start_logging(conversations)  # deprecated in v0.2

    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        llm_config={
            "timeout": 600,
            "seed": 42,
            "config_list": credentials_gpt_4o_mini.config_list,
        },
    )

    sentence_transformer_ef = ef.SentenceTransformerEmbeddingFunction()
    ragproxyagent = RetrieveUserProxyAgent(
        name="ragproxyagent",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=2,
        retrieve_config={
            "docs_path": "./website/docs",
            "chunk_token_size": 2000,
            "model": credentials_gpt_4o_mini.config_list[0]["model"],
            "client": chromadb.PersistentClient(path="/tmp/chromadb"),
            "embedding_function": sentence_transformer_ef,
            "get_or_create": True,
        },
    )

    assistant.reset()

    code_problem = "How can I use FLAML to perform a classification task, set use_spark=True, train 30 seconds and force cancel jobs if time limit is reached."
    ragproxyagent.initiate_chat(
        assistant, message=ragproxyagent.message_generator, problem=code_problem, search_string="spark", silent=True
    )

    print(conversations)


@pytest.mark.skipif(
    sys.platform in ["darwin", "win32"],
    reason=reason,
)
@run_for_optional_imports(["chromadb", "IPython", "openai"], "retrievechat")
def test_retrieve_config():
    # test warning message when no docs_path is provided
    ragproxyagent = RetrieveUserProxyAgent(
        name="ragproxyagent",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=2,
        retrieve_config={
            "chunk_token_size": 2000,
            "get_or_create": True,
        },
    )
    assert ragproxyagent._docs_path is None


if __name__ == "__main__":
    # test_retrievechat()
    test_retrieve_config()
