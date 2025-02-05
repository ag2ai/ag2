# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, Optional, Union

from autogen import SwarmAgent
from autogen.agentchat.contrib.rag.parser_utils import docling_parse_docs
from autogen.tools.tool import Tool

parser_tool = Tool(
    name="docling_parse_docs",
    description="Use this tool to parse and understand text.",
    func_or_tool=docling_parse_docs,
)


DEFALT_DOCLING_PARSER_PROMPT = """
You are an expert in parsing and understanding text. You can use this tool to parse various documents and extract information from them.
"""


class DoclingDocIngestAgent(SwarmAgent):
    """
    A DoclingDocIngestAgent is a swarm agent that ingests documents using the docling_parse_docs tool.
    """

    def __init__(
        self,
        name: str = "DoclingDocIngestAgent",
        llm_config: Optional[Union[dict, Literal[False]]] = None,
    ):
        super().__init__(
            name=name,
            llm_config=llm_config,
            tools=[parser_tool],
            system_message=DEFALT_DOCLING_PARSER_PROMPT,
        )
