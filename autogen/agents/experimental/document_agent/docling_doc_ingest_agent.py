# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Literal, Optional, Union

from autogen import ConversableAgent
from autogen.agentchat.contrib.swarm_agent import SwarmResult
from autogen.agents.experimental.document_agent.parser_utils import docling_parse_docs

from .docling_query_engine import DoclingMdQueryEngine
from .document_utils import preprocess_path

from .docling_query_engine import DoclingMdQueryEngine
from .document_utils import preprocess_path

DOCLING_PARSE_TOOL_NAME = "docling_parse_docs"

DEFAULT_DOCLING_PARSER_PROMPT = f"""
You are an expert in parsing and understanding text. You can use {DOCLING_PARSE_TOOL_NAME} tool to parse various documents and extract information from them.
"""


class DoclingDocIngestAgent(ConversableAgent):
    """
    A DoclingDocIngestAgent is a swarm agent that ingests documents using the docling_parse_docs tool.
    """

    def __init__(
        self,
        name: str = "DoclingDocIngestAgent",
        llm_config: Optional[Union[dict, Literal[False]]] = None,  # type: ignore[type-arg]
        parsed_docs_path: Union[Path, str] = "./parsed_docs",
        query_engine: Optional[DoclingMdQueryEngine] = None,
    ):
        if query_engine:
            self.docling_query_engine = query_engine
        else:
            self.docling_query_engine = DoclingMdQueryEngine()

        parsed_docs_path = preprocess_path(str_or_path=parsed_docs_path)

        def data_ingest_task(context_variables: dict) -> SwarmResult:  # type: ignore[type-arg]
            tasks = context_variables.get("DocumentsToIngest", [])
            while tasks:
                task = tasks.pop()
                input_file_path = task["path_or_url"]
                output_files = docling_parse_docs(
                    input_file_path=input_file_path, output_dir_path=parsed_docs_path, output_formats=["markdown"]
                )

                # Limit to one output markdown file for now.
                if output_files:
                    output_file = output_files[0]
                    if output_file.suffix == ".md":
                        if self.docling_query_engine.get_collection_name() is None:
                            self.docling_query_engine.init_db(input_doc_paths=[output_file])
                        else:
                            self.docling_query_engine.add_docs(new_doc_paths=[output_file])

            context_variables["CompletedTaskCount"] += 1
            print("docling ingest:", context_variables, "\n", context_variables)

            return SwarmResult(
                agent="TaskManagerAgent",
                values=f"Data Ingestion Task Completed for {input_file_path}",
                context_variables=context_variables,
            )

        super().__init__(
            name=name,
            llm_config=llm_config,
            functions=[data_ingest_task],
            system_message=DEFAULT_DOCLING_PARSER_PROMPT,
        )
