# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Optional, Union

from pydantic import BaseModel, Field

from .... import Agent, ConversableAgent, UpdateSystemMessage
from ....agentchat.contrib.rag.query_engine import RAGQueryEngine
from ....agentchat.contrib.swarm_agent import (
    AfterWork,
    AfterWorkOption,
    OnCondition,
    OnContextCondition,
    SwarmResult,
    initiate_swarm_chat,
    register_hand_off,
)
from ....agentchat.utils import ContextExpression
from ....doc_utils import export_module
from ....oai.client import OpenAIWrapper
from .chroma_query_engine import VectorChromaQueryEngine
from .docling_doc_ingest_agent import DoclingDocIngestAgent

__all__ = ["DocAgent"]

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_MESSAGE = """
    You are a document agent.
    You are given a list of documents to ingest and a list of queries to perform.
    You are responsible for ingesting the documents and answering the queries.
"""
TASK_MANAGER_NAME = "TaskManagerAgent"
TASK_MANAGER_SYSTEM_MESSAGE = """
    You are a task manager agent. You have 2 priorities:
    1. You initiate the tasks which updates the context variables based on the task decisions (DocumentTask) from the DocumentTriageAgent.
    If the DocumentTriageAgent has suggested any ingestions or queries, call initiate_tasks to record them.
    Put all ingestion and query tasks into the one tool call.
        i.e. output
        {
            "ingestions": [
                {
                    "path_or_url": "path_or_url"
                }
            ],
            "queries": [
                {
                    "query_type": "RAG_QUERY",
                    "query": "query"
                }
            ],
            "query_results": [
                {
                    "query": "query",
                    "result": "result"
                }
            ]
        }
    2. If there are no documents to ingest and no queries to run, hand control off to the summary agent.

    Put all file paths and URLs into the ingestions. A http/https URL is also a valid path and should be ingested.

    Use the initiate_tasks tool to incorporate all ingestions and queries. Don't call it again until new ingestions or queries are raised.

    New ingestions and queries may be raised from time to time, so use the initiate_tasks again if you see new ingestions/queries.

    Transfer to the summary agent if all ingestion and query tasks are done.
    """

DEFAULT_ERROR_SWARM_MESSAGE: str = """
Document Agent failed to perform task.
"""

ERROR_MANAGER_NAME = "ErrorManagerAgent"
ERROR_MANAGER_SYSTEM_MESSAGE = """
You communicate errors to the user. Include the original error messages in full. Use the format:
The following error(s) have occurred:
- Error 1
- Error 2
"""


class QueryType(Enum):
    RAG_QUERY = "RAG_QUERY"
    # COMMON_QUESTION = "COMMON_QUESTION"


class Ingest(BaseModel):
    path_or_url: str = Field(description="The path or URL of the documents to ingest.")


class Query(BaseModel):
    query_type: QueryType = Field(description="The type of query to perform for the Document Agent.")
    query: str = Field(description="The query to perform for the Document Agent.")


class DocumentTask(BaseModel):
    """The structured output format for task decisions."""

    ingestions: list[Ingest] = Field(description="The list of documents to ingest.")
    queries: list[Query] = Field(description="The list of queries to perform.")

    def format(self) -> str:
        """Format the DocumentTask as a string for the TaskManager to work with."""
        if len(self.ingestions) == 0 and len(self.queries) == 0:
            return "There were no ingestion or query tasks detected."

        instructions = "Tasks:\n\n"
        order = 1

        if len(self.ingestions) > 0:
            instructions += "Ingestions:\n"
            for ingestion in self.ingestions:
                instructions += f"{order}: {ingestion.path_or_url}\n"
                order += 1

            instructions += "\n"

        if len(self.queries) > 0:
            instructions += "Queries:\n"
            for query in self.queries:
                instructions += f"{order}: {query.query}\n"
                order += 1

        return instructions


class DocumentTriageAgent(ConversableAgent):
    """The DocumentTriageAgent is responsible for deciding what type of task to perform from user requests."""

    def __init__(self, llm_config: dict[str, Any]):
        # Add the structured message to the LLM configuration
        structured_config_list = deepcopy(llm_config)
        structured_config_list["response_format"] = DocumentTask

        super().__init__(
            name="DocumentTriageAgent",
            system_message=(
                "You are a document triage agent. "
                "You are responsible for deciding what type of task to perform from a user's request and populating a DocumentTask formatted response. "
                "If the user specifies files or URLs, add them as individual 'ingestions' to DocumentTask. "
                "You can access external websites if given a URL, so put them in as ingestions. "
                "Add the user's questions about the files/URLs as individual 'RAG_QUERY' queries to the 'query' list in the DocumentTask. "
                "Don't make up questions, keep it as concise and close to the user's request as possible."
            ),
            human_input_mode="NEVER",
            llm_config=structured_config_list,
        )


@export_module("autogen.agents.experimental")
class DocAgent(ConversableAgent):
    """
    The DocAgent is responsible for ingest and querying documents.

    Internally, it generates a group of swarm agents to solve tasks.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        llm_config: Optional[dict[str, Any]] = None,
        system_message: Optional[str] = None,
        parsed_docs_path: Optional[Union[str, Path]] = None,
        collection_name: Optional[str] = None,
        query_engine: Optional[RAGQueryEngine] = None,
    ):
        """Initialize the DocAgent.

        Args:
            name (Optional[str]): The name of the DocAgent.
            llm_config (Optional[dict[str, Any]]): The configuration for the LLM.
            system_message (Optional[str]): The system message for the DocAgent.
            parsed_docs_path (Union[str, Path]): The path where parsed documents will be stored.
            collection_name (Optional[str]): The unique name for the data store collection. If omitted, a random name will be used. Populate this to reuse previous ingested data.
            query_engine (Optional[RAGQueryEngine]): The query engine to use for querying documents, defaults to VectorChromaQueryEngine if none provided.
                                                     Use enable_query_citations and implement query_with_citations method to enable citation support. e.g. VectorChromaCitationQueryEngine

        The DocAgent is responsible for generating a group of agents to solve a task.

        The agents that the DocAgent generates are:
        - Triage Agent: responsible for deciding what type of task to perform from user requests.
        - Task Manager Agent: responsible for managing the tasks.
        - Parser Agent: responsible for parsing the documents.
        - Data Ingestion Agent: responsible for ingesting the documents.
        - Query Agent: responsible for answering the user's questions.
        - Error Agent: responsible for returning errors gracefully.
        - Summary Agent: responsible for generating a summary of the user's questions.
        """
        name = name or "DocAgent"
        llm_config = llm_config or {}
        system_message = system_message or DEFAULT_SYSTEM_MESSAGE
        parsed_docs_path = parsed_docs_path or "./parsed_docs"

        # Default Query Engine will be ChromaDB
        if query_engine is None:
            query_engine = VectorChromaQueryEngine(collection_name=collection_name)

        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        self.register_reply([ConversableAgent, None], self.generate_inner_swarm_reply, position=0)

        self._context_variables: dict[str, Any] = {
            "DocumentsToIngest": [],
            "DocumentsIngested": [],
            "QueriesToRun": [],
            "QueryResults": [],
        }

        self._triage_agent = DocumentTriageAgent(llm_config=llm_config)

        def create_error_agent_prompt(agent: ConversableAgent, messages: list[dict[str, Any]]) -> str:
            """Create the error agent prompt, primarily used to update ingested documents for ending"""
            update_ingested_documents()

            return ERROR_MANAGER_SYSTEM_MESSAGE

        self._error_agent = ConversableAgent(
            name=ERROR_MANAGER_NAME,
            system_message=ERROR_MANAGER_SYSTEM_MESSAGE,
            llm_config=llm_config,
            update_agent_state_before_reply=[UpdateSystemMessage(create_error_agent_prompt)],
        )

        def update_ingested_documents() -> None:
            """Updates the list of ingested documents, persisted so we can keep a list over multiple replies"""
            agent_documents_ingested = self._triage_agent.get_context("DocumentsIngested")
            # Update self.documents_ingested with any new documents ingested
            for doc in agent_documents_ingested:
                if doc not in self.documents_ingested:
                    self.documents_ingested.append(doc)

        class TaskInitInfo(BaseModel):
            ingestions: Annotated[list[Ingest], Field(description="List of documents, files, and URLs to ingest")]
            queries: Annotated[list[Query], Field(description="List of queries to run")]

        def initiate_tasks(
            task_init_info: Annotated[TaskInitInfo, "Documents, Files, URLs to ingest and the queries to run"],
            context_variables: Annotated[dict[str, Any], "Context variables"],
        ) -> SwarmResult:
            """Add documents to ingest and queries to answer when received."""
            ingestions = task_init_info.ingestions
            queries = task_init_info.queries

            logger.info("initiate_tasks context_variables", context_variables)
            if "TaskInitiated" in context_variables:
                return SwarmResult(values="Task already initiated", context_variables=context_variables)
            context_variables["DocumentsToIngest"] = ingestions
            context_variables["QueriesToRun"] = [query for query in queries]
            context_variables["TaskInitiated"] = True
            return SwarmResult(
                values="Updated context variables with task decisions",
                context_variables=context_variables,
                agent=TASK_MANAGER_NAME,
            )

        self._task_manager_agent = ConversableAgent(
            name=TASK_MANAGER_NAME,
            system_message=TASK_MANAGER_SYSTEM_MESSAGE,
            llm_config=llm_config,
            functions=[initiate_tasks],
        )

        register_hand_off(
            agent=self._triage_agent,
            hand_to=[
                AfterWork(self._task_manager_agent),
            ],
        )

        self._data_ingestion_agent = DoclingDocIngestAgent(
            llm_config=llm_config,
            query_engine=query_engine,
            parsed_docs_path=parsed_docs_path,
            return_agent_success=TASK_MANAGER_NAME,
            return_agent_error=ERROR_MANAGER_NAME,
        )

        def execute_rag_query(context_variables: dict) -> SwarmResult:  # type: ignore[type-arg]
            """Execute outstanding RAG queries, call the tool once for each outstanding query. Call this tool with no arguments."""
            if len(context_variables["QueriesToRun"]) == 0:
                return SwarmResult(
                    agent=TASK_MANAGER_NAME,
                    values="No queries to run",
                    context_variables=context_variables,
                )

            query = context_variables["QueriesToRun"][0].query
            try:
                if (
                    hasattr(query_engine, "enable_query_citations")
                    and query_engine.enable_query_citations
                    and hasattr(query_engine, "query_with_citations")
                    and callable(query_engine.query_with_citations)
                ):
                    answer_with_citations = query_engine.query_with_citations(query)  # type: ignore[union-attr]
                    answer = answer_with_citations.answer
                    txt_citations = [
                        {
                            "text_chunk": source.node.get_text(),
                            "file_path": source.metadata["file_path"],
                        }
                        for source in answer_with_citations.citations
                    ]
                    logger.info(f"Citations:\n {txt_citations}")
                else:
                    answer = query_engine.query(query)
                    txt_citations = []
                context_variables["QueriesToRun"].pop(0)
                context_variables["CompletedTaskCount"] += 1
                context_variables["QueryResults"].append({"query": query, "answer": answer, "citations": txt_citations})
                return SwarmResult(values=answer, context_variables=context_variables)
            except Exception as e:
                return SwarmResult(
                    agent=ERROR_MANAGER_NAME,
                    values=f"Query failed for '{query}': {e}",
                    context_variables=context_variables,
                )

        self._query_agent = ConversableAgent(
            name="QueryAgent",
            system_message="""
            You are a query agent.
            You answer the user's questions only using the query function provided to you.
            You can only call use the execute_rag_query tool once per turn.
            """,
            llm_config=llm_config,
            functions=[execute_rag_query],
        )

        # Summary agent prompt will include the results of the ingestions and swarms
        def create_summary_agent_prompt(agent: ConversableAgent, messages: list[dict[str, Any]]) -> str:
            """Create the summary agent prompt and updates ingested documents"""
            update_ingested_documents()

            system_message = (
                "You are a summary agent and you provide a summary of all completed tasks and the list of queries and their answers. "
                "Output two sections: 'Ingestions:' and 'Queries:' with the results of the tasks. Number the ingestions and queries. "
                "If there are no ingestions output 'No ingestions', if there are no queries output 'No queries' under their respective sections. "
                "Don't add markdown formatting. "
                "For each query, there is one answer and, optionally, a list of citations."
                "For each citation, it contains two fields: 'text_chunk' and 'file_path'."
                "Format the Query and Answers and Citations as 'Query:\nAnswer:\n\nCitations:'. Add a number to each query if more than one. Use the context below:\n"
                "For each query, output the full citation contents and list them one by one,"
                "format each citation as '\nSource [X] (chunk file_path here):\n\nChunk X:\n(text_chunk here)' and mark a separator between each citation using '\n#########################\n\n'."
                "If there are no citations at all, DON'T INCLUDE ANY mention of citations.\n"
                f"Documents ingested: {agent.get_context('DocumentsIngested')}\n"
                f"Documents left to ingest: {len(agent.get_context('DocumentsToIngest'))}\n"
                f"Queries left to run: {len(agent.get_context('QueriesToRun'))}\n"
                f"Query and Answers and Citations: {agent.get_context('QueryResults')}\n"
            )

            return system_message

        self._summary_agent = ConversableAgent(
            name="SummaryAgent",
            llm_config=llm_config,
            update_agent_state_before_reply=[UpdateSystemMessage(create_summary_agent_prompt)],
        )

        def summary_task(agent: ConversableAgent, messages: list[dict[str, Any]]) -> bool:
            return (
                len(agent.get_context("DocumentsToIngest")) == 0
                and len(agent.get_context("QueriesToRun")) == 0
                and agent.get_context("CompletedTaskCount")
            )

        register_hand_off(
            agent=self._task_manager_agent,
            hand_to=[
                OnContextCondition(  # Go straight to data ingestion agent if we have documents to ingest
                    target=self._data_ingestion_agent,
                    condition=ContextExpression("len(${DocumentsToIngest}) > 0"),
                ),
                OnContextCondition(  # Go to Query agent if we have queries to run (ingestion above run first)
                    target=self._query_agent,
                    condition=ContextExpression("len(${QueriesToRun}) > 0"),
                ),
                OnContextCondition(  # Go to Summary agent if no documents or queries left to run and we have query results
                    target=self._summary_agent,
                    condition=ContextExpression(
                        "len(${DocumentsToIngest}) == 0 and len(${QueriesToRun}) == 0 and len(${QueryResults}) > 0"
                    ),
                ),
                OnCondition(
                    self._summary_agent,
                    "Call this function if all work is done and a summary will be created",
                    available=summary_task,
                ),
                AfterWork(AfterWorkOption.STAY),
            ],
        )

        register_hand_off(
            agent=self._data_ingestion_agent,
            hand_to=[
                AfterWork(self._task_manager_agent),
            ],
        )

        register_hand_off(
            agent=self._query_agent,
            hand_to=[
                AfterWork(self._task_manager_agent),
            ],
        )

        register_hand_off(
            agent=self._summary_agent,
            hand_to=[
                AfterWork(AfterWorkOption.TERMINATE),
            ],
        )

        # The Error Agent always terminates the swarm
        register_hand_off(
            agent=self._error_agent,
            hand_to=[
                AfterWork(AfterWorkOption.TERMINATE),
            ],
        )

        self.register_reply([Agent, None], DocAgent.generate_inner_swarm_reply)

        self.documents_ingested: list[str] = []

    def generate_inner_swarm_reply(
        self,
        messages: Optional[Union[list[dict[str, Any]], str]] = None,
        sender: Optional[Agent] = None,
        config: Optional[OpenAIWrapper] = None,
    ) -> tuple[bool, Optional[Union[str, dict[str, Any]]]]:
        """Reply function that generates the inner swarm reply for the DocAgent."""
        context_variables = {
            "CompletedTaskCount": 0,
            "DocumentsToIngest": [],
            "DocumentsIngested": self.documents_ingested,
            "QueriesToRun": [],
            "QueryResults": [],
        }
        swarm_agents = [
            self._triage_agent,
            self._task_manager_agent,
            self._data_ingestion_agent,
            self._query_agent,
            self._summary_agent,
            self._error_agent,
        ]
        chat_result, context_variables, last_speaker = initiate_swarm_chat(
            initial_agent=self._triage_agent,
            agents=swarm_agents,
            messages=self._get_document_input_message(messages),
            context_variables=context_variables,
            after_work=AfterWorkOption.TERMINATE,
        )
        if last_speaker == self._error_agent:
            # If we finish with the error agent, we return their message which contains the error
            return True, chat_result.summary
        if last_speaker != self._summary_agent:
            # If the swarm finished but not with the summary agent, we assume something has gone wrong with the flow
            return True, DEFAULT_ERROR_SWARM_MESSAGE

        return True, chat_result.summary

    def _get_document_input_message(self, messages: Optional[Union[list[dict[str, Any]], str]]) -> str:  # type: ignore[type-arg]
        """Gets and validates the input message(s) for the document agent."""
        if isinstance(messages, str):
            return messages
        elif (
            isinstance(messages, list)
            and len(messages) > 0
            and "content" in messages[-1]
            and isinstance(messages[-1]["content"], str)
        ):
            return messages[-1]["content"]
        else:
            raise NotImplementedError("Invalid messages format. Must be a list of messages or a string.")
