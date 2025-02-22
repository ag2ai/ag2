# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from .... import Agent, ConversableAgent, UpdateSystemMessage
from ....agentchat.contrib.swarm_agent import (
    AfterWork,
    AfterWorkOption,
    OnCondition,
    SwarmResult,
    initiate_swarm_chat,
    register_hand_off,
)
from ....doc_utils import export_module
from ....oai.client import OpenAIWrapper
from .docling_doc_ingest_agent import DoclingDocIngestAgent
from .docling_query_engine import DoclingMdQueryEngine

__all__ = ["DocAgent"]

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_MESSAGE = """
    You are a document agent.
    You are given a list of documents to ingest and a list of queries to perform.
    You are responsible for ingesting the documents and answering the queries.
"""
TASK_MANAGER_NAME = "TaskManagerAgent"
TASK_MANAGER_SYSTEM_MESSAGE = """
    You are a task manager agent. You would only do the following 2 tasks:
    1. You update the context variables based on the task decisions (DocumentTask) from the DocumentTriageAgent.
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
    2. You would hand off control to the appropriate agent based on the context variables.

    Put all file paths and URLs into the ingestions. A http/https URL is also a valid path and should be ingested.

    Please don't output anything else.

    Use the initiate_tasks tool to incorporate all ingestions and queries. Don't call it again until new ingestions or queries are raised.
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


class DocumentTriageAgent(ConversableAgent):
    """The DocumentTriageAgent is responsible for deciding what type of task to perform from user requests."""

    def __init__(self, llm_config: dict[str, Any]):
        # Add the structured message to the LLM configuration
        structured_config_list = deepcopy(llm_config)
        structured_config_list["response_format"] = DocumentTask

        super().__init__(
            name="DocumentTriageAgent",
            system_message=(
                "You are a document triage agent."
                "You are responsible for deciding what type of task to perform from a user's request and populating a DocumentTask formatted response."
                "If the user specifies files or URLs, add them as individual 'ingestions' to DocumentTask."
                "Add the user's questions about the files/URLs as individual 'RAG_QUERY' queries to the 'query' list in the DocumentTask. Don't make up questions, keep it as concise and close to the user's request as possible."
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
    ):
        """Initialize the DocAgent.

        Args:
            name (Optional[str]): The name of the DocAgent.
            llm_config (Optional[dict[str, Any]]): The configuration for the LLM.
            system_message (Optional[str]): The system message for the DocAgent.
            parsed_docs_path (Union[str, Path]): The path where parsed documents will be stored.
            collection_name (Optional[str]): The unique name for the data store collection. If omitted, a random name will be used. Populate this to reuse previous ingested data.

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

        def initiate_tasks(
            ingestions: list[Ingest],
            queries: list[Query],
            context_variables: dict[str, Any],
        ) -> SwarmResult:
            """Initiate all document and query tasks by storing them in context for future reference."""
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

        query_engine = DoclingMdQueryEngine(collection_name=collection_name)
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

            query = context_variables["QueriesToRun"][0]["query"]
            try:
                answer = query_engine.query(query)
                context_variables["QueriesToRun"].pop(0)
                context_variables["CompletedTaskCount"] += 1
                context_variables["QueryResults"].append({"query": query, "answer": answer})
                return SwarmResult(values=answer, context_variables=context_variables)
            except Exception as e:
                return SwarmResult(
                    agent=ERROR_MANAGER_NAME,
                    values=f"Query failed for '{query}': {e}",
                    context_variables=context_variables,
                )

        self._query_agent = ConversableAgent(
            name="QueryAgent",
            system_message="You are a query agent. You answer the user's questions only using the query function provided to you. You can only call use the execute_rag_query tool once per turn.",
            llm_config=llm_config,
            functions=[execute_rag_query],
        )

        # Summary agent prompt will include the results of the ingestions and swarms
        def create_summary_agent_prompt(agent: ConversableAgent, messages: list[dict[str, Any]]) -> str:
            """Create the summary agent prompt and updates ingested documents"""
            update_ingested_documents()

            system_message = (
                "You are a summary agent and you provide a summary of all completed tasks and the list of queries and their answers. "
                "Format the Query and Answers as 'Query:\nAnswer:'. Add a number to each query if more than one. Use the context below:\n"
                f"Documents ingested: {agent.get_context('DocumentsIngested')}\n"
                f"Documents left to ingest: {len(agent.get_context('DocumentsToIngest'))}\n"
                f"Queries left to run: {len(agent.get_context('QueriesToRun'))}\n"
                f"Query and Answers: {agent.get_context('QueryResults')}\n"
            )

            return system_message

        self._summary_agent = ConversableAgent(
            name="SummaryAgent",
            llm_config=llm_config,
            update_agent_state_before_reply=[UpdateSystemMessage(create_summary_agent_prompt)],
        )

        def has_ingest_tasks(agent: ConversableAgent, messages: list[dict[str, Any]]) -> bool:
            logger.debug("has_ingest_tasks context_variables:", agent._context_variables)
            return len(agent.get_context("DocumentsToIngest")) > 0

        def has_query_tasks(agent: ConversableAgent, messages: list[dict[str, Any]]) -> bool:
            logger.debug("has_query_tasks context_variables:", agent._context_variables)
            if len(agent.get_context("DocumentsToIngest")) > 0:
                return False
            return len(agent.get_context("QueriesToRun")) > 0

        def summary_task(agent: ConversableAgent, messages: list[dict[str, Any]]) -> bool:
            return (
                len(agent.get_context("DocumentsToIngest")) == 0
                and len(agent.get_context("QueriesToRun")) == 0
                and agent.get_context("CompletedTaskCount")
            )

        register_hand_off(
            agent=self._task_manager_agent,
            hand_to=[
                OnCondition(
                    self._data_ingestion_agent,
                    "If there are any DocumentsToIngest in context variables, transfer to data ingestion agent",
                    available=has_ingest_tasks,
                ),
                OnCondition(
                    self._query_agent,
                    "If there are any QueriesToRun in context variables and no DocumentsToIngest, transfer to query_agent",
                    available=has_query_tasks,
                ),
                OnCondition(
                    self._summary_agent,
                    "Call this function as work is done and a summary will be created",
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
