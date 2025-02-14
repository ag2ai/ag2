# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from autogen import Agent, ConversableAgent
from autogen.agentchat.contrib.swarm_agent import (
    AfterWork,
    AfterWorkOption,
    OnCondition,
    SwarmResult,
    initiate_swarm_chat,
    register_hand_off,
)
from autogen.agents.experimental.document_agent.docling_query_engine import DoclingMdQueryEngine
from autogen.oai.client import OpenAIWrapper

from .docling_doc_ingest_agent import DoclingDocIngestAgent

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

    Please don't output anything else.
    """
DEFAULT_ERROR_SWARM_MESSAGE: str = """
Document Agent failed to perform task.
"""


class QueryType(Enum):
    RAG_QUERY = "RAG_QUERY"
    COMMON_QUESTION = "COMMON_QUESTION"


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
    def __init__(self, llm_config: Dict[str, Any]):
        # Add the structured message to the LLM configuration
        structured_config_list = deepcopy(llm_config)
        for config in structured_config_list["config_list"]:
            config["response_format"] = DocumentTask

        super().__init__(
            name="DocumentTriageAgent",
            system_message=(
                "You are a document triage agent."
                "You are responsible for deciding what type of task to perform from user requests."
                "When user uploads new documents or provide links of documents, you should add Ingest task to DocumentTask."
                "When user asks common questions, you should add 'COMMON_QUESTION' Query task to DocumentTask."
                "When user asks questions about information from existing documents, you add 'RAG_QUERY' Query task to DocumentTask."
            ),
            human_input_mode="NEVER",
            llm_config=structured_config_list,
        )


class DocumentAgent(ConversableAgent):
    def __init__(
        self,
        name: str = "Document_Agent",
        llm_config: Dict[str, Any] = {},
        system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
    ):
        """
        Initialize the DocumentAgent.

        Args:
        name (str): The name of the DocumentAgent.
        llm_config (Dict[str, Any]): The configuration for the LLM.
        system_message (str): The system message for the ChatCompletion inference.
        **kwargs: Additional keyword arguments to pass to the parent class.

        The DocumentAgent is responsible for generating a group of agents to solve a task.

        The agents that the DocumentAgent generates are:
        - Triage Agent: responsible for deciding what type of task to perform from user requests.
        - Task Manager Agent: responsible for managing the tasks.
        - Parser Agent: responsible for parsing the documents.
        - Data Ingestion Agent: responsible for ingesting the documents.
        - Query Agent: responsible for answering the user's questions.
        - Summary Agent: responsible for generating a summary of the user's questions.
        """
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        self.register_reply([ConversableAgent, None], self.generate_inner_swarm_reply, position=0)

        self._context_variables = {
            "DocumentsToIngest": [],
            "QueriesToRun": [],
            "QueryResults": [],
        }

        self._triage_agent = DocumentTriageAgent(llm_config=llm_config)

        def initiate_tasks(
            ingestions: list[Ingest],
            queries: list[Query],
            context_variables: dict,  # type: ignore[type-arg]
        ) -> SwarmResult:
            print("initiate_tasks context_variables", context_variables)
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

        query_engine = DoclingMdQueryEngine()
        self._data_ingestion_agent = DoclingDocIngestAgent(llm_config=llm_config, query_engine=query_engine)

        def execute_rag_query(context_variables: dict) -> SwarmResult:  # type: ignore[type-arg]
            query = context_variables["QueriesToRun"][0]["query"]
            answer = query_engine.query(query)
            context_variables["QueriesToRun"].pop(0)
            context_variables["CompletedTaskCount"] += 1
            context_variables["QueryResults"].append({"query": query, "result": answer})
            return SwarmResult(values=answer, context_variables=context_variables)

        self._query_agent = ConversableAgent(
            name="QueryAgent",
            system_message="You are a query agent. You answer the user's questions only using the query function provided to you.",
            llm_config=llm_config,
            functions=[execute_rag_query],
        )

        self._summary_agent = ConversableAgent(
            name="SummaryAgent",
            system_message="You are a summary agent. You would generate a summary of all completed tasks and  answer the user's questions.",
            llm_config=llm_config,
        )

        def has_ingest_tasks(agent: ConversableAgent, messages: List[Dict[str, Any]]) -> bool:
            print("context_variables", agent._context_variables)
            return len(agent.get_context("DocumentsToIngest")) > 0

        def has_query_tasks(agent: ConversableAgent, messages: List[Dict[str, Any]]) -> bool:
            print("context_variables", agent._context_variables)
            if len(agent.get_context("DocumentsToIngest")) > 0:
                return False
            return len(agent.get_context("QueriesToRun")) > 0

        def summary_task(agent: ConversableAgent, messages: List[Dict[str, Any]]) -> bool:
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
                    "If there are no DocumentsToIngest or QueriesToRun in context variables, transfer to summary_agent",
                    available=summary_task,
                ),
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
        self.register_reply([Agent, None], DocumentAgent.generate_inner_swarm_reply)

    def generate_inner_swarm_reply(
        self,
        messages: Union[list[dict], str, None] = None,  # type: ignore[type-arg]
        sender: Optional[Agent] = None,
        config: Optional[OpenAIWrapper] = None,
    ) -> tuple[bool, Union[str, dict, None]]:  # type: ignore[type-arg]
        context_variables = {
            "CompletedTaskCount": 0,
            "DocumentsToIngest": [],
            "QueriesToRun": [],
            "QueryResults": [],
        }
        swarm_agents = [
            self._triage_agent,
            self._task_manager_agent,
            self._data_ingestion_agent,
            self._query_agent,
            self._summary_agent,
        ]
        chat_result, context_variables, last_speaker = initiate_swarm_chat(
            initial_agent=self._triage_agent,
            agents=swarm_agents,
            messages=self._get_document_input_message(messages),
            context_variables=context_variables,
            after_work=AfterWorkOption.TERMINATE,
        )
        if last_speaker != self._summary_agent:
            return False, DEFAULT_ERROR_SWARM_MESSAGE
        return True, chat_result.summary

    def _init_swarm_agents(self, agents: list[ConversableAgent]) -> None:
        for agent in agents:
            agent.reset()

    def _get_document_input_message(self, messages: Union[list[dict], str, None]) -> str:  # type: ignore[type-arg]
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
