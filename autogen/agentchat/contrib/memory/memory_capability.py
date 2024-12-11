# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Dict, List, Optional, Tuple

from autogen import Agent, ConversableAgent

from ..capabilities.agent_capability import AgentCapability
from ..graph_rag.document import Document, DocumentType
from ..text_analyzer_agent import TextAnalyzerAgent
from .memory_store import MemoryOperation, MemoryRequest, MemoryStore
from .models import CommonModel, DomainModel, GeneralModel, MemoryAnalysisModel

logger = logging.getLogger(__name__)


class MemoryCapability(AgentCapability):
    """Memory capability for agents to interact with different types of knowledge storage"""

    def __init__(self, store: Optional[MemoryStore] = None, llm_config: Optional[Dict] = None):
        """Initialize memory capability with store and analyzer"""
        self.store = store or MemoryStore()

        memory_analysis_prompt = """Analyze messages to determine required memory operations.
        Return a structured analysis specifying which memory operations are needed.

        Available Memory Types:

        1. Common Knowledge (common):
           - Purpose: Store reusable reference information
           - Use for: Documentation, procedures, guides
           - Operations: add/update/delete/query
           - Fields: category, content, tags[], metadata{}

        2. Domain Knowledge (domain):
           - Purpose: Store structured relationship information
           - Use for: System architecture, workflows, entity relationships
           - Operations: add/update/query
           - Fields: content[], mode(replace/extend), query

        3. General Memory (general):
           - Purpose: Store searchable temporal information
           - Use for: Meeting notes, events, conversations
           - Operations: add/update/delete/query
           - Fields: content, metadata{}, memory_id, query_filter{}

        Only populate relevant sections and operations.
        Other sections should be null.

        Examples:

        1. Adding Documentation:
        {
          "common": {
            "operation": "add",
            "category": "api_docs",
            "content": "API documentation...",
            "tags": ["api", "reference"]
          }
        }

        2. Querying Domain:
        {
          "domain": {
            "operation": "query",
            "query": "How does component X interact with Y?"
          }
        }

        3. Storing Event:
        {
          "general": {
            "operation": "add",
            "content": "Team meeting discussion...",
            "metadata": {"type": "meeting_notes"}
          }
        }
        """

        self.analyzer = TextAnalyzerAgent(
            name="memory_analyzer",
            system_message=memory_analysis_prompt,
            human_input_mode="NEVER",
            llm_config=llm_config,
            response_format=MemoryAnalysisModel,
        )

    async def _process_with_memory(
        self,
        recipient: ConversableAgent,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        **kwargs,
    ) -> Tuple[bool, Optional[str]]:
        """Process memory operations from messages"""
        if not messages:
            return False, None

        try:
            message = messages[-1]["content"]

            # Get analysis as MemoryAnalysisModel
            analysis = await self.analyzer.a_generate_reply(messages=[{"role": "user", "content": message}])

            if not any([analysis.common, analysis.domain, analysis.general]):
                return False, None

            results = {}

            # Handle each memory type if specified
            if analysis.common:
                results.update(await self._handle_common_operation(analysis.common))
            if analysis.domain:
                results.update(await self._handle_domain_operation(analysis.domain))
            if analysis.general:
                results.update(await self._handle_general_operation(analysis.general))

            context = self._build_context(results)
            if not context:
                return False, None

            response = await recipient.generate_oai_reply(
                messages=[
                    {
                        "role": "user",
                        "content": f"Question/Input: {message}\n\nAvailable Context:\n{context}\n\nRespond based on this context.",
                    }
                ]
            )

            return True, response

        except Exception as e:
            logger.error(f"Error in memory processing: {e}")
            return False, str(e)

    async def _handle_common_operation(self, common: CommonModel) -> Dict[str, Any]:
        """Handle common knowledge operations using direct model"""
        results = {}

        try:
            request = MemoryRequest(
                operation=MemoryOperation(common.operation),
                documents=(
                    [Document(doctype=DocumentType.TEXT, content=common.content, id=common.filename)]
                    if common.content or common.filename
                    else None
                ),
                metadata={"category": common.category, "tags": common.tags, **(common.metadata or {})},
            )

            result = self.store.operate_common_knowledge(request)

            if common.operation == "query":
                results["common_data"] = result
            elif common.operation == "add":
                results["common_id"] = result.get("document_id")

        except Exception as e:
            logger.error(f"Error in common knowledge operation: {e}")
            results["error"] = str(e)

        return results

    async def _handle_domain_operation(self, domain: DomainModel) -> Dict[str, Any]:
        """Handle domain knowledge operations using direct model"""
        results = {}

        try:
            request = MemoryRequest(
                operation=MemoryOperation(domain.operation),
                documents=(
                    [Document(doctype=DocumentType.TEXT, content=content) for content in domain.content]
                    if domain.content
                    else None
                ),
                mode=domain.mode,
                query=domain.query,
            )

            result = self.store.operate_domain_knowledge(request)

            if domain.operation == "query":
                results["domain_data"] = result.get("answer")
                results["source_documents"] = result.get("source_documents")

        except Exception as e:
            logger.error(f"Error in domain knowledge operation: {e}")
            results["error"] = str(e)

        return results

    async def _handle_general_operation(self, general: GeneralModel) -> Dict[str, Any]:
        """Handle general memory operations using direct model"""
        results = {}

        try:
            request = MemoryRequest(
                operation=MemoryOperation(general.operation),
                content=general.content,
                metadata=general.metadata,
                memory_id=general.memory_id,
                query_filter=general.query_filter,
            )

            result = self.store.operate_general_knowledge(request)

            if general.operation == "query":
                results["general_data"] = result.get("matches", [])
            elif general.operation == "add":
                results["general_id"] = result.get("memory_id")

        except Exception as e:
            logger.error(f"Error in general memory operation: {e}")
            results["error"] = str(e)

        return results

    def _build_context(self, results: Dict[str, Any]) -> str:
        """Build context from operation results"""
        parts = []

        if "common_data" in results:
            parts.extend(
                ["Reference Knowledge:", *[f"- {data['content']}" for _, data in results["common_data"].items()], ""]
            )

        if "domain_data" in results:
            parts.extend(
                [
                    "Domain Knowledge:",
                    str(results["domain_data"]),
                    "Source Documents:",
                    *[f"- {doc.content}" for doc in results.get("source_documents", [])],
                    "",
                ]
            )

        if "general_data" in results:
            parts.extend(["Related Memories:", *[f"- {memory.content}" for memory in results["general_data"]], ""])

        if "error" in results:
            parts.extend(["Operation Error:", results["error"], ""])

        return "\n".join(parts)
