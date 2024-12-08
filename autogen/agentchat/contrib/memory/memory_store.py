# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from graphrag_sdk import Ontology
from graphrag_sdk.models import GenerativeModel
from graphrag_sdk.models.openai import OpenAiGenerativeModel

from autogen.agentchat.contrib.graph_rag.document import Document as GraphDocument
from autogen.agentchat.contrib.graph_rag.falkor_graph_query_engine import FalkorGraphQueryEngine
from autogen.agentchat.contrib.vectordb.base import Document, VectorDBFactory


# Common patterns for all stores
@dataclass
class BaseDocument:
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = field()
    metadata: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class StaticMemoryStore:
    """Manages immutable and appendable documents"""

    def __init__(self, path: str = "./static_memory"):
        self.base_path = Path(path)
        self.static_path = self.base_path / "static"
        self.logs_path = self.base_path / "logs"
        self.static_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)

        # Load/create registry
        self.registry_path = self.base_path / "registry.json"
        if self.registry_path.exists():
            self.registry = json.loads(self.registry_path.read_text())
        else:
            self.registry = {"static": {}, "logs": {}}
            self._save_registry()

    def _save_registry(self) -> None:
        self.registry_path.write_text(json.dumps(self.registry, indent=2))

    def store_static(self, content: str, doc_id: str, metadata: Optional[Dict] = None) -> BaseDocument:
        """Store immutable document"""
        if doc_id in self.registry["static"]:
            raise ValueError(f"Document {doc_id} already exists")

        doc = BaseDocument(id=doc_id, content=content, metadata=metadata)
        (self.static_path / doc_id).write_text(content)
        self.registry["static"][doc_id] = doc.metadata
        self._save_registry()
        return doc

    def create_log(self, content: str, doc_id: str, metadata: Optional[Dict] = None) -> BaseDocument:
        """Create appendable log document"""
        if doc_id in self.registry["logs"]:
            raise ValueError(f"Log {doc_id} already exists")

        doc = BaseDocument(id=doc_id, content=content, metadata=metadata)
        with (self.logs_path / doc_id).open("w") as f:
            json.dump({"content": content, "timestamp": doc.timestamp.isoformat()}, f)
            f.write("\n")

        self.registry["logs"][doc_id] = doc.metadata
        self._save_registry()
        return doc

    def append_log(self, doc_id: str, content: str) -> None:
        """Append to existing log"""
        if doc_id not in self.registry["logs"]:
            raise ValueError(f"No log found: {doc_id}")

        with (self.logs_path / doc_id).open("a") as f:
            json.dump({"content": content, "timestamp": datetime.now().isoformat()}, f)
            f.write("\n")

    def get(self, doc_id: str) -> Optional[BaseDocument]:
        """Get document by ID"""
        if doc_id in self.registry["static"]:
            content = (self.static_path / doc_id).read_text()
            return BaseDocument(id=doc_id, content=content, metadata=self.registry["static"][doc_id])

        if doc_id in self.registry["logs"]:
            entries = [json.loads(line) for line in (self.logs_path / doc_id).open()]
            content = "\n".join(f"[{e['timestamp']}] {e['content']}" for e in entries)
            return BaseDocument(id=doc_id, content=content, metadata=self.registry["logs"][doc_id])

        return None

    def get_by_metadata(self, filter_dict: Dict) -> List[BaseDocument]:
        """Find documents by metadata"""
        matches = []
        for doc_id, metadata in {**self.registry["static"], **self.registry["logs"]}.items():
            if all(metadata.get(k) == v for k, v in filter_dict.items()):
                if doc := self.get(doc_id):
                    matches.append(doc)
        return matches

    def delete(self, doc_id: str) -> bool:
        """Delete document"""
        if doc_id in self.registry["static"]:
            (self.static_path / doc_id).unlink(missing_ok=True)
            del self.registry["static"][doc_id]
            self._save_registry()
            return True

        if doc_id in self.registry["logs"]:
            (self.logs_path / doc_id).unlink(missing_ok=True)
            del self.registry["logs"][doc_id]
            self._save_registry()
            return True

        return False


class SemanticMemoryStore:
    """Vector-based semantic memory using autogen's VectorDB"""

    def __init__(
        self, db_type: str = "chroma", collection: str = "semantic_memories", path: str = "./semantic_memory", **kwargs
    ):
        self.db = VectorDBFactory.create_vector_db(db_type=db_type, path=path, **kwargs)
        self.collection = self.db.create_collection(collection, get_or_create=True)
        self.collection_name = collection

    def store(self, content: str, metadata: Optional[Dict] = None) -> str:
        """Store new memory"""
        doc = Document(id=str(uuid4()), content=content, metadata=metadata or {}, embedding=None)
        self.db.insert_docs([doc], self.collection_name)
        return doc["id"]

    def find_similar(self, query: str, limit: int = 5, threshold: float = 0.7) -> List[Tuple[Document, float]]:
        """Find similar memories"""
        results = self.db.retrieve_docs(
            queries=[query], collection_name=self.collection_name, n_results=limit, distance_threshold=threshold
        )
        return results[0] if results else []

    def get(self, ids: List[str]) -> List[Document]:
        """Get memories by IDs"""
        return self.db.get_docs_by_ids(ids, self.collection_name)

    def update(self, memory_id: str, content: Optional[str] = None, metadata: Optional[Dict] = None) -> None:
        """Update memory"""
        if docs := self.get([memory_id]):
            doc = docs[0]
            if content is not None:
                doc["content"] = content
            if metadata is not None:
                doc["metadata"] = {**(doc["metadata"] or {}), **metadata}
            self.db.update_docs([doc], self.collection_name)
        else:
            raise ValueError(f"Memory {memory_id} not found")

    def delete(self, ids: List[str]) -> None:
        """Delete memories"""
        self.db.delete_docs(ids, self.collection_name)


class GraphMemoryStore:
    """Graph-based memory using FalkorDB"""

    def __init__(
        self,
        name: str = "memory_graph",
        host: str = "127.0.0.1",
        port: int = 6379,
        username: Optional[str] = None,
        password: Optional[str] = None,
        model: Optional[GenerativeModel] = None,
        ontology: Optional[Ontology] = None,
    ):
        self.engine = FalkorGraphQueryEngine(
            name=name,
            host=host,
            port=port,
            username=username,
            password=password,
            model=model or OpenAiGenerativeModel("gpt-4"),
            ontology=ontology,
        )

    def init_graph(self, nodes: List[Dict[str, str]]) -> None:
        """Initialize graph from nodes"""
        docs = [
            GraphDocument(path_or_url=node["path"], content=node["content"], metadata=node.get("metadata", {}))
            for node in nodes
        ]
        self.engine.init_db(docs)

    def query(self, query: str, n_results: int = 1, **kwargs):
        """Query graph"""
        return self.engine.query(query, n_results, **kwargs)
