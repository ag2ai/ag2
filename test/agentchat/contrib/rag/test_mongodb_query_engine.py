# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# !/usr/bin/env python3 -m pytest

import os

import pytest
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# Import the class and constant from your module.
from autogen.agentchat.contrib.rag.mongodb_query_engine import (
    DEFAULT_COLLECTION_NAME,
    MongoDBQueryEngine,
)
from autogen.agentchat.contrib.vectordb.base import VectorDBFactory
from autogen.import_utils import skip_on_missing_imports

# ----- Fake classes for simulating MongoDB behavior ----- #


@pytest.mark.openai
@skip_on_missing_imports(["pymongo", "openai", "llama_index"], "rag")
class FakeDBExists:
    def list_collection_names(self):
        return [DEFAULT_COLLECTION_NAME]

    def __getitem__(self, key):
        return {}  # Return a dummy collection


class FakeDBNoCollection:
    def list_collection_names(self):
        return []

    def __getitem__(self, key):
        return {}  # Return a dummy collection


class FakeMongoClientExists:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.admin = self

    def command(self, cmd):
        if cmd == "ping":
            return {"ok": 1}
        raise Exception("Ping failed")

    def __getitem__(self, name):
        return FakeDBExists()


class FakeMongoClientNoCollection:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.admin = self

    def command(self, cmd):
        if cmd == "ping":
            return {"ok": 1}
        raise Exception("Ping failed")

    def __getitem__(self, name):
        return FakeDBNoCollection()


# ----- Fake vector DB and index implementations ----- #


class FakeVectorDB:
    def __init__(self, client):
        self.client = client


class FakeIndex:
    def __init__(self, docs=None):
        self.docs = docs or []

    def as_chat_engine(self, llm):
        # Return a fake chat engine with a query method.
        class FakeChatEngine:
            def query(self, question):
                return f"Answer to {question}"

        return FakeChatEngine()

    def insert(self, doc):
        self.docs.append(doc)


# Fake MongoDBAtlasVectorSearch so that no real collection creation is attempted.
class FakeMongoDBAtlasVectorSearch:
    def __init__(self, mongodb_client, db_name, collection_name):
        self.client = mongodb_client  # so that admin.command("ping") works.
        self.db_name = db_name
        self.collection_name = collection_name
        self.stores_text = True  # Added attribute to mimic real behavior.


# Fake create_vector_db function to be used in place of the factory.
def fake_create_vector_db(
    db_type, connection_string, database_name, index_name, embedding_function, collection_name, overwrite
):
    # Choose a fake MongoClient based on the connection string.
    if "exists" in connection_string:
        client = FakeMongoClientExists(connection_string)
    else:
        client = FakeMongoClientNoCollection(connection_string)
    return FakeVectorDB(client)


# ----- Pytest tests ----- #


def test_connect_db_no_collection(monkeypatch):
    """
    Test connect_db when the target collection does not exist.
    It should catch the error and return False.
    """
    monkeypatch.setattr("autogen.agentchat.contrib.rag.mongodb_query_engine.MongoClient", FakeMongoClientNoCollection)
    monkeypatch.setattr(VectorDBFactory, "create_vector_db", fake_create_vector_db)
    engine = MongoDBQueryEngine(
        connection_string="dummy_no_collection", database_name="vector_db", collection_name=DEFAULT_COLLECTION_NAME
    )
    result = engine.connect_db()
    assert result is False


def test_connect_db_existing(monkeypatch):
    """
    Test connect_db when the collection exists.
    It should succeed and return True.
    """
    monkeypatch.setattr("autogen.agentchat.contrib.rag.mongodb_query_engine.MongoClient", FakeMongoClientExists)
    # Override MongoDBAtlasVectorSearch with our fake.
    monkeypatch.setattr(
        "autogen.agentchat.contrib.rag.mongodb_query_engine.MongoDBAtlasVectorSearch", FakeMongoDBAtlasVectorSearch
    )
    monkeypatch.setattr(VectorDBFactory, "create_vector_db", fake_create_vector_db)
    # Override from_vector_store to accept keyword arguments.
    monkeypatch.setattr(VectorStoreIndex, "from_vector_store", lambda vs, **kwargs: FakeIndex())
    engine = MongoDBQueryEngine(
        connection_string="dummy_exists", database_name="vector_db", collection_name=DEFAULT_COLLECTION_NAME
    )
    result = engine.connect_db()
    assert result is True


def test_init_db_existing_collection(monkeypatch):
    """
    Test init_db when the collection already exists.
    It should raise an error internally and return False.
    """
    monkeypatch.setattr("autogen.agentchat.contrib.rag.mongodb_query_engine.MongoClient", FakeMongoClientExists)
    engine = MongoDBQueryEngine(
        connection_string="dummy_exists", database_name="vector_db", collection_name=DEFAULT_COLLECTION_NAME
    )
    # Use a dummy document name instead of an absolute path.
    result = engine.init_db(new_doc_paths=["dummy_doc.md"])
    # Since the collection exists, init_db should return False.
    assert result is False


def test_init_db_no_documents(monkeypatch):
    """
    Test init_db when no documents are provided.
    It should log a warning and return False.
    """
    monkeypatch.setattr("autogen.agentchat.contrib.rag.mongodb_query_engine.MongoClient", FakeMongoClientNoCollection)
    monkeypatch.setattr(VectorDBFactory, "create_vector_db", fake_create_vector_db)
    # Override load_data to return an empty list.
    monkeypatch.setattr(SimpleDirectoryReader, "load_data", lambda self: [])
    engine = MongoDBQueryEngine(
        connection_string="dummy_no_collection", database_name="vector_db", collection_name=DEFAULT_COLLECTION_NAME
    )
    result = engine.init_db(new_doc_paths=[])
    assert result is False


def test_init_db_success(monkeypatch):
    """
    Test successful initialization of the database.
    It should load documents and build the index.
    """
    monkeypatch.setattr("autogen.agentchat.contrib.rag.mongodb_query_engine.MongoClient", FakeMongoClientNoCollection)
    monkeypatch.setattr(
        "autogen.agentchat.contrib.rag.mongodb_query_engine.MongoDBAtlasVectorSearch", FakeMongoDBAtlasVectorSearch
    )
    monkeypatch.setattr(VectorDBFactory, "create_vector_db", fake_create_vector_db)
    # Simulate document loading returning two dummy docs.
    monkeypatch.setattr(SimpleDirectoryReader, "load_data", lambda self: ["doc1", "doc2"])
    # Override from_documents to return a FakeIndex containing the docs.
    monkeypatch.setattr(VectorStoreIndex, "from_documents", lambda docs, **kwargs: FakeIndex(docs))
    engine = MongoDBQueryEngine(
        connection_string="dummy_no_collection", database_name="vector_db", collection_name=DEFAULT_COLLECTION_NAME
    )
    # Use a dummy document name.
    result = engine.init_db(new_doc_paths=["dummy_doc.md"])
    assert result is True
    # Our fake loader returns ["doc1", "doc2"]
    assert engine.index.docs == ["doc1", "doc2"]


def test_add_records(monkeypatch):
    """
    Test that add_records loads documents and inserts them into the index.
    """
    fake_index = FakeIndex()
    engine = MongoDBQueryEngine(
        connection_string="dummy", database_name="vector_db", collection_name=DEFAULT_COLLECTION_NAME
    )
    engine.index = fake_index
    # Override __init__ of SimpleDirectoryReader to bypass file existence checks.
    monkeypatch.setattr(
        SimpleDirectoryReader, "__init__", lambda self, input_files: setattr(self, "input_files", input_files)
    )
    # Override load_data to return dummy records.
    monkeypatch.setattr(SimpleDirectoryReader, "load_data", lambda self: ["record1", "record2"])
    # Force os.path.exists to always return True so that file existence checks pass.
    monkeypatch.setattr(os.path, "exists", lambda path: True)
    engine.add_records(new_doc_paths_or_urls=["dummy_path"])
    assert fake_index.docs == ["record1", "record2"]


def test_query(monkeypatch):
    """
    Test that query returns the expected response from the fake chat engine.
    """
    fake_index = FakeIndex()
    engine = MongoDBQueryEngine(
        connection_string="dummy", database_name="vector_db", collection_name=DEFAULT_COLLECTION_NAME
    )
    engine.index = fake_index
    answer = engine.query("What is testing?", llm="dummy_llm")
    assert answer == "Answer to What is testing?"
