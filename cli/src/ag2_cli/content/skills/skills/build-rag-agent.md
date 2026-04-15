---
name: build-rag-agent
description: Build a Retrieval-Augmented Generation (RAG) agent using AG2's RetrieveUserProxyAgent with vector database support. Use when the user wants agents that can query documents or knowledge bases.
---

# Build RAG Agent

You are an expert at building AG2 RAG (Retrieval-Augmented Generation) workflows. When the user wants to build a document Q&A or knowledge-base agent:

## 1. Choose the Right Approach

Ask the user:
- What documents do they have? (PDFs, text files, web pages, etc.)
- How large is the corpus? (Small → ChromaDB, Large → Qdrant/pgvector)
- Do they need graph-based retrieval? → Graph RAG with Neo4j/FalkorDB

## 2. Basic RAG Agent (ChromaDB)

```python
import os
from autogen import ConversableAgent, LLMConfig
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

llm_config = LLMConfig(
    {"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}
)

assistant = ConversableAgent(
    name="assistant",
    system_message="You answer questions based on the provided context. If the context doesn't contain the answer, say so.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

rag_proxy = RetrieveUserProxyAgent(
    name="rag_proxy",
    human_input_mode="NEVER",
    retrieve_config={
        "task": "qa",                            # "qa" for Q&A, "code" for code search
        "docs_path": ["./docs/"],                # Path to documents
        "collection_name": "my_collection",
        "chunk_token_size": 2000,
        "model": "gpt-4o-mini",                  # For embedding
        "get_or_create": True,                   # Reuse existing collection
    },
    code_execution_config=False,
)

result = await rag_proxy.a_run(
    assistant,
    message=rag_proxy.message_generator,
    problem="What are the main findings in the report?",
)
await result.process()
```

Requires: `pip install ag2[openai,rag]`

## 3. RAG Config Options

```python
retrieve_config = {
    # Task type
    "task": "qa",                    # "qa", "code", or "default"

    # Document sources
    "docs_path": ["./docs/", "https://example.com/page"],  # Local dirs, files, or URLs

    # Chunking
    "chunk_token_size": 2000,        # Tokens per chunk
    "chunk_mode": "multi_lines",     # "multi_lines" or "one_line"
    "must_break_at_empty_line": True, # Break chunks at empty lines

    # Collection
    "collection_name": "my_docs",
    "get_or_create": True,           # Reuse existing collection if available

    # Retrieval
    "n_results": 5,                  # Number of chunks to retrieve
    "distance_threshold": -1,        # Max distance (-1 for no limit)

    # Embedding
    "model": "gpt-4o-mini",          # Model for embeddings
    "embedding_model": "all-MiniLM-L6-v2",  # Custom embedding model

    # Customization
    "customized_prompt": None,       # Custom prompt template
    "customized_answer_prefix": "",  # Prefix for answers
    "update_context": True,          # Update context with new retrievals
}
```

## 4. RAG with Group Chat

```python
from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import run_group_chat
from autogen.agentchat.group.patterns import AutoPattern
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

llm_config = LLMConfig(
    {"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}
)

rag_proxy = RetrieveUserProxyAgent(
    name="rag_proxy",
    human_input_mode="NEVER",
    retrieve_config={
        "task": "qa",
        "docs_path": ["./docs/"],
        "get_or_create": True,
    },
    code_execution_config=False,
    description="Retrieves relevant documents. Call when facts from the knowledge base are needed.",
)

analyst = ConversableAgent(
    name="analyst",
    system_message="You analyze retrieved documents and provide insights.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    description="Analyzes documents and provides insights. Call after documents are retrieved.",
)

user = ConversableAgent(name="user", llm_config=False, human_input_mode="NEVER")

result = run_group_chat(
    pattern=AutoPattern(
        initial_agent=rag_proxy,
        agents=[rag_proxy, analyst],
        user_agent=user,
        group_manager_args={"llm_config": llm_config},
    ),
    messages="Summarize the key risks mentioned in the documents.",
    max_rounds=10,
)
```

## 5. Rules

- Install RAG extras: `pip install ag2[rag]` (includes ChromaDB)
- Use `get_or_create=True` to avoid re-indexing documents on every run
- Set `task="code"` for code search, `"qa"` for document Q&A
- `docs_path` accepts directories, file paths, and URLs
- For large corpora, consider Qdrant: `pip install ag2[qdrant]` with `QdrantRetrieveUserProxyAgent`
- `chunk_token_size` affects retrieval quality — smaller chunks are more precise, larger chunks have more context
- Always set `code_execution_config=False` on the RAG proxy unless code execution is needed
