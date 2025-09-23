# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import tempfile
from pathlib import Path
from typing import Any

import fitz  # type: ignore
import requests
import urllib3

from autogen.agentchat.contrib.capabilities.text_compressors import LLMLingua
from autogen.agentchat.contrib.capabilities.transforms import TextMessageCompressor
from autogen.agents.experimental.document_agent.parser_utils import docling_parse_docs

logger = logging.getLogger(__name__)


def extract_text_from_pdf(doc_path: str) -> list[dict[str, str]]:
    """Extract compressed text from a PDF file"""
    if isinstance(doc_path, str) and urllib3.util.url.parse_url(doc_path).scheme:
        # Download the PDF
        response = requests.get(doc_path)
        response.raise_for_status()  # Ensure the download was successful

        text = ""
        # Save the PDF to a temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pdf_path = Path(temp_dir) / "temp.pdf"
            with open(temp_pdf_path, "wb") as f:
                f.write(response.content)

            # Open the PDF
            with fitz.open(str(temp_pdf_path)) as doc:
                # Read and extract text from each page
                for page in doc:
                    text += page.get_text()
            llm_lingua = LLMLingua()
            text_compressor = TextMessageCompressor(text_compressor=llm_lingua)
            compressed_text = text_compressor.apply_transform([{"content": text}])

        return compressed_text
    else:
        raise ValueError("doc_path must be a string or a URL")


def aggregate_rag_results(query: str, results: dict[str, Any]) -> str:
    """Aggregate results from multiple RAG engines."""
    if not results:
        return f"Query: {query}\nAnswer: No results found from any RAG engine."

    # Simple aggregation
    answer_parts = [f"Query: {query}"]

    for engine_name, result in results.items():
        answer_parts.append(f"\n{engine_name.upper()} Results:")
        answer_parts.append(f"Answer: {result.get('answer', 'No answer available')}")

        # Add citations if available
        if "citations" in result and result["citations"]:
            answer_parts.append("Citations:")
            for i, citation in enumerate(result["citations"], 1):
                answer_parts.append(f"  [{i}] {citation.get('file_path', 'Unknown')}")

    return "\n".join(answer_parts)


def compress_and_save_text(text: str, input_path: str, parsed_docs_path: Path) -> str:
    """Compress text and save as markdown file."""
    llm_lingua = LLMLingua()
    text_compressor = TextMessageCompressor(text_compressor=llm_lingua)
    compressed_text = text_compressor.apply_transform([{"content": text}])

    if not compressed_text or not compressed_text[0].get("content"):
        raise ValueError("Text compression failed or returned empty result")

    # Create a markdown file with the extracted text
    output_file = parsed_docs_path / f"{Path(input_path).stem}.md"
    parsed_docs_path.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(compressed_text[0]["content"])

    return str(output_file)


def ingest_to_engines(
    output_file: str, input_path: str, rag_config: dict[str, Any], rag_engines: dict[str, Any]
) -> None:
    """Ingest document to configured RAG engines."""
    from autogen.agentchat.contrib.graph_rag.document import Document, DocumentType

    # Determine document type
    doc_type = DocumentType.TEXT
    if input_path.lower().endswith(".pdf"):
        doc_type = DocumentType.PDF
    elif input_path.lower().endswith((".html", ".htm")):
        doc_type = DocumentType.HTML
    elif input_path.lower().endswith(".json"):
        doc_type = DocumentType.JSON

    # Create Document object for graph engines
    graph_doc = Document(doctype=doc_type, path_or_url=output_file, data=None)

    # Ingest to configured engines only
    for rag_type in rag_config:
        engine = rag_engines.get(rag_type)
        if engine is None:
            continue

        try:
            if rag_type == "vector":
                engine.add_docs(new_doc_paths_or_urls=[output_file])
            elif rag_type == "graph":
                # For graph engines, we need to initialize if not done already
                if not hasattr(engine, "_initialized"):
                    engine.init_db([graph_doc])
                    engine._initialized = True
                else:
                    # Add new records to existing graph
                    if hasattr(engine, "add_records"):
                        engine.add_records([graph_doc])
        except Exception as e:
            logger.warning(f"Failed to ingest to {rag_type} engine: {e}")


def process_single_document(
    input_file_path: str, parsed_docs_path: Path, rag_config: dict[str, Any], rag_engines: dict[str, Any]
) -> tuple[str, bool, str]:
    """Process a single document. Returns (path, success, error_msg)."""
    try:
        # Check if the document is a PDF
        is_pdf = False
        if isinstance(input_file_path, str) and (
            input_file_path.lower().endswith(".pdf")
            or (urllib3.util.url.parse_url(input_file_path).scheme and input_file_path.lower().endswith(".pdf"))
        ):
            # Check for PDF extension or URL ending with .pdf
            is_pdf = True

        if is_pdf:
            # Handle PDF with PyMuPDF
            logger.info("PDF found, using PyMuPDF for extraction")
            if urllib3.util.url.parse_url(input_file_path).scheme:
                # Download the PDF
                response = requests.get(input_file_path)
                response.raise_for_status()

                text = ""
                # Save the PDF to a temporary file and extract text
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_pdf_path = Path(temp_dir) / "temp.pdf"
                    with open(temp_pdf_path, "wb") as f:
                        f.write(response.content)

                    # Open the PDF and extract text
                    with fitz.open(temp_pdf_path) as doc:
                        for page in doc:
                            text += page.get_text()

                    # Compress and save
                    output_file = compress_and_save_text(text, input_file_path, parsed_docs_path)

                    # Ingest to all active engines
                    ingest_to_engines(output_file, input_file_path, rag_config, rag_engines)

                    return (input_file_path, True, "")
            else:
                # Local PDF file
                text = ""
                with fitz.open(input_file_path) as doc:
                    for page in doc:
                        text += page.get_text()

                # Compress and save
                output_file = compress_and_save_text(text, input_file_path, parsed_docs_path)

                # Ingest to all active engines
                ingest_to_engines(output_file, input_file_path, rag_config, rag_engines)

                return (input_file_path, True, "")
        else:
            # Handle non-PDF documents with docling
            output_files = docling_parse_docs(
                input_file_path=input_file_path,
                output_dir_path=parsed_docs_path,
                output_formats=["markdown"],
            )

            # Limit to one output markdown file for now.
            if output_files:
                parsed_output_file: Path = output_files[0]
                if parsed_output_file.suffix == ".md":
                    # Ingest to all active engines
                    ingest_to_engines(str(parsed_output_file), input_file_path, rag_config, rag_engines)
                    return (input_file_path, True, "")

            return (input_file_path, False, "No valid markdown output generated")

    except Exception as doc_error:
        return (input_file_path, False, str(doc_error))


def execute_single_query(query_text: str, rag_config: dict[str, Any], rag_engines: dict[str, Any]) -> tuple[str, str]:
    """Execute a single query across configured RAG engines. Returns (query, result)."""
    try:
        results = {}

        # Only query engines that are configured in rag_config
        for rag_type in rag_config:
            engine = rag_engines.get(rag_type)
            if engine is None:
                continue

            try:
                if rag_type == "vector":
                    # Handle vector queries
                    if (
                        hasattr(engine, "enable_query_citations")
                        and getattr(engine, "enable_query_citations", False)
                        and hasattr(engine, "query_with_citations")
                        and callable(getattr(engine, "query_with_citations", None))
                    ):
                        answer_with_citations = getattr(engine, "query_with_citations")(query_text)
                        answer = answer_with_citations.answer
                        txt_citations = [
                            {
                                "text_chunk": source.node.get_text(),
                                "file_path": source.metadata.get("file_path", "Unknown"),
                            }
                            for source in answer_with_citations.citations
                        ]
                        results[rag_type] = {"answer": answer, "citations": txt_citations}
                        logger.info(f"Vector Citations: {txt_citations}")
                    else:
                        answer = engine.query(query_text) if engine else "Vector engine not available"
                        results[rag_type] = {"answer": answer}

                elif rag_type == "graph":
                    # Handle graph queries
                    # Try to connect to existing graph if not already connected
                    if not hasattr(engine, "index"):
                        try:
                            engine.connect_db()
                            logger.info("Connected to existing Neo4j graph for querying")
                        except Exception as connect_error:
                            logger.warning(f"Failed to connect to Neo4j graph: {connect_error}")
                            results[rag_type] = {"answer": f"Error connecting to graph: {connect_error}"}
                            continue

                    graph_result = engine.query(query_text)
                    results[rag_type] = {"answer": graph_result.answer, "results": graph_result.results}

            except Exception as engine_error:
                logger.warning(f"Failed to query {rag_type} engine: {engine_error}")
                results[rag_type] = {"answer": f"Error querying {rag_type}: {engine_error}"}

        # Aggregate results
        aggregated_answer = aggregate_rag_results(query_text, results)
        return (query_text, aggregated_answer)

    except Exception as query_error:
        logger.warning(f"Failed to execute query '{query_text}': {query_error}")
        return (query_text, f"Query: {query_text}\nAnswer: Error executing query: {query_error}")
