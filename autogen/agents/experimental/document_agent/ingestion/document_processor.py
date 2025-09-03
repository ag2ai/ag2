# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import time
from pathlib import Path
from typing import Annotated

from .....doc_utils import export_module
from .....import_utils import optional_import_block, require_optional_import
from ..core.base_interfaces import DocumentProcessor
from ..document_utils import handle_input

with optional_import_block():
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

__all__ = ["DoclingDocumentProcessor"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@require_optional_import(["docling"], "rag")
@export_module("autogen.agents.experimental.document_agent.ingestion")
class DoclingDocumentProcessor(DocumentProcessor):
    """Document processor using Docling for parsing and chunking."""

    def __init__(self, output_dir: Path | str | None = None, chunk_size: int = 512) -> None:
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "output"
        self.chunk_size = chunk_size
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_document(self, input_path: Path | str, output_dir: Path | str | None = None) -> list[Path]:
        """Process a document using Docling and return output file paths."""
        output_dir_path = Path(output_dir) if output_dir else self.output_dir
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # Use existing docling_parse_docs logic
        return self._docling_parse_docs(input_path, output_dir_path)

    def chunk_document(self, document_path: Path | str, chunk_size: int | None = None) -> list[str]:
        """Chunk a document into smaller pieces."""
        chunk_size = chunk_size or self.chunk_size

        with open(document_path, encoding="utf-8") as f:
            content = f.read()

        # Handle empty content - return a single empty chunk
        if not content:
            return [""]

        # Simple chunking by character count
        chunks = []
        for i in range(0, len(content), chunk_size):
            chunk = content[i : i + chunk_size]
            chunks.append(chunk)

        return chunks

    def _docling_parse_docs(
        self,
        input_file_path: Annotated[Path | str, "Path to the input file or directory"],
        output_dir_path: Annotated[Path | str, "Path to the output directory"],
        output_formats: Annotated[list[str], "List of output formats (markdown, json)"] | None = None,
        table_output_format: str = "html",
    ) -> list[Path]:
        """Convert documents using Docling (moved from parser_utils.py)."""
        output_dir_path = Path(output_dir_path).resolve()
        output_dir_path.mkdir(parents=True, exist_ok=True)

        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)

        output_formats = output_formats or ["markdown"]

        input_doc_paths: list[Path] = handle_input(input_file_path, output_dir=str(output_dir_path))

        if not input_doc_paths:
            raise ValueError("No documents found.")

        # Docling Parse PDF with EasyOCR (CPU only)
        pdf_pipeline_options = PdfPipelineOptions()
        pdf_pipeline_options.do_ocr = True
        if hasattr(pdf_pipeline_options.ocr_options, "use_gpu"):
            pdf_pipeline_options.ocr_options.use_gpu = False
        pdf_pipeline_options.do_table_structure = True
        pdf_pipeline_options.table_structure_options.do_cell_matching = True
        pdf_pipeline_options.ocr_options.lang = ["en"]
        pdf_pipeline_options.accelerator_options = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.AUTO)

        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
            },
        )

        start_time = time.time()
        conv_results = list(doc_converter.convert_all(input_doc_paths))
        end_time = time.time() - start_time

        logger.info(f"Document converted in {end_time:.2f} seconds.")

        # Export results
        conv_files = []

        for res in conv_results:
            out_path = Path(output_dir_path).resolve()
            doc_filename = res.input.file.stem
            logger.debug(f"Document {res.input.file.name} converted.\nSaved markdown output to: {out_path!s}")

            if "markdown" in output_formats:
                output_file = out_path / f"{doc_filename}.md"
                with output_file.open("w") as fp:
                    fp.write(res.document.export_to_markdown())
                    conv_files.append(output_file)

            if "json" in output_formats:
                output_file = out_path / f"{doc_filename}.json"
                with output_file.open("w") as fp:
                    fp.write(json.dumps(res.document.export_to_dict()))
                    conv_files.append(output_file)

        return conv_files
