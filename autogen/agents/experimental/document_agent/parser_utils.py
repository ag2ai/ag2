# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import time
from pathlib import Path
from typing import Annotated, Optional, Union

from ....doc_utils import export_module
from ....import_utils import optional_import_block, require_optional_import
from .document_utils import handle_input

with optional_import_block():
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

__all__ = ["docling_parse_docs"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@require_optional_import(["docling"], "rag")
@export_module("autogen.agents.experimental.document_agent")
def docling_parse_docs(  # type: ignore[no-any-unimported]
    input_file_path: Annotated[Union[Path, str], "Path to the input file or directory"],
    output_dir_path: Annotated[Optional[Union[Path, str]], "Path to the output directory"] = None,
    output_formats: Annotated[Optional[list[str]], "List of output formats (markdown, json)"] = None,
    table_output_format: str = "html",
) -> list[Path]:
    """Convert documents into a Deep Search document format using EasyOCR
    with CPU only, and export the document and its tables to the specified
    output directory.

    Supported formats:
        PDF,
        IMAGE,
        DOCX,
        HTML,
        PPTX,
        ASCIIDOC,
        MD,

    Args:
        input_file_path (Union[Path, str]): The path to the input file.
        output_dir_path (Union[Path, str]): The path to the output directory.
        output_formats (list[str], optional): The output formats. Defaults to ["markdown"].
        table_output_format (str, optional): The output format for tables. Defaults to "html".

    Returns:
        list[ConversionResult]: The result of the conversion.
    """
    output_dir_path = output_dir_path or Path("./output")
    output_formats = output_formats or ["markdown"]

    input_doc_paths: list[Path] = handle_input(input_file_path, output_dir=output_dir_path)

    if not input_doc_paths:
        raise ValueError("No documents found.")

    # Docling Parse PDF with EasyOCR (CPU only)
    # ----------------------
    pdf_pipeline_options = PdfPipelineOptions()
    pdf_pipeline_options.do_ocr = True
    if hasattr(pdf_pipeline_options.ocr_options, "use_gpu"):
        pdf_pipeline_options.ocr_options.use_gpu = False  # <-- set this.
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
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    conv_files = []

    for res in conv_results:
        out_path = Path(output_dir_path)
        doc_filename = res.input.file.stem
        logger.debug(f"Document {res.input.file.name} converted.\nSaved markdown output to: {out_path!s}")
        logger.debug(res.document._export_to_indented_text(max_text_len=16))

        if "markdown" in output_formats:
            # Export Docling document format to markdown:
            output_file = out_path / f"{doc_filename}.md"
            with output_file.open("w") as fp:
                fp.write(res.document.export_to_markdown())
                conv_files.append(output_file)

        if "json" in output_formats:
            # Export Docling document format to json
            output_file = out_path / f"{doc_filename}.json"
            with output_file.open("w") as fp:
                fp.write(json.dumps(res.document.export_to_dict()))
                conv_files.append(output_file)

        # Export tables (used for evaluating conversion)
        if table_output_format == "html":
            for table_ix, table in enumerate(res.document.tables):
                # Save the table as html
                element_html_filename = output_dir / f"{doc_filename}-table-{table_ix + 1}.html"
                logger.debug(f"Saving HTML table to {element_html_filename}")
                with element_html_filename.open("w") as fp:
                    fp.write(table.export_to_html())
    return conv_files
