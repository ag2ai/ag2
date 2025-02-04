# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import time
from pathlib import Path
from typing import Union

from ....doc_utils import export_module
from ....import_utils import optional_import_block, require_optional_import
from .document_utils import handle_input

with optional_import_block():
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import ConversionResult
    from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption


_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)


@require_optional_import(["docling"], "rag")
@export_module("autogen.agentchat.contrib.rag")
def docling_parse_docs(  # type: ignore[no-any-unimported]
    input_file_path: Union[Path, str],
    output_dir_path: Union[Path, str],
) -> list["ConversionResult"]:
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

    Returns:
        list[ConversionResult]: The result of the conversion.
    """
    logging.basicConfig(level=logging.INFO)

    input_doc_paths: list[Path] = handle_input(input_file_path, output_dir=output_dir_path)

    if not input_doc_paths:
        raise ValueError("No documents found.")

    # Docling Parse PDF with EasyOCR (CPU only)
    # ----------------------
    pdf_pipeline_options = PdfPipelineOptions()
    pdf_pipeline_options.do_ocr = True
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

    _log.info(f"Document converted in {end_time:.2f} seconds.")

    ## Export results
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for res in conv_results:
        out_path = Path(output_dir_path)
        doc_filename = res.input.file.stem
        _log.info(f"Document {res.input.file.name} converted.\nSaved markdown output to: {out_path!s}")
        _log.debug(res.document._export_to_indented_text(max_text_len=16))
        # Export Docling document format to markdowndoc:
        with (out_path / f"{doc_filename}.md").open("w") as fp:
            fp.write(res.document.export_to_markdown())

        with (out_path / f"{doc_filename}.json").open("w") as fp:
            fp.write(json.dumps(res.document.export_to_dict()))

        # Export tables
        for table_ix, table in enumerate(res.document.tables):
            # Save the table as html
            element_html_filename = output_dir / f"{doc_filename}-table-{table_ix + 1}.html"
            _log.info(f"Saving HTML table to {element_html_filename}")
            with element_html_filename.open("w") as fp:
                fp.write(table.export_to_html())

    return conv_results
