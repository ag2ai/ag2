# Copyright (c) 2023 - 2025, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from autogen.agentchat.contrib.rag.document_utils import handle_input

_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)


from typing import List


def docline_parse_docs(
    input_file_path: str,
    output_dir_path: str,
) -> list[ConversionResult]:
    """
    Convert documents into a Deep Search document format using EasyOCR
    with CPU only, and export the document and its tables to the specified
    output directory.

    Args:
        input_file_path (str): The path/directory to the documents to convert.
        output_dir_path (str): The path to the directory where to export the
            converted document and its files.

    Returns:
        list[ConversionResult]: The result of the conversion.
    """
    logging.basicConfig(level=logging.INFO)

    input_doc_paths: List[Path] = handle_input(input_file_path)

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
        _log.info(f"Document {res.input.file.name} converted.\nSaved markdown output to: {str(out_path)}")
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
