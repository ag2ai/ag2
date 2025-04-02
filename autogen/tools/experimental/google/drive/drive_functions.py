# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import io
from pathlib import Path
from typing import Any, Optional

from .....import_utils import optional_import_block, require_optional_import
from ..model import GoogleFileInfo

with optional_import_block():
    from googleapiclient.http import MediaIoBaseDownload


__all__ = [
    "download_file",
    "list_files_and_folders",
]


@require_optional_import(
    [
        "googleapiclient",
    ],
    "google-api",
)
def list_files_and_folders(service: Any, page_size: int, folder_id: Optional[str]) -> list[GoogleFileInfo]:
    kwargs = {
        "pageSize": page_size,
        "fields": "nextPageToken, files(id, name, mimeType)",
    }
    if folder_id:
        kwargs["q"] = f"'{folder_id}' in parents and trashed=false"  # Search for files in the folder
    response = service.files().list(**kwargs).execute()
    result = response.get("files", [])
    if not isinstance(result, list):
        raise ValueError(f"Expected a list of files, but got {result}")
    result = [GoogleFileInfo(**file_info) for file_info in result]
    return result


def _get_file_extension(mime_type: str) -> Optional[str]:
    """Returns the correct file extension for a given MIME type."""
    mime_extensions = {
        "application/vnd.google-apps.document": "docx",  # Google Docs → Word
        "application/vnd.google-apps.spreadsheet": "csv",  # Google Sheets → CSV
        "application/vnd.google-apps.presentation": "pptx",  # Google Slides → PowerPoint
        "video/quicktime": "mov",
        "application/vnd.google.colaboratory": "ipynb",
        "application/pdf": "pdf",
        "image/jpeg": "jpg",
        "image/png": "png",
        "text/plain": "txt",
        "application/zip": "zip",
    }

    return mime_extensions.get(mime_type)


@require_optional_import(
    [
        "googleapiclient",
    ],
    "google-api",
)
def download_file(service: Any, file_id: str, file_name: str, mime_type: str, download_folder: Path) -> str:
    """Download or export file based on its MIME type."""
    file_extension = _get_file_extension(mime_type)
    if file_extension and (not file_name.endswith(file_extension)):
        file_name = f"{file_name}.{file_extension}"

    # Define export formats for Google Docs, Sheets, and Slides
    export_mime_types = {
        "application/vnd.google-apps.document": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # Google Docs → Word
        "application/vnd.google-apps.spreadsheet": "text/csv",  # Google Sheets → CSV
        "application/vnd.google-apps.presentation": "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # Google Slides → PowerPoint
    }

    # Google Docs, Sheets, and Slides cannot be downloaded directly using service.files().get_media() because they are Google-native files
    if mime_type in export_mime_types:
        request = service.files().export(fileId=file_id, mimeType=export_mime_types[mime_type])
    else:
        # Download normal files (videos, images, etc.)
        request = service.files().get_media(fileId=file_id)

    # Save file
    with io.BytesIO() as buffer:
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        file_path = download_folder / file_name
        with open(file_path, "wb") as f:
            f.write(buffer.getvalue())

    return f"✅ Downloaded: {file_name}"
