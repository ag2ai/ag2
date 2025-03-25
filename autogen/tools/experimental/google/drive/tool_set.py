# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

from .....import_utils import optional_import_block, require_optional_import
from .....tools import ToolSet, tool

with optional_import_block():
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

__all__ = [
    "GoogleDriveToolSet",
]


@require_optional_import(
    [
        "googleapiclient",
    ],
    "google-api",
)
def _list_files(service: Any, page_size: int) -> Any:
    # Call the Drive v3 API
    results = service.files().list(pageSize=page_size, fields="nextPageToken, files(id, name)").execute()
    return results.get("files", [])


class GoogleDriveToolSet(ToolSet):
    def __init__(
        self,
        *,
        credentials: "Credentials",
        download_folder: Union[Path, str],
        exclude: Optional[list[Literal["list_files_in_folder", "download_file", "upload_file"]]] = None,
        api_version: str = "v3",
    ) -> None:
        self.credentials = credentials
        self.api_version = api_version
        self.service = build(serviceName="drive", version=api_version, credentials=credentials)

        tools_list = []

        @tool(description="list all files in a Google Drive folder")
        def list_files_in_folder(
            page_size: Annotated[int, "The number of files to list per page."] = 10,
            folder_path: Annotated[
                Optional[str],
                "The path of the folder to list files in. If not provided, lists files in the root folder.",
            ] = None,
        ) -> Any:
            return _list_files(service=self.service, page_size=page_size)

        if exclude is None:
            exclude = []

        tools_list = [tool for tool in [list_files_in_folder] if tool.name not in exclude]
        super().__init__(tools=tools_list)
