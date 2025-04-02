# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


import unittest
from unittest.mock import MagicMock

import pytest

from autogen.import_utils import optional_import_block, skip_on_missing_imports

with optional_import_block():
    from googleapiclient.discovery import build

    from autogen.tools.experimental.google.authentication.credentials_local_provider import (
        GoogleCredentialsLocalProvider,
    )
    from autogen.tools.experimental.google.authentication.credentials_provider import GoogleCredentialsProvider


@skip_on_missing_imports(
    [
        "google_auth_httplib2",
        "google_auth_oauthlib",
    ],
    "google-api",
)
class TestGoogleCredentialsLocalProvider:
    def test_init(self, tmp_client_secret_json_file_name: str) -> None:
        provider = GoogleCredentialsLocalProvider(
            client_secret_file=tmp_client_secret_json_file_name,
            scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
        )
        assert isinstance(provider, GoogleCredentialsProvider)
        assert provider.host == "localhost"
        assert provider.port == 8080

    def test_get_credentials_from_db(self, tmp_client_secret_json_file_name: str) -> None:
        provider = GoogleCredentialsLocalProvider(
            client_secret_file=tmp_client_secret_json_file_name,
            scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
        )

        with unittest.mock.patch(
            "autogen.tools.experimental.google.authentication.credentials_local_provider.GoogleCredentialsLocalProvider._refresh_or_get_new_credentials",
        ) as mock_refresh_or_get_new_credentials:
            user_creds = MagicMock()
            user_creds.refresh_token = "refresh"
            user_creds.client_id = "client"
            user_creds.client_secret = "secret"
            user_creds.valid = True
            mock_refresh_or_get_new_credentials.return_value = user_creds

            creds = provider.get_credentials()
            mock_refresh_or_get_new_credentials.assert_called_once()
            assert creds == user_creds

    @pytest.mark.skip(reason="This test requires real google credentials and is not suitable for CI at the moment")
    def test_end2end(self) -> None:
        """Shows basic usage of the Sheets API.
        Prints values from a sample spreadsheet.
        """
        client_secret_file = "client_secret_ag2.json"
        # The ID and range of a sample spreadsheet.
        spreadsheet_id = "1iqxU1SnfAqfWlC_7yezC-bW6CrF827NcTvZ81x_Q_KA"
        range_name = "Sheet1!A1:C"

        provider = GoogleCredentialsLocalProvider(
            client_secret_file=client_secret_file,
            scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
        )
        creds = provider.get_credentials()

        service = build("sheets", "v4", credentials=creds)

        # Call the Sheets API
        sheet = service.spreadsheets()

        result = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
        values = result.get("values", [])

        if not values:
            print("No data found.")
            return

        for row in values:
            print(row)
