# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import tempfile
from typing import Generator

import pytest


@pytest.fixture
def tmp_client_secret_json_file_name() -> Generator[str, None, None]:
    with tempfile.NamedTemporaryFile(mode="w") as f:
        client_secret_json = {
            "installed": {
                "client_id": "dummy",
                "project_id": "ag2-sheets",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_secret": "secret",
                "redirect_uris": ["http://localhost"],
            }
        }
        f.write(str(client_secret_json))
        f.seek(0)
        yield f.name
