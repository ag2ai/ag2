# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import shutil
from pathlib import Path

import pytest

from autogen.mcp.mcp_proxy.mcp_proxy import MCPProxy


@pytest.mark.skip
def test_generating_whatsapp():
    tmp_path = Path("tmp") / "mcp_whatsapp"
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(parents=True, exist_ok=True)

    MCPProxy.create(
        openapi_json=(Path(__file__).parent / "data.json").read_text(),
        # openapi_url="https://dev.infobip.com/openapi/products/whatsapp.json",
        client_source_path=tmp_path,
        servers=[{"url": "https://api.infobip.com"}],
    )

    assert tmp_path.exists(), "Failed to create tmp directory"
