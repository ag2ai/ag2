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
        openapi_specification=(Path(__file__).parent / "data_github.json").read_text(),
        # openapi_url="https://dac-static.atlassian.com/cloud/trello/swagger.v3.json?_v=1.592.0",
        # openapi_url="https://dev.infobip.com/openapi/products/whatsapp.json",
        client_source_path="here",
        # servers=[{"url": "https://api.infobip.com"}],
    )

    assert tmp_path.exists(), "Failed to create tmp directory"
