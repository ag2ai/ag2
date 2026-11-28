# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""HTTP options shared by the Gemini clients."""

import httpx
import httpx2
from google.genai import types


def build_http_options(
    http_client: httpx.AsyncClient | httpx2.AsyncClient | None,
    api_version: str | None,
) -> types.HttpOptions | None:
    """Return the SDK's HTTP options, or ``None`` so the SDK keeps its own defaults."""
    if http_client is None and api_version is None:
        return None
    return types.HttpOptions(httpx_async_client=http_client, api_version=api_version)
