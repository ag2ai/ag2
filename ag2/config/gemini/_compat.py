# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Everything that depends on which `google-genai` SDK major is installed lives here."""

import warnings

import google.genai
import httpx
import httpx2
from google.genai import types

SDK_3X = int(google.genai.__version__.split(".")[0]) >= 3
"""`google-genai` 3.x runs on `httpx2` and accepts an `httpx2` client; 2.x knows only `httpx`."""

SDK_2X_DEPRECATION = (
    "Support for google-genai 2.x is deprecated and will be removed in 1.4.0, which requires google-genai>=3. "
    'Upgrade with `pip install -U "google-genai>=3,<4"`.'
)

HTTPX2_NEEDS_SDK_3X = (
    "An httpx2.AsyncClient needs google-genai>=3.0; the installed google-genai {version} only accepts "
    "an httpx.AsyncClient. Upgrade google-genai or pass an httpx.AsyncClient."
)


def build_http_options(
    http_client: httpx.AsyncClient | httpx2.AsyncClient | None,
    api_version: str | None,
) -> types.HttpOptions | None:
    """Return the SDK's HTTP options, or ``None`` so the SDK keeps its own defaults."""
    if http_client is None and api_version is None:
        return None
    if isinstance(http_client, httpx2.AsyncClient) and not SDK_3X:
        raise TypeError(HTTPX2_NEEDS_SDK_3X.format(version=google.genai.__version__))
    return types.HttpOptions(httpx_async_client=http_client, api_version=api_version)


def warn_if_sdk_2x() -> None:
    """Warn the caller of a config's ``create*()`` that the installed google-genai 2.x loses support in 1.4.0."""
    if not SDK_3X:
        warnings.warn(SDK_2X_DEPRECATION, DeprecationWarning, stacklevel=3)
