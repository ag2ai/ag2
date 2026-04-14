# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ("SkillsClientConfig",)


@dataclass
class SkillsClientConfig:
    """HTTP client configuration for downloading skills from GitHub.

    Args:
        github_token: GitHub PAT for authenticated requests. Falls back to the
            ``GITHUB_TOKEN`` env var. Raises the API rate limit from 60 → 5,000 req/hour.
        timeout: Request timeout in seconds. Defaults to 30.
        proxy: Proxy URL, e.g. ``"http://proxy.company.com:8080"``.
        verify_ssl: SSL certificate verification. ``False`` to disable, or a path to a
            CA bundle file. Defaults to ``True``.
        cert: Client certificate. Path to a ``.pem`` file, or a ``(cert, key)`` tuple.
        headers: Extra HTTP headers merged into every request.
    """

    github_token: str | None = None
    timeout: float = 30
    proxy: str | None = None
    verify_ssl: bool | str = True
    cert: str | tuple[str, str] | None = None
    headers: dict[str, str] = field(default_factory=dict)
