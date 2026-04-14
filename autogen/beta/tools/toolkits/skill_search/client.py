# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import tempfile
from pathlib import Path

import httpx

from autogen.beta.exceptions import SkillDownloadError
from autogen.beta.tools.runtime import SkillMetadata, SkillRuntime

from .config import SkillsClientConfig
from .extractor import extract_skill


class SkillsClient:
    """HTTP client for skills.sh search and GitHub tarball downloads.

    Creates a fresh ``httpx.AsyncClient`` per request — no persistent connection
    to manage or close.

    Args:
        config: HTTP client configuration. Falls back to defaults when ``None``.
            Use :class:`SkillsClientConfig` to set a GitHub token, proxy, custom
            certificates, or other connection options.
    """

    SKILLS_SH_API = "https://skills.sh/api"
    GITHUB_API = "https://api.github.com"

    def __init__(self, config: SkillsClientConfig | None = None) -> None:
        cfg = config or SkillsClientConfig()

        self._headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "ag2-skill-search/1.0",
        }
        token = cfg.github_token or os.environ.get("GITHUB_TOKEN")
        if token:
            self._headers["Authorization"] = f"Bearer {token}"
        if cfg.headers:
            self._headers.update(cfg.headers)

        self._timeout = cfg.timeout
        self._proxy = cfg.proxy
        self._verify_ssl = cfg.verify_ssl
        self._cert = cfg.cert

    def _make_client(self, **overrides: object) -> httpx.AsyncClient:
        """Create a new ``httpx.AsyncClient`` with stored config."""
        kwargs: dict[str, object] = {
            "follow_redirects": True,
            "timeout": self._timeout,
            "headers": self._headers.copy(),
            "proxy": self._proxy,
            "verify": self._verify_ssl,
            "cert": self._cert,
        }
        kwargs.update(overrides)
        return httpx.AsyncClient(**kwargs)  # type: ignore[arg-type]

    async def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search skills.sh and return a list of skill records."""
        url = f"{self.SKILLS_SH_API}/search"
        async with self._make_client() as client:
            response = await client.get(url, params={"q": query, "limit": limit})
            response.raise_for_status()
            return response.json().get("skills", [])

    async def download_skill(self, source: str, skill_id: str, runtime: SkillRuntime) -> tuple[SkillMetadata, str]:
        """Download a skill via the GitHub Tarball API and install it via *runtime*.

        Args:
            source:   ``"owner/repo"``
            skill_id: Directory name inside the repo (e.g. ``"react-best-practices"``),
                      or empty string for a standalone repo.
            runtime:  Runtime that receives the extracted skill via ``runtime.install()``.

        Returns:
            A ``(metadata, sha256_hex)`` tuple.

        Raises:
            SkillDownloadError: On HTTP 403 (rate limit) or 404 (not found).
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tar_path = Path(tmp_dir) / "skill.tar.gz"
            hash_obj = hashlib.sha256()

            async with (
                self._make_client(timeout=120) as client,
                client.stream(
                    "GET",
                    f"{self.GITHUB_API}/repos/{source}/tarball",
                ) as resp,
            ):
                if resp.status_code == 403:
                    raise SkillDownloadError(
                        "GitHub rate limit exceeded. Set GITHUB_TOKEN or pass github_token= to SkillsClientConfig."
                    )
                if resp.status_code == 404:
                    raise SkillDownloadError(
                        f"Skill not found: {source}. Check that the repository exists and is public."
                    )
                resp.raise_for_status()
                with tar_path.open("wb") as fh:
                    async for chunk in resp.aiter_bytes():
                        hash_obj.update(chunk)
                        fh.write(chunk)

            staging = Path(tmp_dir) / "staged"
            staging.mkdir()
            meta = extract_skill(tar_path, skill_id, staging)
            runtime.install(staging / meta.name, meta.name)
            return meta, hash_obj.hexdigest()
