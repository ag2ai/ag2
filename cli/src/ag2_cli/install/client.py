"""Remote artifact client — fetches from ag2ai/resource-hub GitHub repo with local caching."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

ARTIFACTS_REPO = "ag2ai/resource-hub"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com"
GITHUB_API_BASE = "https://api.github.com"
CACHE_DIR = Path.home() / ".ag2" / "cache" / "artifacts"
REGISTRY_TTL = 3600  # 1 hour


class FetchError(Exception):
    """Raised when a remote fetch fails."""


class ArtifactClient:
    """Fetches artifacts from the GitHub-hosted registry.

    Remote flow (e.g. `ag2 install skills fastapi`):

    1. Fetch registry.json (single GET, cached 1hr) — knows what exists
    2. For the requested artifact, use GitHub Contents API to list its directory
       (e.g. GET /repos/ag2ai/resource-hub/contents/skills/fastapi)
    3. Download each file via raw.githubusercontent.com to local cache
    4. Cache is keyed by {type}/{name}/{version}/ — immutable once fetched
    5. Installer reads from cache as if it were a local directory
    """

    def __init__(
        self,
        repo: str = ARTIFACTS_REPO,
        cache_dir: Path = CACHE_DIR,
        branch: str = "main",
    ):
        self.repo = repo
        self.branch = branch
        self.cache_dir = cache_dir
        self._headers: dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
        self._init_auth()

    def _init_auth(self) -> None:
        """Use GH_TOKEN or GITHUB_TOKEN for authenticated requests if available."""
        import os

        token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        if token:
            self._headers["Authorization"] = f"token {token}"

    def _raw_url(self, path: str) -> str:
        return f"{GITHUB_RAW_BASE}/{self.repo}/{self.branch}/{path}"

    def _api_url(self, path: str) -> str:
        return f"{GITHUB_API_BASE}/repos/{self.repo}/{path}"

    # -- Registry --

    def fetch_registry(self, force_refresh: bool = False) -> dict:
        """Fetch registry.json, using cache if fresh enough."""
        cache_path = self.cache_dir / "registry.json"
        cache_meta = self.cache_dir / "registry.meta.json"

        if not force_refresh and cache_path.exists() and cache_meta.exists():
            meta = json.loads(cache_meta.read_text())
            if time.time() - meta.get("fetched_at", 0) < REGISTRY_TTL:
                return json.loads(cache_path.read_text())

        url = self._raw_url("registry.json")
        data = self._get_json(url)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data, indent=2))
        cache_meta.write_text(json.dumps({"fetched_at": time.time()}))
        return data

    # -- Artifact fetching --

    @staticmethod
    def _type_dir(artifact_type: str) -> str:
        """Map artifact type to its directory name in the repo."""
        mapping = {
            "template": "templates",
            "skill": "skills",
            "skills": "skills",
            "tool": "tools",
            "dataset": "datasets",
            "agent": "agents",
            "bundle": "bundles",
        }
        return mapping.get(artifact_type, artifact_type)

    def fetch_artifact_manifest(self, artifact_type: str, name: str, owner: str = "ag2ai") -> dict:
        """Fetch a single artifact's artifact.json."""
        type_dir = self._type_dir(artifact_type)
        path = f"{type_dir}/{name}/artifact.json"
        return self._get_json(self._raw_url(path))

    def fetch_artifact_dir(self, artifact_type: str, name: str, owner: str = "ag2ai", version: str = "latest") -> Path:
        """Download an artifact directory to local cache. Returns the cached path.

        Uses the GitHub Contents API to list only the specific artifact directory,
        then downloads each file. This is O(files in artifact), not O(files in repo).

        Cache structure: ~/.ag2/cache/artifacts/{type}/{owner}/{name}/{version}/
        Once cached (marked by .fetched sentinel), subsequent calls return instantly.
        """
        # Parse "owner/name" format
        if "/" in name:
            owner, name = name.split("/", 1)

        type_dir = self._type_dir(artifact_type)
        dest = self.cache_dir / type_dir / owner / name / version
        marker = dest / ".fetched"
        if marker.exists():
            return dest

        # List files in the specific artifact directory via Contents API.
        # Try owner-namespaced path first (e.g. skills/ag2ai/fastapi), then
        # fall back to flat path (e.g. skills/fastapi) for backward compat.
        repo_path = f"{type_dir}/{name}"
        if owner != "ag2ai":
            repo_path = f"{type_dir}/{owner}/{name}"
        files = self._list_contents_recursive(repo_path)

        if not files:
            raise FetchError(f"Artifact not found: {owner}/{name} (looked in {type_dir}/{name})")

        # Download each file to local cache
        dest.mkdir(parents=True, exist_ok=True)
        prefix = f"{repo_path}/"
        for file_path in files:
            rel_path = file_path[len(prefix) :] if file_path.startswith(prefix) else file_path
            local_path = dest / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            content = self._get_bytes(self._raw_url(file_path))
            local_path.write_bytes(content)

        marker.touch()
        return dest

    def _list_contents_recursive(self, repo_path: str) -> list[str]:
        """List all files under a repo path using the GitHub Contents API.

        For a path like 'skills/fastapi', this returns:
        ['skills/fastapi/artifact.json', 'skills/fastapi/rules/imports/SKILL.md', ...]

        Uses recursive directory traversal via the Contents API — one API call
        per directory level. For a typical artifact (2-3 levels), that's 3-5 calls,
        much better than downloading the entire repo tree.
        """
        url = self._api_url(f"contents/{repo_path}?ref={self.branch}")
        try:
            entries = self._get_json_list(url)
        except FetchError:
            return []

        files: list[str] = []
        for entry in entries:
            if entry.get("type") == "file":
                files.append(entry["path"])
            elif entry.get("type") == "dir":
                files.extend(self._list_contents_recursive(entry["path"]))
        return files

    def fetch_file(self, url: str, dest: Path, sha256: str | None = None) -> Path:
        """Download a single file with optional checksum verification."""
        dest.parent.mkdir(parents=True, exist_ok=True)

        with httpx.Client(follow_redirects=True, timeout=300) as client, client.stream("GET", url) as response:
            if response.status_code != 200:
                raise FetchError(f"Failed to download {url}: HTTP {response.status_code}")
            with open(dest, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)

        if sha256:
            import hashlib

            try:
                hasher = hashlib.sha256()
                with open(dest, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hasher.update(chunk)
                if hasher.hexdigest() != sha256:
                    raise FetchError(f"Checksum mismatch for {dest.name}: expected {sha256}")
            except Exception:
                # Clean up the downloaded file on any verification failure
                dest.unlink(missing_ok=True)
                raise

        return dest

    # -- Search --

    def search(self, registry: dict, query: str, artifact_type: str | None = None) -> list[dict]:
        """Search registry entries by keyword across owner, name, description, tags."""
        query_lower = query.lower()
        results = []
        for entry in registry.get("artifacts", []):
            if artifact_type and entry.get("type") != artifact_type:
                continue
            searchable = " ".join([
                entry.get("owner", ""),
                entry.get("name", ""),
                entry.get("display_name", ""),
                entry.get("description", ""),
                " ".join(entry.get("tags", [])),
            ]).lower()
            if query_lower in searchable:
                results.append(entry)
        return results

    def list_artifacts(self, registry: dict, artifact_type: str | None = None) -> list[dict]:
        """List all artifacts, optionally filtered by type."""
        entries = registry.get("artifacts", [])
        if artifact_type:
            entries = [e for e in entries if e.get("type") == artifact_type]
        return entries

    # -- HTTP helpers --

    def _get_json(self, url: str) -> dict:
        with httpx.Client(follow_redirects=True, timeout=30, headers=self._headers) as client:
            resp = client.get(url)
            if resp.status_code == 404:
                raise FetchError(f"Not found: {url}")
            if resp.status_code == 403:
                raise FetchError(
                    "GitHub API rate limit exceeded. Set GH_TOKEN or GITHUB_TOKEN "
                    "environment variable for higher limits."
                )
            if resp.status_code != 200:
                raise FetchError(f"HTTP {resp.status_code} fetching {url}")
            return resp.json()

    def _get_json_list(self, url: str) -> list:
        """Like _get_json but expects a JSON array response (Contents API returns arrays)."""
        with httpx.Client(follow_redirects=True, timeout=30, headers=self._headers) as client:
            resp = client.get(url)
            if resp.status_code == 404:
                raise FetchError(f"Not found: {url}")
            if resp.status_code == 403:
                raise FetchError(
                    "GitHub API rate limit exceeded. Set GH_TOKEN or GITHUB_TOKEN "
                    "environment variable for higher limits."
                )
            if resp.status_code != 200:
                raise FetchError(f"HTTP {resp.status_code} fetching {url}")
            data = resp.json()
            # Contents API returns a list for directories, dict for single files
            if isinstance(data, dict):
                return [data]
            return data

    def _get_bytes(self, url: str) -> bytes:
        with httpx.Client(follow_redirects=True, timeout=30, headers=self._headers) as client:
            resp = client.get(url)
            if resp.status_code != 200:
                raise FetchError(f"HTTP {resp.status_code} fetching {url}")
            return resp.content
