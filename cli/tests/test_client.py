"""Tests for the artifact client module."""

import hashlib
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ag2_cli.install.client import REGISTRY_TTL, ArtifactClient, FetchError

# -- Fixtures --

SAMPLE_REGISTRY = {
    "artifacts": [
        {
            "owner": "ag2ai",
            "name": "fastapi",
            "type": "skill",
            "display_name": "FastAPI Skill",
            "description": "Build REST APIs with FastAPI",
            "tags": ["web", "api", "rest"],
        },
        {
            "owner": "ag2ai",
            "name": "web-search",
            "type": "tool",
            "display_name": "Web Search Tool",
            "description": "Search the web with Google",
            "tags": ["search", "google"],
        },
        {
            "owner": "community",
            "name": "chatbot",
            "type": "template",
            "display_name": "Chatbot Template",
            "description": "A simple chatbot template",
            "tags": ["chat", "starter"],
        },
    ]
}


class TestArtifactClientInit:
    """Test ArtifactClient initialization."""

    def test_default_params(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        assert client.repo == "ag2ai/resource-hub"
        assert client.branch == "main"
        assert client.cache_dir == tmp_path
        assert "Accept" in client._headers

    def test_custom_params(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(
                repo="myorg/myrepo",
                cache_dir=tmp_path / "custom",
                branch="dev",
            )

        assert client.repo == "myorg/myrepo"
        assert client.branch == "dev"
        assert client.cache_dir == tmp_path / "custom"


class TestInitAuth:
    """Test _init_auth reads tokens from environment."""

    def test_gh_token(self, tmp_path: Path):
        with patch.dict("os.environ", {"GH_TOKEN": "ghp_abc123"}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        assert client._headers["Authorization"] == "token ghp_abc123"

    def test_github_token_fallback(self, tmp_path: Path):
        with patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_fallback"}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        assert client._headers["Authorization"] == "token ghp_fallback"

    def test_gh_token_takes_precedence(self, tmp_path: Path):
        with patch.dict(
            "os.environ",
            {"GH_TOKEN": "ghp_primary", "GITHUB_TOKEN": "ghp_fallback"},
            clear=True,
        ):
            client = ArtifactClient(cache_dir=tmp_path)

        assert client._headers["Authorization"] == "token ghp_primary"

    def test_no_token(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        assert "Authorization" not in client._headers


class TestTypeDir:
    """Test _type_dir mapping for all artifact types."""

    @pytest.mark.parametrize(
        "artifact_type,expected",
        [
            ("template", "templates"),
            ("skill", "skills"),
            ("tool", "tools"),
            ("dataset", "datasets"),
            ("agent", "agents"),
            ("bundle", "bundles"),
        ],
    )
    def test_known_types(self, artifact_type: str, expected: str):
        assert ArtifactClient._type_dir(artifact_type) == expected

    def test_unknown_type_returns_as_is(self):
        assert ArtifactClient._type_dir("widgets") == "widgets"


class TestFetchRegistry:
    """Test fetch_registry caching behaviour."""

    def test_returns_cached_data_when_fresh(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        # Write cache files that are still fresh
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "registry.json").write_text(json.dumps(SAMPLE_REGISTRY))
        (tmp_path / "registry.meta.json").write_text(json.dumps({"fetched_at": time.time()}))

        # Should NOT call _get_json since cache is fresh
        with patch.object(client, "_get_json") as mock_get:
            result = client.fetch_registry()

        mock_get.assert_not_called()
        assert result == SAMPLE_REGISTRY

    def test_fetches_from_remote_when_cache_stale(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        # Write stale cache (older than TTL)
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "registry.json").write_text(json.dumps({"artifacts": []}))
        (tmp_path / "registry.meta.json").write_text(json.dumps({"fetched_at": time.time() - REGISTRY_TTL - 100}))

        with patch.object(client, "_get_json", return_value=SAMPLE_REGISTRY):
            result = client.fetch_registry()

        assert result == SAMPLE_REGISTRY
        # Verify cache was updated
        cached = json.loads((tmp_path / "registry.json").read_text())
        assert cached == SAMPLE_REGISTRY

    def test_fetches_from_remote_when_cache_missing(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path / "nonexistent")

        with patch.object(client, "_get_json", return_value=SAMPLE_REGISTRY):
            result = client.fetch_registry()

        assert result == SAMPLE_REGISTRY
        assert (client.cache_dir / "registry.json").exists()
        assert (client.cache_dir / "registry.meta.json").exists()

    def test_force_refresh_bypasses_cache(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        # Write fresh cache
        tmp_path.mkdir(parents=True, exist_ok=True)
        old_data = {"artifacts": [{"name": "old"}]}
        (tmp_path / "registry.json").write_text(json.dumps(old_data))
        (tmp_path / "registry.meta.json").write_text(json.dumps({"fetched_at": time.time()}))

        with patch.object(client, "_get_json", return_value=SAMPLE_REGISTRY):
            result = client.fetch_registry(force_refresh=True)

        assert result == SAMPLE_REGISTRY
        cached = json.loads((tmp_path / "registry.json").read_text())
        assert cached == SAMPLE_REGISTRY


class TestFetchArtifactDir:
    """Test fetch_artifact_dir with caching and error handling."""

    def test_returns_cached_dir_when_fetched_marker_exists(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        # Create the cached artifact directory with .fetched marker
        dest = tmp_path / "skills" / "ag2ai" / "fastapi" / "latest"
        dest.mkdir(parents=True)
        (dest / ".fetched").touch()

        result = client.fetch_artifact_dir("skill", "fastapi")

        assert result == dest

    def test_raises_fetch_error_when_artifact_not_found(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        with (
            patch.object(client, "_list_contents_recursive", return_value=[]),
            pytest.raises(FetchError, match="Artifact not found"),
        ):
            client.fetch_artifact_dir("skill", "nonexistent")

    def test_downloads_files_to_cache(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        file_list = [
            "skills/fastapi/artifact.json",
            "skills/fastapi/rules/SKILL.md",
        ]

        with (
            patch.object(client, "_list_contents_recursive", return_value=file_list),
            patch.object(client, "_get_bytes", return_value=b"content"),
        ):
            result = client.fetch_artifact_dir("skill", "fastapi")

        dest = tmp_path / "skills" / "ag2ai" / "fastapi" / "latest"
        assert result == dest
        assert (dest / ".fetched").exists()
        assert (dest / "artifact.json").exists()
        assert (dest / "rules" / "SKILL.md").exists()


class TestFetchFile:
    """Test fetch_file downloading and checksum verification."""

    def test_downloads_and_writes_file(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        dest = tmp_path / "downloads" / "file.txt"
        file_content = b"hello world"

        # Mock the streaming response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_bytes.return_value = [file_content]
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("ag2_cli.install.client.httpx.Client", return_value=mock_client):
            result = client.fetch_file("https://example.com/file.txt", dest)

        assert result == dest
        assert dest.read_bytes() == file_content

    def test_sha256_verification_pass(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        dest = tmp_path / "file.txt"
        file_content = b"hello world"
        expected_sha = hashlib.sha256(file_content).hexdigest()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_bytes.return_value = [file_content]
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("ag2_cli.install.client.httpx.Client", return_value=mock_client):
            result = client.fetch_file("https://example.com/file.txt", dest, sha256=expected_sha)

        assert result == dest
        assert dest.exists()

    def test_sha256_verification_fail_raises_and_deletes(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        dest = tmp_path / "file.txt"
        file_content = b"hello world"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_bytes.return_value = [file_content]
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with (
            patch("ag2_cli.install.client.httpx.Client", return_value=mock_client),
            pytest.raises(FetchError, match="Checksum mismatch"),
        ):
            client.fetch_file("https://example.com/file.txt", dest, sha256="bad_hash")

        # File should be deleted after checksum failure
        assert not dest.exists()

    def test_fetch_file_http_error(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        dest = tmp_path / "file.txt"

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with (
            patch("ag2_cli.install.client.httpx.Client", return_value=mock_client),
            pytest.raises(FetchError, match="HTTP 500"),
        ):
            client.fetch_file("https://example.com/file.txt", dest)


class TestSearch:
    """Test search filtering by keyword and type."""

    def test_search_by_keyword(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        results = client.search(SAMPLE_REGISTRY, "fastapi")
        assert len(results) == 1
        assert results[0]["name"] == "fastapi"

    def test_search_keyword_in_description(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        results = client.search(SAMPLE_REGISTRY, "google")
        assert len(results) == 1
        assert results[0]["name"] == "web-search"

    def test_search_keyword_in_tags(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        results = client.search(SAMPLE_REGISTRY, "rest")
        assert len(results) == 1
        assert results[0]["name"] == "fastapi"

    def test_search_case_insensitive(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        results = client.search(SAMPLE_REGISTRY, "FASTAPI")
        assert len(results) == 1
        assert results[0]["name"] == "fastapi"

    def test_search_filters_by_type(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        # "search" matches in "web-search" tool's tags
        results = client.search(SAMPLE_REGISTRY, "search", artifact_type="tool")
        assert len(results) == 1
        assert results[0]["name"] == "web-search"

    def test_search_type_filter_excludes_other_types(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        # "search" matches web-search but it's a tool, not a skill
        results = client.search(SAMPLE_REGISTRY, "search", artifact_type="skill")
        assert len(results) == 0

    def test_search_no_matches(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        results = client.search(SAMPLE_REGISTRY, "nonexistent_xyz")
        assert len(results) == 0

    def test_search_by_owner(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        results = client.search(SAMPLE_REGISTRY, "community")
        assert len(results) == 1
        assert results[0]["name"] == "chatbot"


class TestListArtifacts:
    """Test list_artifacts returns all or filters by type."""

    def test_list_all(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        results = client.list_artifacts(SAMPLE_REGISTRY)
        assert len(results) == 3

    def test_list_filter_by_type(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        results = client.list_artifacts(SAMPLE_REGISTRY, artifact_type="skill")
        assert len(results) == 1
        assert results[0]["name"] == "fastapi"

    def test_list_filter_by_type_no_matches(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        results = client.list_artifacts(SAMPLE_REGISTRY, artifact_type="dataset")
        assert len(results) == 0

    def test_list_empty_registry(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        results = client.list_artifacts({"artifacts": []})
        assert len(results) == 0


class TestGetJson:
    """Test _get_json HTTP error handling."""

    def test_404_raises_not_found(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        mock_resp = MagicMock()
        mock_resp.status_code = 404

        mock_http_client = MagicMock()
        mock_http_client.get.return_value = mock_resp
        mock_http_client.__enter__ = MagicMock(return_value=mock_http_client)
        mock_http_client.__exit__ = MagicMock(return_value=False)

        with (
            patch("ag2_cli.install.client.httpx.Client", return_value=mock_http_client),
            pytest.raises(FetchError, match="Not found"),
        ):
            client._get_json("https://example.com/missing")

    def test_403_raises_rate_limit(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        mock_resp = MagicMock()
        mock_resp.status_code = 403

        mock_http_client = MagicMock()
        mock_http_client.get.return_value = mock_resp
        mock_http_client.__enter__ = MagicMock(return_value=mock_http_client)
        mock_http_client.__exit__ = MagicMock(return_value=False)

        with (
            patch("ag2_cli.install.client.httpx.Client", return_value=mock_http_client),
            pytest.raises(FetchError, match="rate limit"),
        ):
            client._get_json("https://example.com/limited")

    def test_500_raises_http_error(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        mock_resp = MagicMock()
        mock_resp.status_code = 500

        mock_http_client = MagicMock()
        mock_http_client.get.return_value = mock_resp
        mock_http_client.__enter__ = MagicMock(return_value=mock_http_client)
        mock_http_client.__exit__ = MagicMock(return_value=False)

        with (
            patch("ag2_cli.install.client.httpx.Client", return_value=mock_http_client),
            pytest.raises(FetchError, match="HTTP 500"),
        ):
            client._get_json("https://example.com/error")

    def test_200_returns_json(self, tmp_path: Path):
        with patch.dict("os.environ", {}, clear=True):
            client = ArtifactClient(cache_dir=tmp_path)

        expected = {"key": "value"}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = expected

        mock_http_client = MagicMock()
        mock_http_client.get.return_value = mock_resp
        mock_http_client.__enter__ = MagicMock(return_value=mock_http_client)
        mock_http_client.__exit__ = MagicMock(return_value=False)

        with patch("ag2_cli.install.client.httpx.Client", return_value=mock_http_client):
            result = client._get_json("https://example.com/data")

        assert result == expected
