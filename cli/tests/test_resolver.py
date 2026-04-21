"""Tests for the dependency resolver module."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _write_artifact_json(directory: Path, data: dict) -> Path:
    """Helper: write an artifact.json into a directory and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    artifact_json = directory / "artifact.json"
    artifact_json.write_text(json.dumps(data))
    return artifact_json


def _make_artifact(
    name: str,
    artifact_type: str = "skill",
    owner: str = "ag2ai",
    version: str = "1.0.0",
    depends: list[str] | None = None,
    source_dir: Path | None = None,
):
    """Helper: create an Artifact dataclass directly."""
    from ag2_cli.install.artifact import Artifact

    return Artifact(
        name=name,
        type=artifact_type,
        owner=owner,
        version=version,
        depends=depends or [],
        source_dir=source_dir,
    )


class TestResolveNoDependencies:
    """resolve() with no dependencies returns empty list."""

    def test_returns_empty_when_no_depends(self, tmp_path: Path):
        from ag2_cli.install.lockfile import Lockfile
        from ag2_cli.install.resolver import DependencyResolver

        client = MagicMock()
        lockfile = Lockfile(tmp_path)
        resolver = DependencyResolver(client, lockfile)

        artifact = _make_artifact("root-skill", depends=[])
        result = resolver.resolve(artifact)

        assert result == []
        client.fetch_artifact_dir.assert_not_called()


class TestResolveSingleDependency:
    """resolve() with a single dependency."""

    def test_returns_single_dep(self, tmp_path: Path):
        from ag2_cli.install.lockfile import Lockfile
        from ag2_cli.install.resolver import DependencyResolver

        dep_dir = tmp_path / "cache" / "dep"
        _write_artifact_json(
            dep_dir,
            {
                "name": "helper",
                "type": "skill",
                "owner": "ag2ai",
                "version": "1.0.0",
            },
        )

        client = MagicMock()
        client.fetch_artifact_dir.return_value = dep_dir

        lockfile = Lockfile(tmp_path)
        resolver = DependencyResolver(client, lockfile)

        root = _make_artifact("root", depends=["skills/ag2ai/helper"])
        result = resolver.resolve(root)

        assert len(result) == 1
        assert result[0].name == "helper"
        assert result[0].version == "1.0.0"

    def test_returns_single_dep_two_part_ref(self, tmp_path: Path):
        from ag2_cli.install.lockfile import Lockfile
        from ag2_cli.install.resolver import DependencyResolver

        dep_dir = tmp_path / "cache" / "dep"
        _write_artifact_json(
            dep_dir,
            {
                "name": "helper",
                "type": "skill",
                "owner": "ag2ai",
                "version": "2.0.0",
            },
        )

        client = MagicMock()
        client.fetch_artifact_dir.return_value = dep_dir

        lockfile = Lockfile(tmp_path)
        resolver = DependencyResolver(client, lockfile)

        root = _make_artifact("root", depends=["skills/helper"])
        result = resolver.resolve(root)

        assert len(result) == 1
        assert result[0].name == "helper"
        assert result[0].version == "2.0.0"
        # Two-part ref defaults owner to "ag2ai"
        client.fetch_artifact_dir.assert_called_once_with("skills", "helper", owner="ag2ai")


class TestResolveDiamondDependencies:
    """resolve() with diamond dependencies (A->B, A->C, B->D, C->D)."""

    def test_diamond_deduplicates_shared_dep(self, tmp_path: Path):
        from ag2_cli.install.lockfile import Lockfile
        from ag2_cli.install.resolver import DependencyResolver

        # Build cached artifact directories for B, C, D
        dir_b = tmp_path / "cache" / "b"
        _write_artifact_json(
            dir_b,
            {
                "name": "b",
                "type": "skill",
                "owner": "ag2ai",
                "version": "1.0.0",
                "depends": ["skills/ag2ai/d"],
            },
        )

        dir_c = tmp_path / "cache" / "c"
        _write_artifact_json(
            dir_c,
            {
                "name": "c",
                "type": "skill",
                "owner": "ag2ai",
                "version": "1.0.0",
                "depends": ["skills/ag2ai/d"],
            },
        )

        dir_d = tmp_path / "cache" / "d"
        _write_artifact_json(
            dir_d,
            {
                "name": "d",
                "type": "skill",
                "owner": "ag2ai",
                "version": "1.0.0",
            },
        )

        def fake_fetch(artifact_type, name, owner="ag2ai"):
            return {"b": dir_b, "c": dir_c, "d": dir_d}[name]

        client = MagicMock()
        client.fetch_artifact_dir.side_effect = fake_fetch

        lockfile = Lockfile(tmp_path)
        resolver = DependencyResolver(client, lockfile)

        root = _make_artifact("a", depends=["skills/ag2ai/b", "skills/ag2ai/c"])
        result = resolver.resolve(root)

        result_names = [a.name for a in result]
        # D should appear exactly once
        assert result_names.count("d") == 1
        # All three deps present
        assert set(result_names) == {"b", "c", "d"}
        # D must come before B and C (dependencies first)
        assert result_names.index("d") < result_names.index("b")
        assert result_names.index("d") < result_names.index("c")


class TestResolveSkipsInstalled:
    """resolve() skips already-installed dependencies."""

    def test_skips_installed_at_same_version(self, tmp_path: Path):
        from ag2_cli.install.lockfile import Lockfile
        from ag2_cli.install.resolver import DependencyResolver

        dep_dir = tmp_path / "cache" / "dep"
        _write_artifact_json(
            dep_dir,
            {
                "name": "helper",
                "type": "skill",
                "owner": "ag2ai",
                "version": "1.0.0",
            },
        )

        client = MagicMock()
        client.fetch_artifact_dir.return_value = dep_dir

        lockfile = Lockfile(tmp_path)
        # Mark the dependency as already installed at the same version
        lockfile.record_install(
            ref="skills/ag2ai/helper",
            version="1.0.0",
            targets=["claude"],
            files=[],
        )

        resolver = DependencyResolver(client, lockfile)
        root = _make_artifact("root", depends=["skills/ag2ai/helper"])
        result = resolver.resolve(root)

        assert result == []

    def test_includes_installed_at_different_version(self, tmp_path: Path):
        from ag2_cli.install.lockfile import Lockfile
        from ag2_cli.install.resolver import DependencyResolver

        dep_dir = tmp_path / "cache" / "dep"
        _write_artifact_json(
            dep_dir,
            {
                "name": "helper",
                "type": "skill",
                "owner": "ag2ai",
                "version": "2.0.0",
            },
        )

        client = MagicMock()
        client.fetch_artifact_dir.return_value = dep_dir

        lockfile = Lockfile(tmp_path)
        # Mark installed at an older version
        lockfile.record_install(
            ref="skills/ag2ai/helper",
            version="1.0.0",
            targets=["claude"],
            files=[],
        )

        resolver = DependencyResolver(client, lockfile)
        root = _make_artifact("root", depends=["skills/ag2ai/helper"])
        result = resolver.resolve(root)

        # New version should be returned for installation
        assert len(result) == 1
        assert result[0].name == "helper"
        assert result[0].version == "2.0.0"


class TestTopologicalSort:
    """_topological_sort with simple chain and cycle detection."""

    def test_simple_chain(self, tmp_path: Path):
        from ag2_cli.install.lockfile import Lockfile
        from ag2_cli.install.resolver import DependencyResolver

        client = MagicMock()
        lockfile = Lockfile(tmp_path)
        resolver = DependencyResolver(client, lockfile)

        # A -> B -> C (A depends on B, B depends on C)
        graph = {
            "A": ["B"],
            "B": ["C"],
            "C": [],
        }
        result = resolver._topological_sort(graph)

        # After reversal: dependencies first => C, B, A
        assert result == ["C", "B", "A"]

    def test_simple_chain_longer(self, tmp_path: Path):
        from ag2_cli.install.lockfile import Lockfile
        from ag2_cli.install.resolver import DependencyResolver

        client = MagicMock()
        lockfile = Lockfile(tmp_path)
        resolver = DependencyResolver(client, lockfile)

        graph = {
            "A": ["B"],
            "B": ["C"],
            "C": ["D"],
            "D": [],
        }
        result = resolver._topological_sort(graph)

        assert result == ["D", "C", "B", "A"]

    def test_independent_nodes(self, tmp_path: Path):
        from ag2_cli.install.lockfile import Lockfile
        from ag2_cli.install.resolver import DependencyResolver

        client = MagicMock()
        lockfile = Lockfile(tmp_path)
        resolver = DependencyResolver(client, lockfile)

        graph = {
            "A": [],
            "B": [],
            "C": [],
        }
        result = resolver._topological_sort(graph)

        # All independent, sorted deterministically (alphabetical due to queue.sort())
        # After reversal: C, B, A
        assert result == ["C", "B", "A"]

    def test_cycle_detection_raises_error(self, tmp_path: Path):
        from ag2_cli.install.lockfile import Lockfile
        from ag2_cli.install.resolver import CyclicDependencyError, DependencyResolver

        client = MagicMock()
        lockfile = Lockfile(tmp_path)
        resolver = DependencyResolver(client, lockfile)

        # A -> B -> C -> A (cycle)
        graph = {
            "A": ["B"],
            "B": ["C"],
            "C": ["A"],
        }

        with pytest.raises(CyclicDependencyError, match="Circular dependency detected"):
            resolver._topological_sort(graph)

    def test_self_cycle_raises_error(self, tmp_path: Path):
        from ag2_cli.install.lockfile import Lockfile
        from ag2_cli.install.resolver import CyclicDependencyError, DependencyResolver

        client = MagicMock()
        lockfile = Lockfile(tmp_path)
        resolver = DependencyResolver(client, lockfile)

        graph = {
            "A": ["A"],
        }

        with pytest.raises(CyclicDependencyError):
            resolver._topological_sort(graph)


class TestFetchDependency:
    """_fetch_dependency with 2-part, 3-part, and invalid refs."""

    def test_two_part_ref(self, tmp_path: Path):
        from ag2_cli.install.lockfile import Lockfile
        from ag2_cli.install.resolver import DependencyResolver

        dep_dir = tmp_path / "cache" / "dep"
        _write_artifact_json(
            dep_dir,
            {
                "name": "my-skill",
                "type": "skill",
                "owner": "ag2ai",
                "version": "1.0.0",
            },
        )

        client = MagicMock()
        client.fetch_artifact_dir.return_value = dep_dir

        lockfile = Lockfile(tmp_path)
        resolver = DependencyResolver(client, lockfile)

        result = resolver._fetch_dependency("skills/my-skill")

        assert result is not None
        assert result.name == "my-skill"
        # Two-part ref: type=skills, name=my-skill, owner defaults to ag2ai
        client.fetch_artifact_dir.assert_called_once_with("skills", "my-skill", owner="ag2ai")

    def test_three_part_ref(self, tmp_path: Path):
        from ag2_cli.install.lockfile import Lockfile
        from ag2_cli.install.resolver import DependencyResolver

        dep_dir = tmp_path / "cache" / "dep"
        _write_artifact_json(
            dep_dir,
            {
                "name": "custom-tool",
                "type": "tool",
                "owner": "myorg",
                "version": "2.1.0",
            },
        )

        client = MagicMock()
        client.fetch_artifact_dir.return_value = dep_dir

        lockfile = Lockfile(tmp_path)
        resolver = DependencyResolver(client, lockfile)

        result = resolver._fetch_dependency("tools/myorg/custom-tool")

        assert result is not None
        assert result.name == "custom-tool"
        assert result.owner == "myorg"
        assert result.version == "2.1.0"
        client.fetch_artifact_dir.assert_called_once_with("tools", "custom-tool", owner="myorg")

    def test_invalid_ref_single_part_returns_none(self, tmp_path: Path):
        from ag2_cli.install.lockfile import Lockfile
        from ag2_cli.install.resolver import DependencyResolver

        client = MagicMock()
        lockfile = Lockfile(tmp_path)
        resolver = DependencyResolver(client, lockfile)

        result = resolver._fetch_dependency("invalid")

        assert result is None
        client.fetch_artifact_dir.assert_not_called()

    def test_invalid_ref_four_parts_returns_none(self, tmp_path: Path):
        from ag2_cli.install.lockfile import Lockfile
        from ag2_cli.install.resolver import DependencyResolver

        client = MagicMock()
        lockfile = Lockfile(tmp_path)
        resolver = DependencyResolver(client, lockfile)

        result = resolver._fetch_dependency("a/b/c/d")

        assert result is None
        client.fetch_artifact_dir.assert_not_called()

    def test_fetch_error_returns_none(self, tmp_path: Path):
        from ag2_cli.install.client import FetchError
        from ag2_cli.install.lockfile import Lockfile
        from ag2_cli.install.resolver import DependencyResolver

        client = MagicMock()
        client.fetch_artifact_dir.side_effect = FetchError("not found")

        lockfile = Lockfile(tmp_path)
        resolver = DependencyResolver(client, lockfile)

        result = resolver._fetch_dependency("skills/ag2ai/missing")

        assert result is None

    def test_missing_artifact_json_returns_none(self, tmp_path: Path):
        from ag2_cli.install.lockfile import Lockfile
        from ag2_cli.install.resolver import DependencyResolver

        # Return a directory that exists but has no artifact.json
        empty_dir = tmp_path / "cache" / "empty"
        empty_dir.mkdir(parents=True)

        client = MagicMock()
        client.fetch_artifact_dir.return_value = empty_dir

        lockfile = Lockfile(tmp_path)
        resolver = DependencyResolver(client, lockfile)

        result = resolver._fetch_dependency("skills/my-skill")

        assert result is None
