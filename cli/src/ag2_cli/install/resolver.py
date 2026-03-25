"""Dependency resolver — topological ordering of artifact dependencies."""

from __future__ import annotations

from .artifact import Artifact, load_artifact_json
from .client import ArtifactClient, FetchError
from .lockfile import Lockfile


class CyclicDependencyError(Exception):
    """Raised when a circular dependency is detected."""


class DependencyResolver:
    """Resolves artifact dependencies in install order."""

    def __init__(self, client: ArtifactClient, lockfile: Lockfile):
        self.client = client
        self.lockfile = lockfile

    def resolve(self, artifact: Artifact) -> list[Artifact]:
        """Resolve all dependencies of an artifact.

        Returns a list in topological order (dependencies first, root last).
        Skips artifacts already installed at the required version.
        """
        if not artifact.depends:
            return []

        # Collect all dependency manifests
        manifests: dict[str, Artifact] = {}
        graph: dict[str, list[str]] = {}
        self._collect(artifact, manifests, graph)

        # Remove already-installed artifacts
        to_install = []
        for ref in self._topological_sort(graph):
            if ref == artifact.ref:
                continue  # The root artifact is handled by the caller
            dep = manifests.get(ref)
            if dep and not self.lockfile.is_installed(ref, dep.version):
                to_install.append(dep)

        return to_install

    def _collect(
        self,
        artifact: Artifact,
        manifests: dict[str, Artifact],
        graph: dict[str, list[str]],
    ) -> None:
        """Recursively collect dependency manifests and build adjacency list."""
        ref = artifact.ref
        if ref in manifests:
            return
        manifests[ref] = artifact
        graph[ref] = []

        for dep_ref in artifact.depends:
            graph[ref].append(dep_ref)
            if dep_ref not in manifests:
                dep_artifact = self._fetch_dependency(dep_ref)
                if dep_artifact:
                    self._collect(dep_artifact, manifests, graph)

    def _fetch_dependency(self, ref: str) -> Artifact | None:
        """Fetch a dependency's manifest from the remote registry.

        Refs can be 2-part ("type/name") or 3-part ("type/owner/name").
        """
        parts = ref.split("/")
        if len(parts) == 3:
            artifact_type, owner, name = parts
        elif len(parts) == 2:
            artifact_type, name = parts
            owner = "ag2ai"
        else:
            return None
        try:
            cached_dir = self.client.fetch_artifact_dir(artifact_type, name, owner=owner)
            artifact_json = cached_dir / "artifact.json"
            if artifact_json.exists():
                return load_artifact_json(artifact_json)
        except FetchError:
            pass
        return None

    def _topological_sort(self, graph: dict[str, list[str]]) -> list[str]:
        """Kahn's algorithm for topological ordering."""
        # Compute in-degrees
        in_degree: dict[str, int] = dict.fromkeys(graph, 0)
        for node in graph:
            for dep in graph[node]:
                if dep not in in_degree:
                    in_degree[dep] = 0
                in_degree[dep] += 1

        # Start with nodes that have no incoming edges
        queue = [node for node, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            queue.sort()  # Deterministic ordering
            node = queue.pop(0)
            result.append(node)
            for dep in graph.get(node, []):
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

        if len(result) != len(in_degree):
            visited = set(result)
            cycle_nodes = [n for n in in_degree if n not in visited]
            raise CyclicDependencyError(f"Circular dependency detected involving: {', '.join(cycle_nodes)}")

        # Reverse: dependencies first, dependents last (install order)
        result.reverse()
        return result
