"""Install subsystem — manages artifacts, skill packs, and IDE targets."""

from .artifact import Artifact, InstallResult, load_artifact, load_artifact_json
from .client import ArtifactClient, FetchError
from .lockfile import Lockfile
from .registry import ContentItem, Pack, list_packs, load_pack, parse_frontmatter
from .resolver import DependencyResolver
from .targets import detect_targets, get_all_targets, get_target

__all__ = [
    # Artifact model
    "Artifact",
    "InstallResult",
    "load_artifact",
    "load_artifact_json",
    # Client
    "ArtifactClient",
    "FetchError",
    # Lockfile
    "Lockfile",
    # Legacy registry (backward compat)
    "ContentItem",
    "Pack",
    "list_packs",
    "load_pack",
    "parse_frontmatter",
    # Resolver
    "DependencyResolver",
    # Targets
    "detect_targets",
    "get_all_targets",
    "get_target",
]
