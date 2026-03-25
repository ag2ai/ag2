"""Artifact model and manifest parser for the ag2 artifact system."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VariableSpec:
    prompt: str
    default: str = ""
    transform: str | None = None
    choices: list[str] | None = None


@dataclass
class RemoteFile:
    name: str
    url: str
    size: str = ""
    sha256: str = ""


@dataclass
class SkillsConfig:
    dir: str = "skills/"
    auto_install: bool = True


@dataclass
class TemplateConfig:
    scaffold: str = "scaffold/"
    variables: dict[str, VariableSpec] = field(default_factory=dict)
    ignore: list[str] = field(default_factory=list)
    post_install: list[str] = field(default_factory=list)


@dataclass
class ToolConfig:
    kind: str = "ag2"  # "ag2" | "mcp"
    source: str = "src/"
    runtime: str | None = None
    entry_point: str | None = None
    transport: str | None = None
    mcp_config: dict | None = None
    functions: list[dict] = field(default_factory=list)
    tools_provided: list[dict] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    install_to: str = "tools/"


@dataclass
class DatasetConfig:
    inline: str | None = None
    remote: list[RemoteFile] = field(default_factory=list)
    format: str = "jsonl"
    schema: dict | None = None
    splits: dict[str, str] = field(default_factory=dict)
    eval_compatible: bool = False


@dataclass
class AgentConfig:
    source: str = "agent.md"
    model: str = "sonnet"
    tools: list[str] = field(default_factory=list)
    max_turns: int = 50
    memory: str = "project"
    mcp_servers: dict = field(default_factory=dict)
    preload_skills: list[str] = field(default_factory=list)


@dataclass
class BundleRef:
    ref: str
    required: bool = True


@dataclass
class BundleConfig:
    artifacts: list[BundleRef] = field(default_factory=list)
    install_order: list[str] = field(default_factory=lambda: ["skills", "tools", "templates", "datasets", "agents"])


DEFAULT_OWNER = "ag2ai"


@dataclass
class Artifact:
    name: str
    type: str  # "skills" | "template" | "tool" | "dataset" | "agent" | "bundle"
    owner: str = DEFAULT_OWNER
    display_name: str = ""
    description: str = ""
    version: str = "0.0.0"
    authors: list[str] = field(default_factory=list)
    license: str = ""
    tags: list[str] = field(default_factory=list)
    requires: dict[str, str] = field(default_factory=dict)
    depends: list[str] = field(default_factory=list)
    skills_config: SkillsConfig | None = None
    template: TemplateConfig | None = None
    tool: ToolConfig | None = None
    dataset: DatasetConfig | None = None
    agent: AgentConfig | None = None
    bundle: BundleConfig | None = None
    # Resolved at load time — not from JSON
    source_dir: Path | None = None

    @property
    def qualified_name(self) -> str:
        """Full owner/name identifier like 'ag2ai/web-search'."""
        return f"{self.owner}/{self.name}"

    @property
    def ref(self) -> str:
        """Canonical reference like 'tools/ag2ai/web-search'."""
        type_dir = _pluralize_type(self.type)
        return f"{type_dir}/{self.owner}/{self.name}"


def _pluralize_type(artifact_type: str) -> str:
    """Map artifact type to its plural directory name."""
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


def parse_artifact_id(artifact_id: str) -> tuple[str, str]:
    """Parse 'owner/name' or just 'name' into (owner, name).

    Examples:
        'web-search'         → ('ag2ai', 'web-search')
        'ag2ai/web-search'   → ('ag2ai', 'web-search')
        'myorg/custom-tool'  → ('myorg', 'custom-tool')
    """
    if "/" in artifact_id:
        owner, _, name = artifact_id.partition("/")
        return owner, name
    return DEFAULT_OWNER, artifact_id


@dataclass
class InstallResult:
    artifact: Artifact
    files_created: list[Path] = field(default_factory=list)
    targets_used: list[str] = field(default_factory=list)
    dependencies_installed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# -- Parsing --


def _parse_variable_spec(raw: dict) -> VariableSpec:
    return VariableSpec(
        prompt=raw.get("prompt", ""),
        default=raw.get("default", ""),
        transform=raw.get("transform"),
        choices=raw.get("choices"),
    )


def _parse_remote_file(raw: dict) -> RemoteFile:
    return RemoteFile(
        name=raw.get("name", ""),
        url=raw.get("url", ""),
        size=raw.get("size", ""),
        sha256=raw.get("sha256", ""),
    )


def _parse_template_config(raw: dict) -> TemplateConfig:
    variables = {}
    for k, v in raw.get("variables", {}).items():
        variables[k] = _parse_variable_spec(v) if isinstance(v, dict) else VariableSpec(prompt=k, default=str(v))
    return TemplateConfig(
        scaffold=raw.get("scaffold", "scaffold/"),
        variables=variables,
        ignore=raw.get("ignore", []),
        post_install=raw.get("post_install", []),
    )


def _parse_tool_config(raw: dict) -> ToolConfig:
    return ToolConfig(
        kind=raw.get("kind", "ag2"),
        source=raw.get("source", "src/"),
        runtime=raw.get("runtime"),
        entry_point=raw.get("entry_point"),
        transport=raw.get("transport"),
        mcp_config=raw.get("mcp_config"),
        functions=raw.get("functions", []),
        tools_provided=raw.get("tools_provided", []),
        requires=raw.get("requires", []),
        install_to=raw.get("install_to", "tools/"),
    )


def _parse_dataset_config(raw: dict) -> DatasetConfig:
    return DatasetConfig(
        inline=raw.get("inline"),
        remote=[_parse_remote_file(r) for r in raw.get("remote", [])],
        format=raw.get("format", "jsonl"),
        schema=raw.get("schema"),
        splits=raw.get("splits", {}),
        eval_compatible=raw.get("eval_compatible", False),
    )


def _parse_agent_config(raw: dict) -> AgentConfig:
    return AgentConfig(
        source=raw.get("source", "agent.md"),
        model=raw.get("model", "sonnet"),
        tools=raw.get("tools", []),
        max_turns=raw.get("max_turns", 50),
        memory=raw.get("memory", "project"),
        mcp_servers=raw.get("mcp_servers", {}),
        preload_skills=raw.get("preload_skills", []),
    )


def _parse_bundle_config(raw: dict) -> BundleConfig:
    artifacts = []
    for entry in raw.get("artifacts", []):
        if isinstance(entry, str):
            artifacts.append(BundleRef(ref=entry))
        else:
            artifacts.append(BundleRef(ref=entry["ref"], required=entry.get("required", True)))
    return BundleConfig(
        artifacts=artifacts,
        install_order=raw.get("install_order", ["skills", "tools", "templates", "datasets", "agents"]),
    )


def load_artifact_json(path: Path) -> Artifact:
    """Parse an artifact.json file into an Artifact."""
    raw = json.loads(path.read_text())
    artifact_type = raw.get("type", "skills")

    skills_cfg = None
    if "skills" in raw and isinstance(raw["skills"], dict):
        skills_cfg = SkillsConfig(
            dir=raw["skills"].get("dir", "skills/"),
            auto_install=raw["skills"].get("auto_install", True),
        )

    # Resolve owner: explicit field > first author > default
    owner = raw.get("owner", "")
    if not owner:
        authors = raw.get("authors", [])
        owner = authors[0] if authors else DEFAULT_OWNER

    artifact = Artifact(
        name=raw["name"],
        type=artifact_type,
        owner=owner,
        display_name=raw.get("display_name", raw["name"]),
        description=raw.get("description", ""),
        version=raw.get("version", "0.0.0"),
        authors=raw.get("authors", []),
        license=raw.get("license", ""),
        tags=raw.get("tags", []),
        requires=raw.get("requires", {}),
        depends=raw.get("depends", []),
        skills_config=skills_cfg,
        source_dir=path.parent,
    )

    if artifact_type == "template" and "template" in raw:
        artifact.template = _parse_template_config(raw["template"])
    elif artifact_type == "tool" and "tool" in raw:
        artifact.tool = _parse_tool_config(raw["tool"])
    elif artifact_type == "dataset" and "dataset" in raw:
        artifact.dataset = _parse_dataset_config(raw["dataset"])
    elif artifact_type == "agent" and "agent" in raw:
        artifact.agent = _parse_agent_config(raw["agent"])
    elif artifact_type == "bundle" and "bundle" in raw:
        artifact.bundle = _parse_bundle_config(raw["bundle"])

    return artifact


def load_legacy_manifest(pack_dir: Path) -> Artifact:
    """Adapt an existing manifest.json (bundled skills) into an Artifact."""
    manifest_path = pack_dir / "manifest.json"
    raw = json.loads(manifest_path.read_text())
    return Artifact(
        name=raw["name"],
        type="skills",
        display_name=raw.get("display_name", raw["name"]),
        description=raw.get("description", ""),
        version=raw.get("version", "0.0.0"),
        skills_config=SkillsConfig(dir=".", auto_install=True),
        source_dir=pack_dir,
    )


def load_artifact(path: Path) -> Artifact | None:
    """Load an artifact from a directory — tries artifact.json, falls back to manifest.json."""
    artifact_json = path / "artifact.json"
    if artifact_json.exists():
        return load_artifact_json(artifact_json)
    manifest_json = path / "manifest.json"
    if manifest_json.exists():
        return load_legacy_manifest(path)
    return None
