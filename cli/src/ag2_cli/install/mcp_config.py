"""MCP configuration — auto-configures MCP servers across IDEs."""

from __future__ import annotations

import json
from pathlib import Path

# IDE -> (config file path, top-level key for server entries)
MCP_IDE_CONFIGS: dict[str, tuple[str, str]] = {
    "claude": (".mcp.json", "mcpServers"),
    "cursor": (".cursor/mcp.json", "mcpServers"),
    "vscode": (".vscode/mcp.json", "servers"),
}


def detect_mcp_targets(project_dir: Path) -> list[str]:
    """Detect which IDEs with MCP support are present in the project."""
    found = []
    detectors = {
        "claude": ["CLAUDE.md", ".claude"],
        "cursor": [".cursor"],
        "vscode": [".vscode"],
    }
    for ide, paths in detectors.items():
        if any((project_dir / p).exists() for p in paths):
            found.append(ide)
    return found


def configure_mcp_server(
    project_dir: Path,
    server_name: str,
    config: dict,
    ide_targets: list[str] | None = None,
) -> list[Path]:
    """Add an MCP server entry to IDE config files.

    Args:
        project_dir: Project root directory.
        server_name: Name for the MCP server entry (e.g. "github-mcp").
        config: Server config dict with command/args/env (or url/headers for HTTP).
        ide_targets: Specific IDEs to configure. If None, auto-detects.

    Returns:
        List of config files that were written.
    """
    if ide_targets is None:
        ide_targets = detect_mcp_targets(project_dir)
    if not ide_targets:
        # Default to Claude Code
        ide_targets = ["claude"]

    written = []
    for ide in ide_targets:
        if ide not in MCP_IDE_CONFIGS:
            continue
        config_path_str, server_key = MCP_IDE_CONFIGS[ide]
        config_path = project_dir / config_path_str

        existing = _read_json(config_path)

        # Ensure the top-level servers object exists
        if server_key not in existing:
            existing[server_key] = {}
        existing[server_key][server_name] = config

        _write_json(config_path, existing)
        written.append(config_path)

    return written


def remove_mcp_server(
    project_dir: Path,
    server_name: str,
) -> list[Path]:
    """Remove an MCP server entry from all IDE config files."""
    removed = []
    for ide, (config_path_str, server_key) in MCP_IDE_CONFIGS.items():
        config_path = project_dir / config_path_str
        if not config_path.exists():
            continue
        existing = _read_json(config_path)
        servers = existing.get(server_key, {})
        if server_name in servers:
            del servers[server_name]
            _write_json(config_path, existing)
            removed.append(config_path)
    return removed


def _read_json(path: Path) -> dict:
    """Read a JSON file, returning empty dict if it doesn't exist or is invalid."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _write_json(path: Path, data: dict) -> None:
    """Write JSON atomically (write to temp, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.rename(path)
