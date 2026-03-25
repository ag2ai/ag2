"""Tests for MCP configuration management."""

import json
from pathlib import Path


class TestConfigureMcpServer:
    """Test MCP server config writing."""

    def test_creates_claude_config(self, tmp_path: Path):
        from ag2_cli.install.mcp_config import configure_mcp_server

        paths = configure_mcp_server(
            project_dir=tmp_path,
            server_name="test-server",
            config={"command": "npx", "args": ["-y", "test-mcp"], "env": {}},
            ide_targets=["claude"],
        )

        assert len(paths) == 1
        config = json.loads((tmp_path / ".mcp.json").read_text())
        assert "mcpServers" in config
        assert "test-server" in config["mcpServers"]
        assert config["mcpServers"]["test-server"]["command"] == "npx"

    def test_creates_cursor_config(self, tmp_path: Path):
        from ag2_cli.install.mcp_config import configure_mcp_server

        configure_mcp_server(
            project_dir=tmp_path,
            server_name="my-mcp",
            config={"command": "uv", "args": ["run", "server.py"]},
            ide_targets=["cursor"],
        )

        config = json.loads((tmp_path / ".cursor" / "mcp.json").read_text())
        assert "mcpServers" in config
        assert "my-mcp" in config["mcpServers"]

    def test_creates_vscode_config_with_servers_key(self, tmp_path: Path):
        from ag2_cli.install.mcp_config import configure_mcp_server

        configure_mcp_server(
            project_dir=tmp_path,
            server_name="my-mcp",
            config={"command": "node", "args": ["server.js"]},
            ide_targets=["vscode"],
        )

        config = json.loads((tmp_path / ".vscode" / "mcp.json").read_text())
        # VS Code uses "servers" not "mcpServers"
        assert "servers" in config
        assert "my-mcp" in config["servers"]

    def test_merges_into_existing_config(self, tmp_path: Path):
        from ag2_cli.install.mcp_config import configure_mcp_server

        # Create existing config
        mcp_json = tmp_path / ".mcp.json"
        existing = {"mcpServers": {"existing-server": {"command": "node", "args": ["existing.js"]}}}
        mcp_json.write_text(json.dumps(existing))

        configure_mcp_server(
            project_dir=tmp_path,
            server_name="new-server",
            config={"command": "uv", "args": ["run", "new.py"]},
            ide_targets=["claude"],
        )

        config = json.loads(mcp_json.read_text())
        assert "existing-server" in config["mcpServers"]
        assert "new-server" in config["mcpServers"]

    def test_multiple_ide_targets(self, tmp_path: Path):
        from ag2_cli.install.mcp_config import configure_mcp_server

        paths = configure_mcp_server(
            project_dir=tmp_path,
            server_name="multi",
            config={"command": "test"},
            ide_targets=["claude", "cursor", "vscode"],
        )

        assert len(paths) == 3


class TestDetectMcpTargets:
    """Test MCP target auto-detection."""

    def test_detects_claude(self, tmp_path: Path):
        from ag2_cli.install.mcp_config import detect_mcp_targets

        (tmp_path / ".claude").mkdir()
        targets = detect_mcp_targets(tmp_path)
        assert "claude" in targets

    def test_detects_cursor(self, tmp_path: Path):
        from ag2_cli.install.mcp_config import detect_mcp_targets

        (tmp_path / ".cursor").mkdir()
        targets = detect_mcp_targets(tmp_path)
        assert "cursor" in targets

    def test_detects_multiple(self, tmp_path: Path):
        from ag2_cli.install.mcp_config import detect_mcp_targets

        (tmp_path / "CLAUDE.md").touch()
        (tmp_path / ".vscode").mkdir()
        targets = detect_mcp_targets(tmp_path)
        assert "claude" in targets
        assert "vscode" in targets

    def test_empty_project(self, tmp_path: Path):
        from ag2_cli.install.mcp_config import detect_mcp_targets

        targets = detect_mcp_targets(tmp_path)
        assert targets == []


class TestRemoveMcpServer:
    """Test MCP server removal."""

    def test_removes_server_entry(self, tmp_path: Path):
        from ag2_cli.install.mcp_config import configure_mcp_server, remove_mcp_server

        configure_mcp_server(tmp_path, "to-remove", {"command": "test"}, ide_targets=["claude"])
        configure_mcp_server(tmp_path, "to-keep", {"command": "test2"}, ide_targets=["claude"])

        removed = remove_mcp_server(tmp_path, "to-remove")
        assert len(removed) == 1

        config = json.loads((tmp_path / ".mcp.json").read_text())
        assert "to-remove" not in config["mcpServers"]
        assert "to-keep" in config["mcpServers"]
