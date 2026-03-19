"""Tests for ag2 proxy — wrapping CLI/API/module as AG2 tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ag2_cli.app import app
from ag2_cli.commands.proxy import (
    ToolParam,
    ToolSpec,
    _generate_tool_file,
    _inspect_module_functions,
    _openapi_type_to_python,
    _parse_cli_help,
    _parse_openapi_spec,
    _python_type,
    _wrap_scripts,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def openapi_spec_file(tmp_path: Path) -> Path:
    """Create a minimal OpenAPI spec file."""
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "servers": [{"url": "https://api.example.com"}],
        "paths": {
            "/users": {
                "get": {
                    "operationId": "list_users",
                    "summary": "List all users",
                    "parameters": [
                        {
                            "name": "limit",
                            "in": "query",
                            "schema": {"type": "integer"},
                            "description": "Max results",
                            "required": False,
                        }
                    ],
                },
                "post": {
                    "operationId": "create_user",
                    "summary": "Create a user",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "User name",
                                        },
                                        "email": {
                                            "type": "string",
                                            "description": "User email",
                                        },
                                    },
                                    "required": ["name"],
                                }
                            }
                        }
                    },
                },
            },
            "/users/{user_id}": {
                "get": {
                    "operationId": "get_user",
                    "summary": "Get a user by ID",
                    "parameters": [
                        {
                            "name": "user_id",
                            "in": "path",
                            "schema": {"type": "integer"},
                            "required": True,
                        }
                    ],
                },
            },
        },
    }
    f = tmp_path / "openapi.json"
    f.write_text(json.dumps(spec))
    return f


@pytest.fixture
def scripts_dir(tmp_path: Path) -> Path:
    """Create a directory with sample scripts."""
    d = tmp_path / "scripts"
    d.mkdir()
    (d / "deploy.sh").write_text("#!/bin/bash\necho deploying")
    (d / "deploy.sh").chmod(0o755)
    (d / "check.py").write_text("#!/usr/bin/env python3\nprint('checking')")
    (d / "check.py").chmod(0o755)
    (d / "readme.txt").write_text("not a script")
    return d


# ---------------------------------------------------------------------------
# ToolSpec / ToolParam tests
# ---------------------------------------------------------------------------


class TestToolSpec:
    def test_basic_creation(self) -> None:
        spec = ToolSpec(name="test", description="A test tool")
        assert spec.name == "test"
        assert spec.params == []

    def test_with_params(self) -> None:
        spec = ToolSpec(
            name="test",
            description="A test tool",
            params=[
                ToolParam(name="query", type="str", description="Search query"),
                ToolParam(name="limit", type="int", required=False, default=10),
            ],
        )
        assert len(spec.params) == 2
        assert spec.params[1].default == 10


# ---------------------------------------------------------------------------
# Type mapping tests
# ---------------------------------------------------------------------------


class TestTypeMapping:
    def test_openapi_types(self) -> None:
        assert _openapi_type_to_python({"type": "string"}) == "str"
        assert _openapi_type_to_python({"type": "integer"}) == "int"
        assert _openapi_type_to_python({"type": "number"}) == "float"
        assert _openapi_type_to_python({"type": "boolean"}) == "bool"
        assert _openapi_type_to_python({"type": "array"}) == "list"
        assert _openapi_type_to_python({"type": "object"}) == "dict"
        assert _openapi_type_to_python({}) == "str"

    def test_python_type(self) -> None:
        assert _python_type("str") == "str"
        assert _python_type("int") == "int"
        assert _python_type("unknown") == "str"


# ---------------------------------------------------------------------------
# CLI parsing tests
# ---------------------------------------------------------------------------


class TestParseCLI:
    def test_parses_echo_help(self) -> None:
        # echo --help should work on most systems
        spec = _parse_cli_help("echo")
        assert spec.name == "echo"
        assert spec.source_type == "cli"

    def test_parses_subcommand(self) -> None:
        spec = _parse_cli_help("git", "status")
        assert "git" in spec.name
        assert "status" in spec.name
        assert spec.source_type == "cli"


# ---------------------------------------------------------------------------
# OpenAPI parsing tests
# ---------------------------------------------------------------------------


class TestParseOpenAPI:
    def test_parses_spec(self, openapi_spec_file: Path) -> None:
        spec = json.loads(openapi_spec_file.read_text())
        tools = _parse_openapi_spec(spec)
        assert len(tools) >= 3
        names = {t.name for t in tools}
        assert "list_users" in names
        assert "create_user" in names
        assert "get_user" in names

    def test_list_users_tool(self, openapi_spec_file: Path) -> None:
        spec = json.loads(openapi_spec_file.read_text())
        tools = _parse_openapi_spec(spec)
        list_users = next(t for t in tools if t.name == "list_users")
        assert list_users.description == "List all users"
        assert any(p.name == "limit" for p in list_users.params)
        assert list_users.source_type == "openapi"

    def test_path_params(self, openapi_spec_file: Path) -> None:
        spec = json.loads(openapi_spec_file.read_text())
        tools = _parse_openapi_spec(spec)
        get_user = next(t for t in tools if t.name == "get_user")
        assert any(p.name == "user_id" for p in get_user.params)

    def test_request_body_params(self, openapi_spec_file: Path) -> None:
        spec = json.loads(openapi_spec_file.read_text())
        tools = _parse_openapi_spec(spec)
        create_user = next(t for t in tools if t.name == "create_user")
        param_names = {p.name for p in create_user.params}
        assert "name" in param_names
        assert "email" in param_names

    def test_empty_spec(self) -> None:
        tools = _parse_openapi_spec({"paths": {}})
        assert len(tools) == 0


# ---------------------------------------------------------------------------
# Module inspection tests
# ---------------------------------------------------------------------------


class TestInspectModule:
    def test_inspect_json_module(self) -> None:
        tools = _inspect_module_functions("json", ["dumps", "loads"])
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "json_dumps" in names
        assert "json_loads" in names

    def test_inspect_with_filter(self) -> None:
        tools = _inspect_module_functions("json", ["dumps"])
        assert len(tools) == 1
        assert tools[0].name == "json_dumps"

    def test_inspect_has_params(self) -> None:
        tools = _inspect_module_functions("json", ["dumps"])
        assert len(tools[0].params) > 0

    def test_inspect_skips_private(self) -> None:
        tools = _inspect_module_functions("json")
        names = {t.name for t in tools}
        assert not any("_" == n[0] for n in names if n.startswith("json_"))


# ---------------------------------------------------------------------------
# Script wrapping tests
# ---------------------------------------------------------------------------


class TestWrapScripts:
    def test_wraps_executable_scripts(self, scripts_dir: Path) -> None:
        tools = _wrap_scripts(scripts_dir)
        names = {t.name for t in tools}
        assert "deploy" in names
        assert "check" in names
        # readme.txt should not be included (not executable, wrong extension)
        assert "readme" not in names

    def test_script_tool_has_args_param(self, scripts_dir: Path) -> None:
        tools = _wrap_scripts(scripts_dir)
        for t in tools:
            assert any(p.name == "args" for p in t.params)


# ---------------------------------------------------------------------------
# Code generation tests
# ---------------------------------------------------------------------------


class TestGenerateToolFile:
    def test_generates_valid_python(self, tmp_path: Path) -> None:
        tools = [
            ToolSpec(
                name="my_tool",
                description="A test tool",
                params=[
                    ToolParam(name="query", type="str", description="Search query"),
                    ToolParam(name="limit", type="int", required=False, default=10),
                ],
                source_type="cli",
                implementation='return f"result for {query}, limit={limit}"',
            )
        ]
        output = tmp_path / "tools.py"
        content = _generate_tool_file(tools, output)
        assert output.exists()
        assert "def my_tool(" in content
        assert "query: str" in content
        assert "limit: int = 10" in content
        assert "A test tool" in content

    def test_generates_optional_params(self, tmp_path: Path) -> None:
        tools = [
            ToolSpec(
                name="opt_tool",
                description="Tool with optional params",
                params=[
                    ToolParam(name="name", type="str", required=False, default=None),
                    ToolParam(name="verbose", type="bool", required=False, default=False),
                ],
                source_type="cli",
                implementation="return 'ok'",
            )
        ]
        output = tmp_path / "tools.py"
        content = _generate_tool_file(tools, output)
        assert "str | None = None" in content
        assert "bool = False" in content


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestProxyCLI:
    def test_cli_preview(self) -> None:
        result = runner.invoke(
            app, ["proxy", "cli", "echo", "--preview"]
        )
        assert result.exit_code == 0
        assert "echo" in result.output

    def test_cli_with_subcommands(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "proxy", "cli", "git",
                "--subcommands", "status",
                "--output", str(tmp_path / "tools.py"),
            ],
        )
        assert result.exit_code == 0
        assert (tmp_path / "tools.py").exists()


class TestProxyOpenAPI:
    def test_openapi_preview(self, openapi_spec_file: Path) -> None:
        result = runner.invoke(
            app, ["proxy", "openapi", str(openapi_spec_file), "--preview"]
        )
        assert result.exit_code == 0
        assert "list_users" in result.output

    def test_openapi_generates_file(
        self, openapi_spec_file: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "api_tools.py"
        result = runner.invoke(
            app,
            ["proxy", "openapi", str(openapi_spec_file), "--output", str(output)],
        )
        assert result.exit_code == 0
        assert output.exists()
        content = output.read_text()
        assert "def list_users" in content
        assert "def create_user" in content

    def test_openapi_filter_endpoints(
        self, openapi_spec_file: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "filtered.py"
        result = runner.invoke(
            app,
            [
                "proxy", "openapi", str(openapi_spec_file),
                "--endpoints", "list_users",
                "--output", str(output),
            ],
        )
        assert result.exit_code == 0
        content = output.read_text()
        assert "def list_users" in content
        assert "def create_user" not in content


class TestProxyModule:
    def test_module_preview(self) -> None:
        result = runner.invoke(
            app,
            ["proxy", "module", "json", "--functions", "dumps,loads", "--preview"],
        )
        assert result.exit_code == 0
        assert "json_dumps" in result.output
        assert "json_loads" in result.output

    def test_module_generates_file(self, tmp_path: Path) -> None:
        output = tmp_path / "json_tools.py"
        result = runner.invoke(
            app,
            [
                "proxy", "module", "json",
                "--functions", "dumps",
                "--output", str(output),
            ],
        )
        assert result.exit_code == 0
        assert output.exists()


class TestProxyScripts:
    def test_scripts_preview(self, scripts_dir: Path) -> None:
        result = runner.invoke(
            app, ["proxy", "scripts", str(scripts_dir), "--preview"]
        )
        assert result.exit_code == 0
        assert "deploy" in result.output

    def test_scripts_generates_file(
        self, scripts_dir: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "script_tools.py"
        result = runner.invoke(
            app,
            ["proxy", "scripts", str(scripts_dir), "--output", str(output)],
        )
        assert result.exit_code == 0
        assert output.exists()
