---
sidebarTitle: Proxy
title: "ag2 proxy"
description: "Auto-generate AG2 tools from CLIs, REST APIs, or Python modules."
---

The `ag2 proxy` command wraps existing interfaces as AG2-compatible tools.

## CLI Wrapping

Generate typed Python wrappers for CLI commands:

```bash
ag2 proxy --command kubectl --subcommands "get pods,logs,describe"
```

The CLI introspects `--help` output to extract flags, arguments, and descriptions, then generates `@tool`-decorated functions that call the CLI via `subprocess`.

## OpenAPI Wrapping

Generate tools from an OpenAPI/Swagger specification:

```bash
ag2 proxy --openapi https://api.example.com/openapi.json
```

This parses the spec and generates one tool per endpoint with proper type hints, authentication handling via environment variables, and error handling.

## Python Module Wrapping

Wrap functions from a Python module:

```bash
ag2 proxy --module pandas --functions "read_csv,describe"
```

This introspects function signatures and generates tool wrappers with proper typing.

## MCP Serving

Instead of generating Python files, serve the wrapped tools as an MCP server:

```bash
ag2 proxy --command git --subcommands "status,log,diff" --serve-mcp
```

## Options

| Flag | Description |
|------|-------------|
| `--command` | CLI command to wrap |
| `--subcommands` | Comma-separated subcommands to include |
| `--openapi` | URL or path to OpenAPI spec |
| `--module` | Python module to wrap |
| `--functions` | Comma-separated functions to wrap |
| `--scripts` | Directory of shell scripts to wrap |
| `--serve-mcp` | Serve as MCP server instead of generating files |
| `--output` | Output directory for generated files |
