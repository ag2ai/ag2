# ag2 proxy

> Wrap any CLI tool, REST API, or Python module as AG2 tools automatically.

## Problem

The biggest bottleneck in agent development is creating tool wrappers.
Every API, CLI tool, and Python function needs a typed wrapper with
proper descriptions. This is mechanical, repetitive work.

## Commands

```bash
# Wrap CLI tool subcommands as AG2 tools
ag2 proxy cli git --subcommands "status,log,diff" --output tools/git_tools.py

# Wrap a REST API from its OpenAPI spec
ag2 proxy openapi https://api.example.com/openapi.json --output tools/api.py

# Wrap Python module functions
ag2 proxy module pandas --functions "read_csv,describe,to_json" --output tools/pandas_tools.py

# Wrap shell scripts in a directory
ag2 proxy scripts ./scripts/ --output tools/script_tools.py
```

## CLI Wrapping

```bash
ag2 proxy cli kubectl --subcommands "get pods,get services,logs,describe" --output tools/k8s.py
```

Generated:
```python
from autogen.tools import tool
import subprocess

@tool(name="kubectl_get_pods", description="List Kubernetes pods")
def kubectl_get_pods(namespace: str = "default", label: str | None = None) -> str:
    """List pods in a Kubernetes namespace.

    Args:
        namespace: Kubernetes namespace (default: "default")
        label: Optional label selector (e.g., "app=nginx")
    """
    cmd = ["kubectl", "get", "pods", "-n", namespace, "-o", "json"]
    if label:
        cmd.extend(["-l", label])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout
```

Options:
- `--subcommands` / `-s` — comma-separated subcommands to wrap
- `--output` / `-o` — output Python file (default: `tools_generated.py`)
- `--preview` — preview generated code without writing to disk

## OpenAPI Wrapping

```bash
ag2 proxy openapi https://api.example.com/openapi.json --output tools/api.py
ag2 proxy openapi ./openapi.yaml --endpoints "listUsers,getUser" --preview
```

Parses an OpenAPI/Swagger spec and generates one `@tool` function per
endpoint with:
- Proper parameter types from the schema
- Authentication via environment variables
- Error handling
- Response formatting

Options:
- `--endpoints` / `-e` — comma-separated operation IDs to include (default: all)
- `--output` / `-o` — output Python file (default: `tools_generated.py`)
- `--preview` — preview generated code without writing to disk

## Python Module Wrapping

```bash
ag2 proxy module requests --functions "get,post" --output tools/http_tools.py
ag2 proxy module json --functions "dumps,loads" --preview
```

Introspects the function signatures and docstrings to generate AG2 tools
with proper schemas.

Options:
- `--functions` / `-f` — comma-separated function names to wrap (default: all public)
- `--output` / `-o` — output Python file (default: `tools_generated.py`)
- `--preview` — preview generated code without writing to disk

## Shell Script Wrapping

```bash
ag2 proxy scripts ./scripts/ --output tools/script_tools.py
ag2 proxy scripts ./ops/ --preview
```

Wraps all shell scripts (`.sh`, `.bash`, `.zsh`, `.py`) and executable
files in a directory as AG2 tool functions using `subprocess.run()`.

Options:
- `--output` / `-o` — output Python file (default: `tools_generated.py`)
- `--preview` — preview generated code without writing to disk

## Implementation Notes

### CLI Introspection
1. Run `command --help` and parse the output
2. For subcommands, recursively parse each `command subcommand --help`
3. Extract: command name, description, flags, arguments
4. Generate typed Python wrappers using `subprocess.run()`

### OpenAPI Parsing
Use the `openapi-spec-validator` or just parse the JSON/YAML directly.
Map OpenAPI types to Python types:
- `string` → `str`
- `integer` → `int`
- `number` → `float`
- `boolean` → `bool`
- `array` → `list`
- `object` → `dict`

### Safety
All generated tools include:
- Timeouts on subprocess calls
- Error handling (non-zero exit codes, HTTP errors)
- No hardcoded credentials
- Read-only by default for database operations
