"""ag2 proxy — wrap CLI tools, REST APIs, and Python modules as AG2 tools.

Auto-generates AG2-compatible tool functions from existing interfaces,
making any CLI command, OpenAPI endpoint, or Python function instantly
usable by agents.
"""

from __future__ import annotations

import importlib
import inspect
import json
import keyword
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer
from rich.syntax import Syntax
from rich.table import Table

from ..ui import console

app = typer.Typer(
    help="Wrap CLI tools, REST APIs, and Python modules as AG2 tools.",
    rich_markup_mode="rich",
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ToolParam:
    """A parameter for a generated tool."""

    name: str
    type: str = "str"
    description: str = ""
    required: bool = True
    default: Any = None


@dataclass
class ToolSpec:
    """Specification for a tool to generate."""

    name: str
    description: str
    params: list[ToolParam] = field(default_factory=list)
    source_type: str = ""  # "cli", "openapi", "module", "script"
    implementation: str = ""  # code body


# ---------------------------------------------------------------------------
# CLI wrapping
# ---------------------------------------------------------------------------


def _parse_cli_help(command: str, subcommand: str | None = None) -> ToolSpec:
    """Parse --help output from a CLI command to build a ToolSpec."""
    cmd_parts = [command]
    if subcommand:
        cmd_parts.append(subcommand)

    # Get help text
    try:
        result = subprocess.run(
            [*cmd_parts, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        help_text = result.stdout or result.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise typer.Exit(1) from exc

    # Strip man-page overstrike formatting (X\bX for bold, _\bX for underline)
    help_text = re.sub(r".\x08", "", help_text)

    tool_name = "_".join(cmd_parts).replace("-", "_")
    description = ""
    params: list[ToolParam] = []

    # Extract description — skip man page headers like "GIT-STATUS(1) ... GIT-STATUS(1)"
    # and look for the NAME section description or first meaningful line
    man_header_re = re.compile(r"^[A-Z\-]+\(\d+\)")
    lines_iter = help_text.split("\n")
    in_name_section = False
    for line in lines_iter:
        stripped = line.strip()
        if stripped == "NAME":
            in_name_section = True
            continue
        if in_name_section and stripped:
            # NAME section line like "git-status - Show the working tree status"
            description = stripped.split(" - ", 1)[1].strip() if " - " in stripped else stripped
            break
        if (
            stripped
            and not stripped.startswith("Usage")
            and not stripped.startswith("-")
            and not man_header_re.match(stripped)
        ):
            description = stripped
            break

    if not description:
        description = f"Run {' '.join(cmd_parts)}"

    # Parse flags/options from help text
    flag_pattern = re.compile(
        r"^\s+(-\w,?\s+)?(--[a-zA-Z][\w-]*)(?:\s+(\w+))?\s{2,}(.+)",
        re.MULTILINE,
    )
    seen_params: set[str] = set()
    for match in flag_pattern.finditer(help_text):
        long_flag = match.group(2)
        arg_name = match.group(3)
        flag_desc = match.group(4).strip()

        param_name = long_flag.lstrip("-").replace("-", "_")

        # Skip invalid Python identifiers and keywords
        if not param_name.isidentifier() or keyword.iskeyword(param_name):
            continue

        # Skip duplicates
        if param_name in seen_params:
            continue
        seen_params.add(param_name)

        param_type = "str" if arg_name else "bool"
        params.append(
            ToolParam(
                name=param_name,
                type=param_type,
                description=flag_desc,
                required=False,
                default=False if param_type == "bool" else None,
            )
        )

    # Build implementation
    impl_lines = [
        f'cmd = ["{command}"' + (f', "{subcommand}"' if subcommand else "") + "]",
    ]
    for p in params:
        flag = f"--{p.name.replace('_', '-')}"
        if p.type == "bool":
            impl_lines.append(f'if {p.name}:\n        cmd.append("{flag}")')
        else:
            impl_lines.append(f'if {p.name} is not None:\n        cmd.extend(["{flag}", str({p.name})])')

    impl_lines.extend([
        "result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)",
        "if result.returncode != 0:",
        '    return f"Error: {result.stderr}"',
        "return result.stdout",
    ])

    return ToolSpec(
        name=tool_name,
        description=description,
        params=params,
        source_type="cli",
        implementation="\n    ".join(impl_lines),
    )


# ---------------------------------------------------------------------------
# OpenAPI wrapping
# ---------------------------------------------------------------------------


def _load_openapi_spec(source: str) -> dict[str, Any]:
    """Load an OpenAPI spec from a URL or file path."""
    import yaml

    path = Path(source)
    if path.exists():
        with open(path) as f:
            if path.suffix in (".yaml", ".yml"):
                return yaml.safe_load(f)
            return json.load(f)

    # Try as URL
    import httpx

    resp = httpx.get(source, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "")
    if "yaml" in content_type or source.endswith((".yaml", ".yml")):
        return yaml.safe_load(resp.text)
    return resp.json()


def _openapi_type_to_python(schema: dict[str, Any]) -> str:
    """Map OpenAPI type to Python type hint."""
    t = schema.get("type", "string")
    mapping = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "array": "list",
        "object": "dict",
    }
    return mapping.get(t, "str")


def _parse_openapi_spec(spec: dict[str, Any]) -> list[ToolSpec]:
    """Parse OpenAPI spec into tool specifications."""
    tools: list[ToolSpec] = []
    base_url = ""

    # Extract base URL
    servers = spec.get("servers", [])
    if servers:
        base_url = servers[0].get("url", "")

    paths = spec.get("paths", {})
    for path, methods in paths.items():
        for method, operation in methods.items():
            if method in ("parameters", "summary", "description"):
                continue
            if not isinstance(operation, dict):
                continue

            op_id = operation.get("operationId", "")
            if not op_id:
                op_id = f"{method}_{path}".replace("/", "_").replace("{", "").replace("}", "")
            tool_name = re.sub(r"[^a-zA-Z0-9_]", "_", op_id).strip("_")

            description = operation.get("summary", "") or operation.get("description", "")
            if not description:
                description = f"{method.upper()} {path}"

            params: list[ToolParam] = []

            # Path and query parameters
            for param in operation.get("parameters", []):
                schema = param.get("schema", {})
                params.append(
                    ToolParam(
                        name=param["name"],
                        type=_openapi_type_to_python(schema),
                        description=param.get("description", ""),
                        required=param.get("required", False),
                        default=schema.get("default"),
                    )
                )

            # Request body
            request_body = operation.get("requestBody", {})
            if request_body:
                content = request_body.get("content", {})
                json_schema = content.get("application/json", {}).get("schema", {})
                if json_schema.get("properties"):
                    for prop_name, prop_schema in json_schema["properties"].items():
                        required_props = json_schema.get("required", [])
                        params.append(
                            ToolParam(
                                name=prop_name,
                                type=_openapi_type_to_python(prop_schema),
                                description=prop_schema.get("description", ""),
                                required=prop_name in required_props,
                            )
                        )

            # Build implementation
            url = f"{base_url}{path}"
            path_params = re.findall(r"\{(\w+)\}", path)
            impl_lines = ["import httpx"]

            url_expr = f'f"{url}"' if path_params else f'"{url}"'
            impl_lines.append(f"url = {url_expr}")

            # Build query params
            query_params = [p for p in params if p.name not in path_params]
            body_params = []
            if request_body:
                body_params = [
                    p
                    for p in query_params
                    if p.name
                    in list(
                        request_body
                        .get("content", {})
                        .get("application/json", {})
                        .get("schema", {})
                        .get("properties", {})
                    )
                ]
                query_params = [p for p in query_params if p not in body_params]

            if query_params:
                impl_lines.append(
                    "params = {"
                    + ", ".join(f'"{p.name}": {p.name}' for p in query_params if p.name not in path_params)
                    + "}"
                )
                impl_lines.append("params = {k: v for k, v in params.items() if v is not None}")

            if body_params:
                impl_lines.append("body = {" + ", ".join(f'"{p.name}": {p.name}' for p in body_params) + "}")

            req_args = "url"
            if query_params:
                req_args += ", params=params"
            if body_params:
                req_args += ", json=body"
            impl_lines.append(f"resp = httpx.{method}({req_args}, timeout=30)")
            impl_lines.append("resp.raise_for_status()")
            impl_lines.append("return resp.text")

            tools.append(
                ToolSpec(
                    name=tool_name,
                    description=description[:200],
                    params=params,
                    source_type="openapi",
                    implementation="\n    ".join(impl_lines),
                )
            )

    return tools


# ---------------------------------------------------------------------------
# Python module wrapping
# ---------------------------------------------------------------------------


def _inspect_module_functions(module_name: str, function_names: list[str] | None = None) -> list[ToolSpec]:
    """Inspect a Python module and extract tool specs from functions."""
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        console.print(f"[error]Cannot import module: {module_name}[/error]")
        raise typer.Exit(1)

    tools: list[ToolSpec] = []

    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if function_names and name not in function_names:
            continue
        if name.startswith("_"):
            continue

        sig = inspect.signature(obj)
        doc = inspect.getdoc(obj) or f"Call {module_name}.{name}"

        params: list[ToolParam] = []
        for pname, param in sig.parameters.items():
            if pname in ("self", "cls"):
                continue
            ptype = "str"
            if param.annotation != inspect.Parameter.empty:
                ptype = getattr(param.annotation, "__name__", str(param.annotation))
            required = param.default is inspect.Parameter.empty
            default = None if required else param.default

            params.append(
                ToolParam(
                    name=pname,
                    type=ptype,
                    description="",
                    required=required,
                    default=default,
                )
            )

        # Build implementation
        args_str = ", ".join(f"{p.name}={p.name}" for p in params)
        impl = f"import {module_name}\n    return {module_name}.{name}({args_str})"

        tools.append(
            ToolSpec(
                name=f"{module_name.replace('.', '_')}_{name}",
                description=doc.split("\n")[0][:200],
                params=params,
                source_type="module",
                implementation=impl,
            )
        )

    return tools


# ---------------------------------------------------------------------------
# Script wrapping
# ---------------------------------------------------------------------------


def _wrap_scripts(scripts_dir: Path) -> list[ToolSpec]:
    """Wrap shell scripts in a directory as AG2 tools."""
    scripts_dir = scripts_dir.resolve()
    if not scripts_dir.is_dir():
        console.print(f"[error]Not a directory: {scripts_dir}[/error]")
        raise typer.Exit(1)

    tools: list[ToolSpec] = []
    for script in sorted(scripts_dir.iterdir()):
        if script.is_file() and (
            script.suffix in (".sh", ".bash", ".zsh", ".py") or script.stat().st_mode & 0o111  # executable
        ):
            name = script.stem.replace("-", "_").replace(" ", "_")
            tools.append(
                ToolSpec(
                    name=name,
                    description=f"Run script: {script.name}",
                    params=[
                        ToolParam(
                            name="args",
                            type="str",
                            description="Arguments to pass to the script.",
                            required=False,
                            default="",
                        )
                    ],
                    source_type="script",
                    implementation=(
                        f"import shlex\n"
                        f"    cmd = [{repr(str(script))}] + (shlex.split(args) if args else [])\n"
                        f"    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)\n"
                        f"    if result.returncode != 0:\n"
                        f'        return f"Error: {{result.stderr}}"\n'
                        f"    return result.stdout"
                    ),
                )
            )

    return tools


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------


def _python_type(t: str) -> str:
    """Normalize type string for Python annotation."""
    mapping = {"str": "str", "int": "int", "float": "float", "bool": "bool", "list": "list", "dict": "dict"}
    return mapping.get(t, "str")


def _generate_tool_file(tools: list[ToolSpec], output_path: Path, *, write: bool = True) -> str:
    """Generate a Python file with AG2 tool functions."""
    imports = {"from __future__ import annotations", "import subprocess"}

    lines = [
        '"""Auto-generated AG2 tools by `ag2 proxy`."""\n',
    ]

    # Collect imports from implementations
    for tool in tools:
        if "import httpx" in tool.implementation:
            imports.add("import httpx")
        if "import shlex" in tool.implementation:
            imports.add("import shlex")

    lines.insert(1, "\n".join(sorted(imports)) + "\n")

    for tool in tools:
        # Build function signature
        param_strs: list[str] = []
        for p in tool.params:
            pt = _python_type(p.type)
            if p.required:
                param_strs.append(f"{p.name}: {pt}")
            elif p.default is None:
                param_strs.append(f"{p.name}: {pt} | None = None")
            elif isinstance(p.default, bool):
                param_strs.append(f"{p.name}: {pt} = {p.default}")
            elif isinstance(p.default, str):
                param_strs.append(f'{p.name}: {pt} = "{p.default}"')
            else:
                param_strs.append(f"{p.name}: {pt} = {p.default}")

        sig = ", ".join(param_strs)

        lines.append(f"\ndef {tool.name}({sig}) -> str:")
        lines.append(f'    """{tool.description}"""')
        lines.append(f"    {tool.implementation}")
        lines.append("")

    content = "\n".join(lines)

    if write:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
    return content


def _display_tools(tools: list[ToolSpec]) -> None:
    """Display generated tools summary."""
    table = Table(show_edge=False, pad_edge=True)
    table.add_column("Tool", style="bold")
    table.add_column("Source", style="dim")
    table.add_column("Params", justify="right")
    table.add_column("Description")

    for t in tools:
        table.add_row(t.name, t.source_type, str(len(t.params)), t.description[:50])

    console.print(table)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("cli")
def proxy_cli(
    command: str = typer.Argument(..., help="CLI command to wrap (e.g., 'git')."),
    subcommands: str = typer.Option(
        None,
        "--subcommands",
        "-s",
        help="Comma-separated subcommands to wrap (e.g., 'status,log,diff').",
    ),
    output: Path = typer.Option(
        Path("tools_generated.py"),
        "--output",
        "-o",
        help="Output Python file.",
    ),
    preview: bool = typer.Option(False, "--preview", help="Preview generated code without writing."),
) -> None:
    """Wrap a CLI tool as AG2 tool functions.

    [dim]Examples:[/dim]
      [command]ag2 proxy cli git --subcommands status,log,diff[/command]
      [command]ag2 proxy cli kubectl --subcommands "get,describe,logs" --preview[/command]
    """
    cmds = [s.strip() for s in subcommands.split(",") if s.strip()] if subcommands else [None]

    tools: list[ToolSpec] = []
    for sub in cmds:
        console.print(f"[dim]Parsing {command}" + (f" {sub}" if sub else "") + " --help...[/dim]")
        try:
            tools.append(_parse_cli_help(command, sub))
        except typer.Exit:
            console.print(f"[warning]Could not parse help for {command}" + (f" {sub}" if sub else "") + "[/warning]")

    if not tools:
        console.print("[error]No tools generated.[/error]")
        raise typer.Exit(1)

    console.print()
    _display_tools(tools)
    console.print()

    if preview:
        content = _generate_tool_file(tools, output, write=False)
        console.print(Syntax(content, "python", theme="monokai", line_numbers=True))
    else:
        _generate_tool_file(tools, output)
        console.print(f"[success]Generated {len(tools)} tools in {output}[/success]")


@app.command("openapi")
def proxy_openapi(
    source: str = typer.Argument(..., help="OpenAPI spec URL or file path."),
    endpoints: str | None = typer.Option(
        None,
        "--endpoints",
        "-e",
        help="Comma-separated operation IDs to include (default: all).",
    ),
    output: Path = typer.Option(
        Path("tools_generated.py"),
        "--output",
        "-o",
        help="Output Python file.",
    ),
    preview: bool = typer.Option(False, "--preview", help="Preview generated code without writing."),
) -> None:
    """Wrap a REST API as AG2 tool functions from its OpenAPI spec.

    [dim]Examples:[/dim]
      [command]ag2 proxy openapi ./openapi.yaml[/command]
      [command]ag2 proxy openapi https://api.example.com/openapi.json --preview[/command]
    """
    console.print(f"[dim]Loading OpenAPI spec from {source}...[/dim]")
    try:
        spec = _load_openapi_spec(source)
    except Exception as exc:
        console.print(f"[error]Failed to load spec: {exc}[/error]")
        raise typer.Exit(1)

    tools = _parse_openapi_spec(spec)

    if endpoints:
        allowed = {e.strip() for e in endpoints.split(",")}
        tools = [t for t in tools if t.name in allowed]

    if not tools:
        console.print("[error]No tools generated from the spec.[/error]")
        raise typer.Exit(1)

    console.print()
    _display_tools(tools)
    console.print()

    if preview:
        content = _generate_tool_file(tools, output, write=False)
        console.print(Syntax(content, "python", theme="monokai", line_numbers=True))
    else:
        _generate_tool_file(tools, output)
        console.print(f"[success]Generated {len(tools)} tools in {output}[/success]")


@app.command("module")
def proxy_module(
    module_name: str = typer.Argument(..., help="Python module to wrap (e.g., 'json', 'pandas')."),
    functions: str | None = typer.Option(
        None,
        "--functions",
        "-f",
        help="Comma-separated function names to wrap (default: all public).",
    ),
    output: Path = typer.Option(
        Path("tools_generated.py"),
        "--output",
        "-o",
        help="Output Python file.",
    ),
    preview: bool = typer.Option(False, "--preview", help="Preview generated code without writing."),
) -> None:
    """Wrap Python module functions as AG2 tool functions.

    [dim]Examples:[/dim]
      [command]ag2 proxy module json --functions "dumps,loads"[/command]
      [command]ag2 proxy module pandas --functions "read_csv" --preview[/command]
    """
    func_list = [f.strip() for f in functions.split(",") if f.strip()] if functions else None

    console.print(f"[dim]Inspecting module {module_name}...[/dim]")
    tools = _inspect_module_functions(module_name, func_list)

    if not tools:
        console.print(f"[error]No public functions found in {module_name}.[/error]")
        raise typer.Exit(1)

    console.print()
    _display_tools(tools)
    console.print()

    if preview:
        content = _generate_tool_file(tools, output, write=False)
        console.print(Syntax(content, "python", theme="monokai", line_numbers=True))
    else:
        _generate_tool_file(tools, output)
        console.print(f"[success]Generated {len(tools)} tools in {output}[/success]")


@app.command("scripts")
def proxy_scripts(
    scripts_dir: Path = typer.Argument(..., help="Directory containing shell scripts."),
    output: Path = typer.Option(
        Path("tools_generated.py"),
        "--output",
        "-o",
        help="Output Python file.",
    ),
    preview: bool = typer.Option(False, "--preview", help="Preview generated code without writing."),
) -> None:
    """Wrap shell scripts in a directory as AG2 tool functions.

    [dim]Examples:[/dim]
      [command]ag2 proxy scripts ./scripts/[/command]
      [command]ag2 proxy scripts ./ops/ --preview[/command]
    """
    tools = _wrap_scripts(scripts_dir)

    if not tools:
        console.print(f"[error]No scripts found in {scripts_dir}.[/error]")
        raise typer.Exit(1)

    console.print()
    _display_tools(tools)
    console.print()

    if preview:
        content = _generate_tool_file(tools, output, write=False)
        console.print(Syntax(content, "python", theme="monokai", line_numbers=True))
    else:
        _generate_tool_file(tools, output)
        console.print(f"[success]Generated {len(tools)} tools in {output}[/success]")
