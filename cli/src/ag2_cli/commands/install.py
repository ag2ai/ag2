"""ag2 install — install skills, templates, tools, datasets, agents, and bundles."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import typer
from rich.panel import Panel
from rich.table import Table

from ..install import detect_targets, get_all_targets, get_target, load_pack
from ..install.artifact import InstallResult
from ..install.client import ArtifactClient, FetchError
from ..install.lockfile import Lockfile
from ..install.resolver import DependencyResolver
from ..ui import console

app = typer.Typer(
    help="Install skills, templates, tools, datasets, agents, and bundles.",
    rich_markup_mode="rich",
)


# ---------------------------------------------------------------------------
# Helper: create the installer stack (shared across commands)
# ---------------------------------------------------------------------------


def _make_installers(project_dir: Path):
    """Create all installer instances with shared client, lockfile, and resolver."""
    from ..install.installers.agents import AgentInstaller
    from ..install.installers.bundles import BundleInstaller
    from ..install.installers.datasets import DatasetInstaller
    from ..install.installers.skills import SkillsInstaller
    from ..install.installers.templates import TemplateInstaller
    from ..install.installers.tools import ToolInstaller

    client = ArtifactClient()
    lockfile = Lockfile(project_dir)
    resolver = DependencyResolver(client, lockfile)

    skills = SkillsInstaller(client, lockfile, resolver)
    templates = TemplateInstaller(client, lockfile, resolver, skills)
    tools = ToolInstaller(client, lockfile, resolver, skills)
    datasets = DatasetInstaller(client, lockfile, resolver, skills)
    agents = AgentInstaller(client, lockfile, resolver, skills)
    bundles = BundleInstaller(client, lockfile, resolver, skills, templates, tools, datasets, agents)

    return {
        "client": client,
        "lockfile": lockfile,
        "resolver": resolver,
        "skills": skills,
        "templates": templates,
        "tools": tools,
        "datasets": datasets,
        "agents": agents,
        "bundles": bundles,
    }


# ---------------------------------------------------------------------------
# ag2 install skills
# ---------------------------------------------------------------------------


@app.command("skills")
def install_skills(
    packs: list[str] | None = typer.Argument(None, help="Skill packs to install (default: ag2)."),
    name: str | None = typer.Option(None, "--name", "-n", help="Install a specific item by name."),
    target: str | None = typer.Option(None, "--target", "-t", help='Target IDE/agent (comma-separated, or "all").'),
    project_dir: Path = typer.Option(Path("."), "--project-dir", "-d", help="Project directory."),
    refresh: bool = typer.Option(False, "--refresh", help="Force refresh registry cache."),
) -> None:
    """Install AI agent skills (rules, workflows, agents) into your IDE.

    [dim]Skills are not limited to AG2 — install skills for any framework.
    Auto-detects your IDE when --target is omitted.[/dim]
    """
    project = project_dir.resolve()
    pack_names = packs or ["ag2"]
    targets = _resolve_targets(target, project)
    stack = _make_installers(project)

    if refresh:
        with contextlib.suppress(FetchError):
            stack["client"].fetch_registry(force_refresh=True)

    console.print(f"\n[heading]Installing skills[/heading] ({', '.join(pack_names)})\n")

    try:
        results = stack["skills"].install(pack_names, targets, project, name_filter=name)
    except FetchError as e:
        console.print(f"[error]{e}[/error]")
        raise typer.Exit(1)

    total = 0
    for result in results:
        n = len(result.files_created)
        total += n
        console.print(f"  [success]\u2713[/success] {result.artifact.display_name or result.artifact.name} — {n} files")
        for tgt_name in result.targets_used:
            tgt = get_target(tgt_name)
            label = tgt.display_name if tgt else tgt_name
            console.print(f"    {label}", style="dim")

    console.print(f"\n[success]Done.[/success] {total} files installed.\n")


# ---------------------------------------------------------------------------
# ag2 install template
# ---------------------------------------------------------------------------


@app.command("template")
def install_template(
    name: str = typer.Argument(..., help="Template name to install."),
    target: str | None = typer.Option(None, "--target", "-t", help="Target IDE/agent."),
    project_dir: Path = typer.Option(Path("."), "--project-dir", "-d", help="Project directory."),
    var: list[str] | None = typer.Option(None, "--var", help="Template variable (key=value)."),
    preview: bool = typer.Option(False, "--preview", help="Preview without writing."),
) -> None:
    """Install a project template with full AI context.

    [dim]Templates scaffold a working project and install skills that teach
    your AI assistant the architecture and conventions.[/dim]
    """
    project = project_dir.resolve()
    targets = _resolve_targets(target, project)
    stack = _make_installers(project)

    # Parse --var key=value pairs
    variables = {}
    if var:
        for v in var:
            if "=" in v:
                k, _, val = v.partition("=")
                variables[k.strip()] = val.strip()

    console.print(f"\n[heading]Installing template:[/heading] {name}\n")

    try:
        result = stack["templates"].install(name, project, targets, variables=variables, preview=preview)
    except FetchError as e:
        console.print(f"[error]{e}[/error]")
        raise typer.Exit(1)

    _print_result(result, preview=preview)


# ---------------------------------------------------------------------------
# ag2 install tool
# ---------------------------------------------------------------------------


@app.command("tool")
def install_tool(
    name: str = typer.Argument(..., help="Tool name to install."),
    target: str | None = typer.Option(None, "--target", "-t", help="Target IDE/agent."),
    project_dir: Path = typer.Option(Path("."), "--project-dir", "-d", help="Project directory."),
) -> None:
    """Install an AG2 tool function or MCP server.

    [dim]AG2 tools install source code and skills. MCP servers also auto-configure
    your IDE's MCP settings.[/dim]
    """
    project = project_dir.resolve()
    targets = _resolve_targets(target, project)
    stack = _make_installers(project)

    console.print(f"\n[heading]Installing tool:[/heading] {name}\n")

    try:
        result = stack["tools"].install(name, project, targets)
    except FetchError as e:
        console.print(f"[error]{e}[/error]")
        raise typer.Exit(1)

    _print_result(result)

    # Print available functions/tools
    artifact = result.artifact
    if artifact.tool:
        funcs = artifact.tool.functions or artifact.tool.tools_provided
        if funcs:
            console.print("\n  [heading]Available tools:[/heading]")
            for fn in funcs:
                console.print(f"    {fn['name']:30s} {fn.get('description', '')}", style="dim")
            console.print()


# ---------------------------------------------------------------------------
# ag2 install dataset
# ---------------------------------------------------------------------------


@app.command("dataset")
def install_dataset(
    name: str = typer.Argument(..., help="Dataset name to install."),
    target: str | None = typer.Option(None, "--target", "-t", help="Target IDE/agent."),
    project_dir: Path = typer.Option(Path("."), "--project-dir", "-d", help="Project directory."),
    full: bool = typer.Option(False, "--full", help="Download all remote files (may be large)."),
) -> None:
    """Install a dataset for evaluation or training.

    [dim]By default, only inline sample data is installed. Use --full to download
    remote files (which may be large).[/dim]
    """
    project = project_dir.resolve()
    targets = _resolve_targets(target, project)
    stack = _make_installers(project)

    console.print(f"\n[heading]Installing dataset:[/heading] {name}\n")

    try:
        result = stack["datasets"].install(name, project, targets, full=full)
    except FetchError as e:
        console.print(f"[error]{e}[/error]")
        raise typer.Exit(1)

    _print_result(result)


# ---------------------------------------------------------------------------
# ag2 install agent
# ---------------------------------------------------------------------------


@app.command("agent")
def install_agent(
    name: str = typer.Argument(..., help="Agent name to install."),
    target: str | None = typer.Option(None, "--target", "-t", help="Target IDE/agent."),
    project_dir: Path = typer.Option(Path("."), "--project-dir", "-d", help="Project directory."),
) -> None:
    """Install a pre-built Claude Code subagent.

    [dim]Agents are installed to .claude/agents/ and can be invoked via
    natural language, @-mention, or --agent flag.[/dim]
    """
    project = project_dir.resolve()
    targets = _resolve_targets(target, project)
    stack = _make_installers(project)

    console.print(f"\n[heading]Installing agent:[/heading] {name}\n")

    try:
        result = stack["agents"].install(name, project, targets)
    except FetchError as e:
        console.print(f"[error]{e}[/error]")
        raise typer.Exit(1)

    _print_result(result)

    # Print usage instructions
    agent_name = f"ag2-{name}"
    console.print("  [heading]Usage:[/heading]")
    console.print(f'    Natural language:  "Use the {name} agent to ..."')
    console.print(f"    @-mention:         @{agent_name} <task>")
    console.print(f"    Session-wide:      claude --agent {agent_name}")
    console.print()


# ---------------------------------------------------------------------------
# ag2 install bundle
# ---------------------------------------------------------------------------


@app.command("bundle")
def install_bundle(
    name: str = typer.Argument(..., help="Bundle name to install."),
    target: str | None = typer.Option(None, "--target", "-t", help="Target IDE/agent."),
    project_dir: Path = typer.Option(Path("."), "--project-dir", "-d", help="Project directory."),
) -> None:
    """Install a curated collection of artifacts.

    [dim]Bundles combine skills, templates, tools, datasets, and agents into
    a ready-to-use starter kit.[/dim]
    """
    project = project_dir.resolve()
    targets = _resolve_targets(target, project)
    stack = _make_installers(project)

    console.print(f"\n[heading]Installing bundle:[/heading] {name}\n")

    try:
        result = stack["bundles"].install(name, project, targets)
    except FetchError as e:
        console.print(f"[error]{e}[/error]")
        raise typer.Exit(1)

    _print_result(result)


# ---------------------------------------------------------------------------
# ag2 install search
# ---------------------------------------------------------------------------


@app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="Search query."),
    artifact_type: str | None = typer.Option(None, "--type", help="Filter by type."),
) -> None:
    """Search for artifacts across all types."""
    client = ArtifactClient()

    try:
        registry = client.fetch_registry()
    except FetchError as e:
        console.print(f"[error]Could not fetch registry: {e}[/error]")
        raise typer.Exit(1)

    results = client.search(registry, query, artifact_type=artifact_type)

    if not results:
        console.print(f'\n  No results for "{query}"\n')
        raise typer.Exit(0)

    panel_lines = []
    for entry in results:
        type_label = entry.get("type", "?")
        owner = entry.get("owner", "ag2ai")
        name = entry.get("name", "?")
        qualified = f"{owner}/{name}"
        version = entry.get("version", "")
        desc = entry.get("description", "")
        panel_lines.append(f"  [command]{type_label:10s}[/command] {qualified} v{version}")
        if desc:
            panel_lines.append(f"             [dim]{desc}[/dim]")
        panel_lines.append("")

    console.print()
    console.print(
        Panel(
            "\n".join(panel_lines),
            title=f'AG2 Artifacts \u2014 Results for "{query}"',
            border_style="cyan",
        )
    )
    console.print("  Install with: [command]ag2 install <type> <owner/name>[/command]\n")


# ---------------------------------------------------------------------------
# ag2 install list
# ---------------------------------------------------------------------------


@app.command("list")
def list_cmd(
    what: str = typer.Argument(
        "all", help="What to list: all, skills, templates, tools, datasets, agents, bundles, targets, or installed."
    ),
) -> None:
    """List available artifacts, targets, or installed items."""
    if what == "targets":
        _list_targets()
    elif what == "installed":
        _list_installed(Path(".").resolve())
    elif what == "all":
        _list_all_remote()
    else:
        _list_remote(what)


# ---------------------------------------------------------------------------
# ag2 install update
# ---------------------------------------------------------------------------


@app.command("update")
def update_cmd(
    name: str | None = typer.Argument(None, help="Specific artifact to update (default: all)."),
    project_dir: Path = typer.Option(Path("."), "--project-dir", "-d", help="Project directory."),
) -> None:
    """Update installed artifacts to their latest versions."""
    project = project_dir.resolve()
    lockfile = Lockfile(project)
    installed = lockfile.list_installed()

    if not installed:
        console.print("\n  No artifacts installed.\n")
        raise typer.Exit(0)

    if name:
        installed = [i for i in installed if name in i.ref]

    client = ArtifactClient()
    try:
        registry = client.fetch_registry(force_refresh=True)
    except FetchError as e:
        console.print(f"[error]Could not fetch registry: {e}[/error]")
        raise typer.Exit(1)

    # Build registry lookup keyed by the same ref format used in lockfile:
    # "{type_dir}/{owner}/{name}" e.g. "tools/ag2ai/web-search"
    from ..install.artifact import _pluralize_type

    registry_versions = {}
    for e in registry.get("artifacts", []):
        type_dir = _pluralize_type(e.get("type", ""))
        owner = e.get("owner", "ag2ai")
        ref = f"{type_dir}/{owner}/{e['name']}"
        registry_versions[ref] = e.get("version", "0.0.0")

    updates = []
    for info in installed:
        remote_version = registry_versions.get(info.ref)
        if remote_version and remote_version != info.version:
            updates.append((info, remote_version))

    if not updates:
        console.print("\n  All artifacts are up to date.\n")
        raise typer.Exit(0)

    console.print(f"\n[heading]Updates available[/heading] ({len(updates)})\n")
    for info, new_version in updates:
        console.print(f"  {info.ref}: {info.version} \u2192 {new_version}")

    console.print("\n  Re-run the install command for each artifact to update.\n")


# ---------------------------------------------------------------------------
# ag2 install uninstall
# ---------------------------------------------------------------------------


@app.command("uninstall")
def uninstall_cmd(
    name: str = typer.Argument("skills", help="Artifact to uninstall (ref like 'skills/ag2' or shorthand)."),
    target: str | None = typer.Option(None, "--target", "-t", help="Target IDE/agent."),
    project_dir: Path = typer.Option(Path("."), "--project-dir", "-d", help="Project directory."),
) -> None:
    """Remove installed artifacts from your project."""
    project = project_dir.resolve()

    # Try lockfile-based uninstall first
    lockfile = Lockfile(project)
    info = lockfile.get_installed(name)

    if info:
        # Precise uninstall using tracked files
        removed = 0
        for rel_path in info.files:
            f = project / rel_path
            if f.exists():
                f.unlink()
                removed += 1
        lockfile.record_uninstall(name)
        console.print(f"\n[success]Done.[/success] Removed {removed} files for {name}.\n")
        return

    # Fallback: legacy target-based uninstall for skills
    targets = _resolve_targets(target, project)
    console.print("\n[heading]Uninstalling AG2 skills...[/heading]\n")
    total = 0
    for tgt in targets:
        removed_paths = tgt.uninstall(project)
        total += len(removed_paths)
        if removed_paths:
            console.print(f"  [success]\u2713[/success] {tgt.display_name:25s} {len(removed_paths)} files removed")

    if total:
        console.print(f"\n[success]Done.[/success] Removed {total} files from {len(targets)} target(s).\n")
    else:
        console.print("  Nothing to remove.\n")


# ---------------------------------------------------------------------------
# ag2 install from
# ---------------------------------------------------------------------------


@app.command("from")
def install_from(
    source: str = typer.Argument(..., help="Source URL or local path."),
    target: str | None = typer.Option(None, "--target", "-t", help="Target IDE/agent."),
    project_dir: Path = typer.Option(Path("."), "--project-dir", "-d", help="Project directory."),
) -> None:
    """Install an artifact from a URL or local path.

    [dim]Supports GitHub URLs, local directories, or any git-hosted artifact.[/dim]
    """
    project = project_dir.resolve()
    targets = _resolve_targets(target, project)

    # Local path
    source_path = Path(source).resolve()
    if source_path.is_dir():
        from ..install.artifact import load_artifact

        artifact = load_artifact(source_path)
        if artifact is None:
            console.print(f"[error]No artifact.json or manifest.json found in {source}[/error]")
            raise typer.Exit(1)

        artifact.source_dir = source_path
        console.print(f"\n[heading]Installing from local:[/heading] {source_path}\n")

        stack = _make_installers(project)
        _install_local_artifact(artifact, stack, project, targets)
        return

    # GitHub URL — extract repo and path
    console.print("[warning]Remote URL installation coming soon.[/warning]")
    console.print(f"  Source: {source}")
    console.print("  For now, clone the repo locally and use: ag2 install from ./path\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_result(result: InstallResult, preview: bool = False) -> None:
    """Print installation result summary."""
    n = len(result.files_created)

    if preview:
        console.print(f"  [info]Preview:[/info] {n} files would be created:")
        for f in result.files_created[:20]:
            console.print(f"    {f}", style="dim")
        if n > 20:
            console.print(f"    ... and {n - 20} more", style="dim")
        console.print()
        return

    console.print(f"  [success]\u2713[/success] {n} files installed")

    if result.dependencies_installed:
        console.print(f"  [success]\u2713[/success] Dependencies: {', '.join(result.dependencies_installed)}")

    for w in result.warnings:
        console.print(f"  [warning]![/warning] {w}")

    console.print("\n[success]Done.[/success]\n")


def _install_local_artifact(artifact, stack, project, targets) -> None:
    """Install a locally-resolved artifact by delegating to the right installer."""
    from ..install.installers.skills import load_skills_from_artifact

    if artifact.type == "skills":
        items = load_skills_from_artifact(artifact)
        if items:
            all_files = []
            for tgt in targets:
                paths = tgt.install(project, items)
                all_files.extend(paths)
            stack["lockfile"].record_install(
                ref=artifact.ref,
                version=artifact.version,
                targets=[t.name for t in targets],
                files=all_files,
            )
            console.print(f"  [success]\u2713[/success] Installed {len(items)} skills ({len(all_files)} files)")
    elif artifact.type == "template":
        result = _install_local_via_cache(artifact, stack["templates"], "template", project, targets)
        _print_result(result)
    elif artifact.type == "tool":
        result = _install_local_via_cache(artifact, stack["tools"], "tool", project, targets)
        _print_result(result)
    elif artifact.type == "agent":
        result = _install_local_via_cache(artifact, stack["agents"], "agent", project, targets)
        _print_result(result)
        agent_name = f"ag2-{artifact.name}"
        console.print("  [heading]Usage:[/heading]")
        console.print(f'    Natural language:  "Use the {artifact.name} agent to ..."')
        console.print(f"    @-mention:         @{agent_name} <task>")
        console.print(f"    Session-wide:      claude --agent {agent_name}")
    elif artifact.type == "dataset":
        result = _install_local_via_cache(artifact, stack["datasets"], "dataset", project, targets)
        _print_result(result)
    else:
        console.print(f"  [warning]Type '{artifact.type}' not yet supported for local install.[/warning]")

    console.print()


def _install_local_via_cache(artifact, installer, artifact_type, project, targets):
    """Stage a local artifact into the client cache, then install normally."""
    import shutil

    client = installer.client
    type_dir = client._type_dir(artifact_type)
    cache_dest = client.cache_dir / type_dir / artifact.owner / artifact.name / "latest"

    # Copy local artifact to cache
    if cache_dest.exists():
        shutil.rmtree(cache_dest)
    shutil.copytree(artifact.source_dir, cache_dest)
    (cache_dest / ".fetched").touch()

    # Pass qualified name so installer finds the right cache path
    return installer.install(artifact.qualified_name, project, targets)


def _list_targets() -> None:
    """List all supported IDE/agent targets."""
    table = Table(title="Supported Targets", show_edge=False, pad_edge=False)
    table.add_column("Name", style="command")
    table.add_column("IDE / Agent", style="info")
    for t in get_all_targets():
        table.add_row(t.name, t.display_name)
    console.print()
    console.print(table)
    console.print()


def _list_installed(project_dir: Path) -> None:
    """List installed artifacts from lockfile."""
    lockfile = Lockfile(project_dir)
    installed = lockfile.list_installed()

    if not installed:
        console.print("\n  No artifacts installed (no .ag2-artifacts.lock found).\n")
        return

    table = Table(title="Installed Artifacts", show_edge=False, pad_edge=False)
    table.add_column("Artifact", style="command", min_width=25)
    table.add_column("Version", style="info")
    table.add_column("Files", style="count")
    table.add_column("Installed", style="dim")
    for info in installed:
        table.add_row(info.ref, info.version, str(len(info.files)), info.installed_at[:10])
    console.print()
    console.print(table)
    console.print()


def _list_skills_pack() -> None:
    """List bundled skills pack (backward compat)."""
    pack = load_pack("skills")
    if pack is None:
        console.print("[error]Skills pack not found.[/error]")
        raise typer.Exit(1)
    console.print(
        f"\n[heading]{pack.display_name}[/heading] v{pack.version} ([count]{len(pack.items)}[/count] items)\n"
    )
    for category in ["rule", "skill", "agent", "command"]:
        items = [i for i in pack.items if i.category == category]
        if items:
            table = Table(
                title=f"{category.title()}s ({len(items)})",
                show_edge=False,
                pad_edge=False,
                title_style="bold",
            )
            table.add_column("Name", style="command", min_width=30)
            table.add_column("Description", style="dim")
            for item in items:
                table.add_row(item.name, item.description)
            console.print(table)
            console.print()


def _list_all_remote() -> None:
    """List all artifacts from remote registry, grouped by type."""
    client = ArtifactClient()
    try:
        registry = client.fetch_registry()
    except FetchError:
        console.print("[warning]Could not fetch registry.[/warning]")
        return

    entries = registry.get("artifacts", [])
    if not entries:
        console.print("\n  No artifacts found in registry.\n")
        return

    # Group by type
    by_type: dict[str, list[dict]] = {}
    for entry in entries:
        t = entry.get("type", "unknown")
        by_type.setdefault(t, []).append(entry)

    # Display order
    type_order = ["skills", "template", "tool", "dataset", "agent", "bundle"]
    type_labels = {
        "skills": ("Skills", "ag2 install skills"),
        "template": ("Templates", "ag2 install template"),
        "tool": ("Tools", "ag2 install tool"),
        "dataset": ("Datasets", "ag2 install dataset"),
        "agent": ("Agents", "ag2 install agent"),
        "bundle": ("Bundles", "ag2 install bundle"),
    }

    console.print()
    for t in type_order:
        items = by_type.get(t, [])
        if not items:
            continue
        label, cmd = type_labels.get(t, (t.title(), f"ag2 install {t}"))
        table = Table(title=f"{label} ({len(items)})", show_edge=False, pad_edge=False, title_style="bold")
        table.add_column("Artifact", style="command", min_width=30)
        table.add_column("Version", style="info", min_width=8)
        table.add_column("Description", style="dim")
        for entry in items:
            owner = entry.get("owner", "ag2ai")
            qualified = f"{owner}/{entry.get('name', '?')}"
            table.add_row(qualified, entry.get("version", "?"), entry.get("description", ""))
        console.print(table)
        console.print(f"  Install: [command]{cmd} <owner/name>[/command]\n")


def _list_remote(artifact_type: str) -> None:
    """List artifacts from remote registry for a specific type."""
    client = ArtifactClient()
    try:
        registry = client.fetch_registry()
    except FetchError:
        console.print("[warning]Could not fetch registry.[/warning]")
        return

    # Normalize type name (accept both singular and plural)
    type_map = {
        "templates": "template",
        "tools": "tool",
        "datasets": "dataset",
        "agents": "agent",
        "bundles": "bundle",
    }
    normalized = type_map.get(artifact_type, artifact_type)

    entries = client.list_artifacts(registry, artifact_type=normalized)

    if not entries:
        console.print(f"\n  No {artifact_type} found in registry.\n")
        return

    table = Table(title=f"Available {artifact_type.title()}", show_edge=False, pad_edge=False)
    table.add_column("Artifact", style="command", min_width=30)
    table.add_column("Version", style="info")
    table.add_column("Description", style="dim")
    for entry in entries:
        owner = entry.get("owner", "ag2ai")
        qualified = f"{owner}/{entry.get('name', '?')}"
        table.add_row(qualified, entry.get("version", "?"), entry.get("description", ""))
    console.print()
    console.print(table)
    cmd_map = {
        "skills": "skills",
        "template": "template",
        "tool": "tool",
        "dataset": "dataset",
        "agent": "agent",
        "bundle": "bundle",
    }
    cmd = cmd_map.get(normalized, normalized)
    console.print(f"\n  Install: [command]ag2 install {cmd} <owner/name>[/command]\n")


# ---------------------------------------------------------------------------
# Target resolution (carried over from original, unchanged)
# ---------------------------------------------------------------------------


def _resolve_targets(target: str | None, project: Path):
    """Resolve target string to a list of Target objects."""
    if target == "all":
        return get_all_targets()
    elif target:
        targets = []
        for t in target.split(","):
            t = t.strip()
            tgt = get_target(t)
            if tgt is None:
                console.print(f"[error]Unknown target: {t}[/error]")
                console.print("Run [command]ag2 install list targets[/command] to see available targets.")
                raise typer.Exit(1)
            targets.append(tgt)
        return targets
    else:
        detected = detect_targets(project)
        if _is_interactive():
            return _interactive_select(detected)
        if not detected:
            console.print("[warning]No IDE/agent detected in this project.[/warning]")
            console.print(
                "Use [command]--target[/command] to specify one, or [command]--target all[/command] for everything."
            )
            raise typer.Exit(1)
        return detected


def _is_interactive() -> bool:
    try:
        return sys.stdin.isatty()
    except Exception:
        return False


def _interactive_select(detected: list) -> list:
    """Show an interactive multi-select prompt for target selection."""
    all_targets = get_all_targets()
    detected_names = {t.name for t in detected}

    console.print("\n[heading]Select install targets:[/heading]\n")
    for i, t in enumerate(all_targets, 1):
        marker = " [success](detected)[/success]" if t.name in detected_names else ""
        console.print(f"  {i:2d}. {t.display_name}{marker}")
    console.print()

    if detected:
        default_nums = [str(i) for i, t in enumerate(all_targets, 1) if t.name in detected_names]
        default = ",".join(default_nums)
        raw = typer.prompt(
            "Enter numbers (comma-separated), 'all', or press Enter to use detected",
            default=default,
            show_default=False,
        )
    else:
        raw = typer.prompt(
            "Enter numbers (comma-separated), or 'all'",
            default="",
            show_default=False,
        )

    raw = raw.strip()
    if raw.lower() == "all":
        return all_targets

    selected = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            idx = int(part)
        except ValueError:
            tgt = get_target(part)
            if tgt:
                selected.append(tgt)
            else:
                console.print(f"  [warning]Skipping unknown input: {part}[/warning]")
            continue
        if 1 <= idx <= len(all_targets):
            selected.append(all_targets[idx - 1])
        else:
            console.print(f"  [warning]Skipping invalid number: {idx}[/warning]")

    if not selected:
        console.print("[error]No targets selected.[/error]")
        raise typer.Exit(1)

    return selected
