"""ag2 install — install skills, templates, and tools into your project."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.table import Table

from ..install import detect_targets, get_all_targets, get_target, list_packs, load_pack
from ..ui import console

app = typer.Typer(
    help="Install skills, templates, and tools into your project.",
    rich_markup_mode="rich",
)


@app.command("skills")
def install_skills(
    name: str | None = typer.Option(None, "--name", "-n", help="Install a specific item by name."),
    target: str | None = typer.Option(None, "--target", "-t", help='Target IDE/agent (comma-separated, or "all").'),
    project_dir: Path = typer.Option(Path("."), "--project-dir", "-d", help="Project directory."),
) -> None:
    """Install AG2 skills (rules, workflows, agents) into your IDE.

    [dim]Auto-detects your IDE when --target is omitted.[/dim]
    """
    project = project_dir.resolve()
    pack = load_pack("skills")
    if pack is None:
        console.print("[error]Skills pack not found.[/error]")
        raise typer.Exit(1)

    items = pack.items
    if name:
        items = [i for i in items if i.name == name]
        if not items:
            console.print(f"[error]No item named '{name}' in skills pack.[/error]")
            console.print("\nAvailable items:")
            for i in pack.items:
                console.print(f"  {i.name}", style="dim")
            raise typer.Exit(1)

    targets = _resolve_targets(target, project)

    console.print(f"\n[heading]Installing {pack.display_name}[/heading] ([count]{len(items)}[/count] items)\n")
    total = 0
    for tgt in targets:
        paths = tgt.install(project, items)
        total += len(paths)
        console.print(f"  [success]✓[/success] {tgt.display_name:25s} {len(paths)} files")

    console.print(f"\n[success]Done.[/success] {total} files installed for {len(targets)} target(s).\n")


@app.command("templates")
def install_templates(
    template_name: str = typer.Argument(..., help="Template name to install."),
    target: str | None = typer.Option(None, "--target", "-t", help='Target IDE/agent (comma-separated, or "all").'),
    project_dir: Path = typer.Option(Path("."), "--project-dir", "-d", help="Project directory."),
    list_available: bool = typer.Option(False, "--list", "-l", help="List available templates."),
    preview: bool = typer.Option(False, "--preview", help="Preview what would be installed without writing."),
) -> None:
    """Install project templates from the AG2 artifacts repository.

    Templates are complete, runnable project scaffolds that give your IDE agent
    full context for building AG2 applications.

    [dim]Fetches from github.com/ag2ai/artifacts/templates/[/dim]
    """
    # TODO: Implement template fetching from ag2ai/artifacts repo
    console.print("[warning]Template installation is coming soon.[/warning]")
    console.print("Track progress at: https://github.com/ag2ai/artifacts")
    raise typer.Exit(0)


@app.command("list")
def list_cmd(
    what: str = typer.Argument("skills", help="What to list: skills, templates, or targets."),
) -> None:
    """List available skills, templates, or supported targets."""
    if what == "targets":
        table = Table(title="Supported Targets", show_edge=False, pad_edge=False)
        table.add_column("Name", style="command")
        table.add_column("IDE / Agent", style="info")
        for t in get_all_targets():
            table.add_row(t.name, t.display_name)
        console.print()
        console.print(table)
        console.print()
    elif what == "skills":
        pack = load_pack("skills")
        if pack is None:
            console.print("[error]Skills pack not found.[/error]")
            raise typer.Exit(1)
        console.print(f"\n[heading]{pack.display_name}[/heading] v{pack.version} ([count]{len(pack.items)}[/count] items)\n")
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
    elif what == "templates":
        console.print("[warning]Template listing is coming soon.[/warning]")
    else:
        console.print(f"[error]Unknown list type: {what}[/error]")
        raise typer.Exit(1)


@app.command("uninstall")
def uninstall(
    pack_type: str = typer.Argument("skills", help="Pack to uninstall."),
    target: str | None = typer.Option(None, "--target", "-t", help='Target IDE/agent (comma-separated, or "all").'),
    project_dir: Path = typer.Option(Path("."), "--project-dir", "-d", help="Project directory."),
) -> None:
    """Remove installed skill files from your project."""
    project = project_dir.resolve()
    targets = _resolve_targets(target, project)

    console.print("\n[heading]Uninstalling AG2 skills...[/heading]\n")
    total = 0
    for tgt in targets:
        removed = tgt.uninstall(project)
        total += len(removed)
        if removed:
            console.print(f"  [success]✓[/success] {tgt.display_name:25s} {len(removed)} files removed")

    if total:
        console.print(f"\n[success]Done.[/success] Removed {total} files from {len(targets)} target(s).\n")
    else:
        console.print("  Nothing to remove.\n")


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
            console.print("Use [command]--target[/command] to specify one, or [command]--target all[/command] for everything.")
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
        default_nums = [
            str(i) for i, t in enumerate(all_targets, 1) if t.name in detected_names
        ]
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
