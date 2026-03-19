"""AG2 CLI — main Typer application."""

from __future__ import annotations

import typer

from .commands import arena, create, install, proxy, publish, replay, run, serve, test
from .ui import console, print_banner

app = typer.Typer(
    name="ag2",
    help="AG2 CLI — build, run, test, and deploy multi-agent applications.",
    no_args_is_help=False,
    rich_markup_mode="rich",
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit."),
) -> None:
    """[bold cyan]AG2 CLI[/] — build, run, test, and deploy multi-agent applications."""
    if version:
        from . import __version__

        print_banner(console)
        console.print(f"  ag2-cli [count]{__version__}[/count]")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        print_banner(console)
        console.print("  Run [command]ag2 --help[/command] to see available commands.\n")


# Register command groups
app.add_typer(install.app, name="install")
app.add_typer(create.app, name="create")
app.command("run")(run.run_cmd)
app.command("chat")(run.chat_cmd)
app.command("serve")(serve.serve_cmd)
app.add_typer(test.app, name="test")
app.add_typer(publish.app, name="publish")
app.add_typer(arena.app, name="arena")
app.add_typer(proxy.app, name="proxy")
app.add_typer(replay.app, name="replay")
