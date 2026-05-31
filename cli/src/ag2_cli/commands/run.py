"""ag2 run / ag2 chat — run agents and interactive chat sessions."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer
from rich.panel import Panel
from rich.table import Table

from ..core.runner import RunResult, execute
from ..ui import console
from ._shared import extract_cost
from ._shared import require_ag2 as _require_ag2

# ---------------------------------------------------------------------------
# Live rendering callbacks
# ---------------------------------------------------------------------------


def _on_print(text: str) -> None:
    """Forward AG2 print output to the Rich console in real-time."""
    # Skip empty lines from AG2's default formatting
    if text.strip():
        console.print(text, highlight=False)


def _on_event(event: Any) -> None:
    """Render structured AG2 events (from run_group_chat) with Rich."""
    try:
        from autogen.events.agent_events import (
            RunCompletionEvent,
            TextEvent,
            ToolCallEvent,
            ToolResponseEvent,
        )
    except ImportError:
        return

    # Events from run_group_chat are wrapped: data is in event.content
    content = getattr(event, "content", event)

    if isinstance(event, TextEvent):
        sender = getattr(content, "sender", "agent")
        text = getattr(content, "content", "")
        if text and str(text).strip():
            console.print(
                Panel(
                    str(text),
                    title=f"[bold]{sender}[/bold]",
                    border_style="bright_cyan",
                    padding=(0, 1),
                )
            )
    elif isinstance(event, ToolCallEvent):
        tool_calls = getattr(content, "tool_calls", [])
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", "tool") if fn else "tool"
            console.print(f"  [dim]>> tool: {name}[/dim]")
    elif isinstance(event, ToolResponseEvent):
        console.print("  [dim]>> tool response received[/dim]")
    elif isinstance(event, RunCompletionEvent):
        pass  # Handled via RunResult


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _display_header(discovered: Any, title: str = "AG2 Run") -> None:
    """Display a header panel with agent info."""
    from ..core.discovery import DiscoveredAgent

    d: DiscoveredAgent = discovered
    if d.kind == "main":
        subtitle = f"main() from {d.source_file.name}"
    elif d.kind == "agents":
        subtitle = f"{len(d.agents)} agents: {', '.join(d.agent_names)}"
    else:
        subtitle = f"Agent: {', '.join(d.agent_names)}"

    console.print(
        Panel(
            f"[dim]{subtitle}[/dim]",
            title=f"[bold cyan]{title}[/bold cyan]",
            border_style="cyan",
            width=60,
        )
    )
    console.print()


def _display_summary(result: RunResult) -> None:
    """Display a summary footer after execution."""
    table = Table(show_header=False, show_edge=False, box=None, pad_edge=False)
    table.add_column(style="dim", width=12)
    table.add_column()

    table.add_row("Turns:", str(result.turns))
    table.add_row("Time:", f"{result.elapsed:.1f}s")

    if result.cost:
        total = extract_cost(result.cost)
        # Find token counts from first model entry
        tokens = ""
        if isinstance(result.cost, dict):
            usage = result.cost.get("usage_excluding_cached_inference", {})
            for k, v in usage.items():
                if isinstance(v, dict) and "total_tokens" in v:
                    tokens = f" ({v['prompt_tokens']}+{v['completion_tokens']} tokens)"
                    break
        table.add_row("Cost:", f"${total:.6f}{tokens}")

    if result.last_speaker:
        table.add_row("Last agent:", result.last_speaker)

    if result.errors:
        table.add_row("Errors:", f"[error]{len(result.errors)}[/error]")

    console.print()
    console.print(Panel(table, border_style="dim", width=60))


def _discover(agent_file: Path) -> Any:
    """Discover agents from a file, with error handling."""
    path = Path(agent_file).resolve()
    if not path.exists():
        console.print(f"[error]File not found: {path}[/error]")
        raise typer.Exit(1)

    if path.suffix in (".yaml", ".yml"):
        from ..core.discovery import build_agents_from_yaml, load_yaml_config

        config = load_yaml_config(path)
        return build_agents_from_yaml(config)

    from ..core.discovery import discover

    try:
        return discover(path)
    except (ValueError, ImportError) as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def run_cmd(
    agent_file: Path = typer.Argument(..., help="Python file or YAML config defining agents."),
    message: str | None = typer.Option(None, "--message", "-m", help="Input message to send."),
    verbose: bool = typer.Option(False, "--verbose", "-V", help="Show detailed agent activity."),
    output_json: bool = typer.Option(False, "--json", help="Output result as JSON (suppresses live rendering)."),
    max_turns: int = typer.Option(5, "--max-turns", help="Maximum conversation turns."),
) -> None:
    """Run an agent or team from a Python file or YAML config.

    [dim]Examples:[/dim]
      [command]ag2 run my_team.py -m "Research quantum computing"[/command]
      [command]ag2 run team.yaml -m "Hello" --json[/command]
      [command]echo "Hello" | ag2 run my_agent.py[/command]
    """
    _require_ag2()

    # Read from stdin if no message and stdin has data
    if message is None and not sys.stdin.isatty():
        message = sys.stdin.read().strip()

    discovered = _discover(agent_file)

    if not output_json:
        _display_header(discovered)

    if message is None:
        console.print("[error]--message / -m is required.[/error]")
        raise typer.Exit(1)

    # Execute with live rendering (unless JSON mode)
    result = execute(
        discovered,
        message,
        max_turns=max_turns,
        on_print=None if output_json else _on_print,
        on_event=None if output_json else _on_event,
    )

    if output_json:
        data = {
            "output": result.output,
            "turns": result.turns,
            "elapsed": round(result.elapsed, 2),
            "agent_names": result.agent_names,
            "errors": result.errors,
        }
        if result.cost:
            data["cost"] = result.cost
        print(json.dumps(data, indent=2, default=str))
    else:
        # Show output for main() that doesn't emit events
        if result.output and not result.history:
            console.print(result.output)
        _display_summary(result)

    if result.errors:
        if not output_json:
            for err in result.errors:
                console.print(f"[error]{err}[/error]")
        raise typer.Exit(1)


def chat_cmd(
    agent_file: Path | None = typer.Argument(None, help="Python file defining agent(s)."),
    model: str | None = typer.Option(None, "--model", "-M", help="LLM model for ad-hoc chat."),
    system: str | None = typer.Option(None, "--system", "-s", help="System prompt for ad-hoc chat."),
    verbose: bool = typer.Option(False, "--verbose", "-V", help="Show detailed agent activity."),
    max_turns: int = typer.Option(10, "--max-turns", help="Maximum turns per message."),
    no_record: bool = typer.Option(False, "--no-record", help="Disable automatic session recording."),
) -> None:
    """Start an interactive terminal chat with an agent or team.

    Sessions are automatically recorded for later replay with
    [command]ag2 replay[/command]. Use --no-record to disable.

    [dim]Examples:[/dim]
      [command]ag2 chat my_agent.py[/command]
      [command]ag2 chat --model gpt-4o --system "You are a Python expert"[/command]
      [command]ag2 chat my_team.py --verbose[/command]
    """
    ag2 = _require_ag2()

    # Discover or create agent
    if agent_file is not None:
        discovered = _discover(agent_file)
    elif model:
        llm_config = ag2.LLMConfig({"model": model})
        system_msg = system or "You are a helpful assistant."
        agent = ag2.AssistantAgent(name="assistant", system_message=system_msg, llm_config=llm_config)
        from ..core.discovery import DiscoveredAgent

        discovered = DiscoveredAgent(
            kind="agent",
            source_file=Path("<ad-hoc>"),
            agent=agent,
            agent_names=["assistant"],
        )
    else:
        console.print("[error]Provide an agent file or use --model for ad-hoc chat.[/error]")
        console.print("  [command]ag2 chat my_agent.py[/command]")
        console.print('  [command]ag2 chat --model gpt-4o --system "You are a Python expert"[/command]')
        raise typer.Exit(1)

    # Header
    info = (
        f"Team: {', '.join(discovered.agent_names)}"
        if discovered.kind == "agents"
        else f"Agent: {', '.join(discovered.agent_names)}"
    )
    console.print(
        Panel(
            f"[dim]{info}[/dim]\n[dim]Type /quit to exit, /cost for token usage[/dim]",
            title="[bold cyan]AG2 Chat[/bold cyan]",
            border_style="cyan",
            width=60,
        )
    )
    console.print()

    total_cost: Any = None
    turn_count = 0
    chat_history: list[dict[str, Any]] = []

    while True:
        try:
            user_input = console.input("[bold]You:[/bold] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit", "/q"):
            console.print("[dim]Goodbye![/dim]")
            break
        if user_input.lower() == "/cost":
            if total_cost:
                console.print(f"[dim]Total cost: {total_cost}[/dim]")
            else:
                console.print("[dim]No cost data available yet.[/dim]")
            continue
        if user_input.lower() == "/history":
            console.print(f"[dim]Turns so far: {turn_count}[/dim]")
            continue

        turn_count += 1
        try:
            result = execute(
                discovered,
                user_input,
                max_turns=1,
                on_print=_on_print,
                on_event=_on_event,
                clear_history=(turn_count == 1),
            )
            # Surface errors from execute (e.g. LLM API failures)
            if result.errors:
                for err in result.errors:
                    console.print(f"[error]Error: {err}[/error]")
                continue
            # Show output for main() that doesn't emit live events
            if result.output and discovered.kind == "main":
                console.print(
                    Panel(
                        result.output,
                        title="[bold]assistant[/bold]",
                        border_style="bright_cyan",
                        padding=(0, 1),
                    )
                )
            if result.cost:
                total_cost = result.cost
            # Accumulate history for session recording
            chat_history.append({"role": "user", "content": user_input, "name": "user"})
            if result.output:
                speaker = result.last_speaker or "assistant"
                chat_history.append({"role": "assistant", "content": result.output, "name": speaker})
        except Exception as exc:
            console.print(f"[error]Error: {exc}[/error]")
            if verbose:
                console.print_exception()

    # Record the chat session (always-on unless --no-record)
    if not no_record and chat_history:
        from datetime import datetime, timezone

        from .replay import Session, SessionEvent, SessionMeta, create_session_id, save_session

        session_id = create_session_id()
        events = [
            SessionEvent(
                turn=i + 1,
                speaker=msg["name"],
                content=msg["content"],
                role=msg["role"],
            )
            for i, msg in enumerate(chat_history)
        ]
        cost_val = extract_cost(total_cost) if total_cost else 0.0
        meta = SessionMeta(
            session_id=session_id,
            agent_file=str(agent_file or "<ad-hoc>"),
            agent_names=discovered.agent_names,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
            turns=len(events),
            duration=0.0,
            total_cost=cost_val,
            input_message=chat_history[0]["content"] if chat_history else "",
            final_output=chat_history[-1]["content"] if chat_history else "",
        )
        session = Session(meta=meta, events=events)
        save_session(session)
        console.print(f"[dim]Session recorded: {session_id}[/dim]")

    console.print(f"\n[dim]Session ended after {turn_count} turn(s).[/dim]")
    if total_cost:
        console.print(f"[dim]Total cost: {total_cost}[/dim]")
