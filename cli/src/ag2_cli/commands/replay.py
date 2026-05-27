"""ag2 replay — record, replay, and debug agent conversations.

Provides session recording, step-through debugging, conversation
branching, side-by-side comparison, and export capabilities.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table

from ..ui import console

app = typer.Typer(
    help="Record, replay, and debug agent conversations.",
    rich_markup_mode="rich",
)

SESSIONS_DIR = Path.home() / ".ag2" / "sessions"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SessionEvent:
    """A single event in a recorded session."""

    turn: int
    speaker: str
    content: str
    role: str  # "user", "assistant", "system", "tool"
    timestamp: float = 0.0
    elapsed: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionMeta:
    """Metadata for a recorded session."""

    session_id: str
    agent_file: str
    agent_names: list[str]
    created_at: str
    turns: int
    duration: float
    total_cost: float = 0.0
    input_message: str = ""
    final_output: str = ""


@dataclass
class Session:
    """A complete recorded session."""

    meta: SessionMeta
    events: list[SessionEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def save_session(session: Session) -> Path:
    """Save a session to disk. Returns the file path."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = _session_path(session.meta.session_id)
    data = {
        "meta": asdict(session.meta),
        "events": [asdict(e) for e in session.events],
    }
    path.write_text(json.dumps(data, indent=2, default=str))
    return path


def load_session(session_id: str) -> Session:
    """Load a session from disk."""
    path = _session_path(session_id)
    if not path.exists():
        # Try prefix match
        candidates = list(SESSIONS_DIR.glob(f"{session_id}*.json"))
        if len(candidates) == 1:
            path = candidates[0]
        elif len(candidates) > 1:
            console.print(f"[error]Ambiguous session ID. Matches: {len(candidates)}[/error]")
            for c in candidates:
                console.print(f"  [dim]{c.stem}[/dim]")
            raise typer.Exit(1)
        else:
            console.print(f"[error]Session not found: {session_id}[/error]")
            raise typer.Exit(1)

    data = json.loads(path.read_text())
    meta = SessionMeta(**data["meta"])
    events = [SessionEvent(**e) for e in data["events"]]
    return Session(meta=meta, events=events)


def list_sessions() -> list[SessionMeta]:
    """List all recorded sessions, newest first."""
    if not SESSIONS_DIR.exists():
        return []

    sessions: list[SessionMeta] = []
    for path in sorted(SESSIONS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text())
            sessions.append(SessionMeta(**data["meta"]))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return sessions


def delete_session(session_id: str) -> bool:
    """Delete a session. Returns True if deleted."""
    path = _session_path(session_id)
    if path.exists():
        path.unlink()
        return True
    # Try prefix match
    candidates = list(SESSIONS_DIR.glob(f"{session_id}*.json"))
    if len(candidates) == 1:
        candidates[0].unlink()
        return True
    return False


# ---------------------------------------------------------------------------
# Recording helpers (used by run/chat commands)
# ---------------------------------------------------------------------------


def create_session_id() -> str:
    """Generate a short, human-friendly session ID."""
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def record_from_run_result(
    result: Any,
    agent_file: str,
    message: str,
) -> Session:
    """Create a Session from a RunResult (from core.runner)."""
    session_id = create_session_id()

    events: list[SessionEvent] = []
    history = getattr(result, "history", []) or []

    for i, msg in enumerate(history):
        events.append(
            SessionEvent(
                turn=i + 1,
                speaker=msg.get("name", msg.get("role", "unknown")),
                content=msg.get("content", ""),
                role=msg.get("role", "assistant"),
                metadata={k: v for k, v in msg.items() if k not in ("content", "role", "name")},
            )
        )

    cost = 0.0
    if hasattr(result, "cost") and result.cost and isinstance(result.cost, dict):
        usage = result.cost.get("usage_excluding_cached_inference", {})
        cost = usage.get("total_cost", 0)

    meta = SessionMeta(
        session_id=session_id,
        agent_file=agent_file,
        agent_names=getattr(result, "agent_names", []),
        created_at=datetime.now(tz=timezone.utc).isoformat(),
        turns=len(events),
        duration=getattr(result, "elapsed", 0.0),
        total_cost=cost,
        input_message=message,
        final_output=getattr(result, "output", ""),
    )

    return Session(meta=meta, events=events)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _render_event(event: SessionEvent, width: int = 60) -> Panel:
    """Render a single event as a Rich panel."""
    style = "blue" if event.role == "user" else "green"
    if event.role == "tool":
        style = "yellow"
    elif event.role == "system":
        style = "dim"

    subtitle_parts = []
    if event.elapsed > 0:
        subtitle_parts.append(f"{event.elapsed:.1f}s")
    if event.metadata.get("tool_calls"):
        subtitle_parts.append("tool call")

    subtitle = f"[dim]({', '.join(subtitle_parts)})[/dim]" if subtitle_parts else ""

    return Panel(
        event.content[:2000] if event.content else "[dim]<empty>[/dim]",
        title=f"[bold]Turn {event.turn}: {event.speaker}[/bold]",
        subtitle=subtitle,
        border_style=style,
        width=width,
    )


def _render_session_header(meta: SessionMeta) -> Panel:
    """Render session metadata as a header panel."""
    info_lines = [
        f"[dim]Session:[/dim]  {meta.session_id}",
        f"[dim]Agent:[/dim]    {meta.agent_file}",
        f"[dim]Agents:[/dim]   {', '.join(meta.agent_names) if meta.agent_names else 'N/A'}",
        f"[dim]Turns:[/dim]    {meta.turns}",
        f"[dim]Duration:[/dim] {meta.duration:.1f}s",
    ]
    if meta.total_cost > 0:
        info_lines.append(f"[dim]Cost:[/dim]     ${meta.total_cost:.6f}")
    if meta.input_message:
        msg_preview = meta.input_message[:80]
        if len(meta.input_message) > 80:
            msg_preview += "..."
        info_lines.append(f"[dim]Input:[/dim]    {msg_preview}")

    return Panel(
        "\n".join(info_lines),
        title="[bold cyan]AG2 Replay[/bold cyan]",
        border_style="cyan",
        width=64,
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("list")
def replay_list(
    limit: int = typer.Option(20, "--limit", "-n", help="Max sessions to show."),
) -> None:
    """List recorded sessions.

    [dim]Examples:[/dim]
      [command]ag2 replay list[/command]
      [command]ag2 replay list --limit 5[/command]
    """
    sessions = list_sessions()

    if not sessions:
        console.print("[dim]No recorded sessions yet.[/dim]")
        console.print(
            "[dim]Use [command]ag2 chat[/command] to start a recorded session (recording is automatic).[/dim]"
        )
        raise typer.Exit(0)

    table = Table(title="Recorded Sessions", width=80)
    table.add_column("Session ID", style="bold", min_width=22)
    table.add_column("Agent", min_width=15)
    table.add_column("Turns", justify="right", width=6)
    table.add_column("Duration", justify="right", width=8)
    table.add_column("Date", style="dim", width=16)

    for meta in sessions[:limit]:
        date_str = meta.created_at[:16].replace("T", " ") if meta.created_at else ""
        table.add_row(
            meta.session_id,
            Path(meta.agent_file).stem if meta.agent_file else "N/A",
            str(meta.turns),
            f"{meta.duration:.1f}s",
            date_str,
        )

    console.print()
    console.print(table)
    if len(sessions) > limit:
        console.print(f"[dim]  ... and {len(sessions) - limit} more[/dim]")
    console.print()


@app.command("show")
def replay_show(
    session_id: str = typer.Argument(..., help="Session ID (or prefix)."),
) -> None:
    """Replay a recorded session.

    [dim]Examples:[/dim]
      [command]ag2 replay show 20260319-143022-a1b2c3[/command]
      [command]ag2 replay show 20260319[/command]
    """
    session = load_session(session_id)
    console.print()
    console.print(_render_session_header(session.meta))
    console.print()

    for event in session.events:
        console.print(_render_event(event))
        console.print()


@app.command("step")
def replay_step(
    session_id: str = typer.Argument(..., help="Session ID (or prefix)."),
) -> None:
    """Step through a recorded session interactively.

    Controls: [Enter] next | [p] previous | [g N] go to turn N | [q] quit

    [dim]Examples:[/dim]
      [command]ag2 replay step 20260319-143022-a1b2c3[/command]
    """
    session = load_session(session_id)
    events = session.events

    if not events:
        console.print("[dim]Session has no events.[/dim]")
        raise typer.Exit(0)

    console.print()
    console.print(_render_session_header(session.meta))
    console.print()

    idx = 0
    while 0 <= idx < len(events):
        console.print(f"[dim]Turn {idx + 1}/{len(events)}[/dim]")
        console.print(_render_event(events[idx]))
        console.print()

        try:
            cmd = console.input("[dim][Enter] next | [p] prev | [g N] goto | [q] quit > [/dim]").strip().lower()
        except (KeyboardInterrupt, EOFError):
            break

        if cmd == "" or cmd == "n":
            idx += 1
        elif cmd == "p":
            idx = max(0, idx - 1)
        elif cmd.startswith("g ") or cmd == "g":
            parts = cmd.split()
            if len(parts) < 2:
                console.print("[warning]Usage: g <turn_number>[/warning]")
            else:
                try:
                    target = int(parts[1])
                    idx = max(0, min(target - 1, len(events) - 1))
                except ValueError:
                    console.print("[warning]Usage: g <turn_number>[/warning]")
        elif cmd in ("q", "quit"):
            break
        else:
            idx += 1

    console.print("[dim]End of replay.[/dim]")


@app.command("branch")
def replay_branch(
    session_id: str = typer.Argument(..., help="Session ID to branch from."),
    at: int = typer.Option(..., "--at", help="Turn number to branch from."),
    message: str = typer.Option(None, "--message", "-m", help="New message for the branch (default: re-run same)."),
) -> None:
    """Branch a session at a specific turn and re-run with a new message.

    [dim]Examples:[/dim]
      [command]ag2 replay branch 20260319-a1b2c3 --at 3 -m "Try a different approach"[/command]
    """
    try:
        import autogen  # noqa: F401
    except ImportError:
        console.print("[error]ag2 is not installed.[/error]")
        raise typer.Exit(1)

    session = load_session(session_id)

    if at < 1 or at > len(session.events):
        console.print(f"[error]Turn {at} is out of range (session has {len(session.events)} turns).[/error]")
        raise typer.Exit(1)

    # Get the events up to the branch point
    prefix_events = session.events[:at]
    branch_msg = message or (prefix_events[-1].content if prefix_events else session.meta.input_message)

    console.print(
        Panel(
            f"[dim]Branching from session {session.meta.session_id}\nAt turn {at}, with message:[/dim]\n\n{branch_msg}",
            title="[bold cyan]AG2 Replay \u2014 Branch[/bold cyan]",
            border_style="cyan",
            width=64,
        )
    )

    # Re-run: replay prior turns to rebuild context, then send new message
    from ..core.discovery import discover
    from ..core.runner import execute

    agent_path = Path(session.meta.agent_file)
    if not agent_path.exists():
        console.print(f"[error]Agent file not found: {agent_path}[/error]")
        console.print("[dim]The original agent file is needed to re-run.[/dim]")
        raise typer.Exit(1)

    discovered = discover(agent_path)

    # Replay prior turns to rebuild agent memory
    prior_user_msgs = [e.content for e in prefix_events if e.role == "user" and e.content]

    if discovered.kind == "agent" and prior_user_msgs:
        # Replay each prior user message so the agent accumulates context
        for i, prior_msg in enumerate(prior_user_msgs):
            console.print(f"  [dim]Replaying turn {i + 1}...[/dim]")
            execute(
                discovered,
                prior_msg,
                clear_history=(i == 0),
            )

        # Now send the branch message with full context
        result = execute(
            discovered,
            branch_msg,
            clear_history=False,
        )
    else:
        # For main() or agents list, just run with the branch message
        result = execute(discovered, branch_msg)

    # Record the branched session (include prior turns + new response)
    branch_session_id = f"branch-{create_session_id()}"
    branch_events = list(prefix_events)  # copy prior events
    branch_events.append(
        SessionEvent(
            turn=len(branch_events) + 1,
            speaker="user",
            content=branch_msg,
            role="user",
        )
    )
    if result.output:
        branch_events.append(
            SessionEvent(
                turn=len(branch_events) + 1,
                speaker=result.last_speaker or "assistant",
                content=result.output,
                role="assistant",
                elapsed=result.elapsed,
            )
        )

    cost_val = 0.0
    if result.cost and isinstance(result.cost, dict):
        cost_val = result.cost.get("usage_excluding_cached_inference", {}).get("total_cost", 0)

    branch_meta = SessionMeta(
        session_id=branch_session_id,
        agent_file=str(agent_path),
        agent_names=session.meta.agent_names,
        created_at=datetime.now(tz=timezone.utc).isoformat(),
        turns=len(branch_events),
        duration=result.elapsed,
        total_cost=cost_val,
        input_message=session.meta.input_message,
        final_output=result.output or "",
    )
    new_session = Session(meta=branch_meta, events=branch_events)
    save_session(new_session)

    console.print(f"\n[success]Branched session saved: {branch_session_id}[/success]")
    console.print(f"[dim]Output: {result.output[:200]}[/dim]")


@app.command("compare")
def replay_compare(
    session_a: str = typer.Argument(..., help="First session ID."),
    session_b: str = typer.Argument(..., help="Second session ID."),
) -> None:
    """Compare two sessions side-by-side.

    [dim]Examples:[/dim]
      [command]ag2 replay compare 20260319-a1b2c3 20260319-d4e5f6[/command]
    """
    sa = load_session(session_a)
    sb = load_session(session_b)

    console.print()

    # Header comparison
    table = Table(title="Session Comparison", width=80)
    table.add_column("", style="dim", width=12)
    table.add_column(sa.meta.session_id[:20], min_width=30)
    table.add_column(sb.meta.session_id[:20], min_width=30)

    table.add_row("Agent", sa.meta.agent_file, sb.meta.agent_file)
    table.add_row("Turns", str(sa.meta.turns), str(sb.meta.turns))
    table.add_row(
        "Duration",
        f"{sa.meta.duration:.1f}s",
        f"{sb.meta.duration:.1f}s",
    )
    table.add_row(
        "Cost",
        f"${sa.meta.total_cost:.6f}",
        f"${sb.meta.total_cost:.6f}",
    )

    console.print(table)
    console.print()

    # Side-by-side events
    max_turns = max(len(sa.events), len(sb.events))
    for i in range(max_turns):
        panels = []
        if i < len(sa.events):
            panels.append(_render_event(sa.events[i], width=38))
        else:
            panels.append(Panel("[dim]—[/dim]", width=38, border_style="dim"))

        if i < len(sb.events):
            panels.append(_render_event(sb.events[i], width=38))
        else:
            panels.append(Panel("[dim]—[/dim]", width=38, border_style="dim"))

        console.print(Columns(panels, padding=(0, 1)))
        console.print()


@app.command("export")
def replay_export(
    session_id: str = typer.Argument(..., help="Session ID to export."),
    format: str = typer.Option("md", "--format", "-f", help="Export format: md, json, html."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output file (default: stdout)."),
) -> None:
    """Export a session transcript.

    [dim]Examples:[/dim]
      [command]ag2 replay export 20260319-a1b2c3 --format md[/command]
      [command]ag2 replay export 20260319-a1b2c3 --format json -o session.json[/command]
    """
    session = load_session(session_id)

    if format == "json":
        data = {
            "meta": asdict(session.meta),
            "events": [asdict(e) for e in session.events],
        }
        content = json.dumps(data, indent=2, default=str)

    elif format in ("md", "markdown"):
        lines = [
            f"# Session: {session.meta.session_id}",
            "",
            f"- **Agent:** {session.meta.agent_file}",
            f"- **Agents:** {', '.join(session.meta.agent_names)}",
            f"- **Date:** {session.meta.created_at}",
            f"- **Turns:** {session.meta.turns}",
            f"- **Duration:** {session.meta.duration:.1f}s",
            f"- **Cost:** ${session.meta.total_cost:.6f}",
            "",
            "---",
            "",
        ]
        for event in session.events:
            lines.append(f"## Turn {event.turn}: {event.speaker}")
            lines.append("")
            lines.append(event.content)
            lines.append("")
        content = "\n".join(lines)

    elif format == "html":
        from html import escape as html_escape

        html_events = []
        for event in session.events:
            color = "#4a9eff" if event.role == "user" else "#50c878"
            speaker = html_escape(event.speaker)
            body = html_escape(event.content) if event.content else ""
            html_events.append(
                f'<div style="margin:10px 0;padding:10px;border-left:3px solid {color};">'
                f"<strong>Turn {event.turn}: {speaker}</strong><br>"
                f"<pre>{body}</pre></div>"
            )
        sid = html_escape(session.meta.session_id)
        agent_file = html_escape(session.meta.agent_file)
        content = (
            f"<html><head><title>Session {sid}</title>"
            "<style>body{font-family:monospace;max-width:800px;margin:0 auto;}</style></head>"
            f"<body><h1>Session: {sid}</h1>"
            f"<p>Agent: {agent_file} | "
            f"Turns: {session.meta.turns} | "
            f"Duration: {session.meta.duration:.1f}s</p>"
            f"{''.join(html_events)}</body></html>"
        )
    else:
        console.print(f"[error]Unknown format: {format}. Use md, json, or html.[/error]")
        raise typer.Exit(1)

    if output:
        output.write_text(content)
        console.print(f"[success]Exported to {output}[/success]")
    else:
        print(content)


@app.command("delete")
def replay_delete(
    session_id: str = typer.Argument(..., help="Session ID to delete."),
) -> None:
    """Delete a recorded session.

    [dim]Examples:[/dim]
      [command]ag2 replay delete 20260319-a1b2c3[/command]
    """
    if delete_session(session_id):
        console.print(f"[success]Deleted session: {session_id}[/success]")
    else:
        console.print(f"[error]Session not found: {session_id}[/error]")
        raise typer.Exit(1)


@app.command("clear")
def replay_clear() -> None:
    """Delete all recorded sessions.

    [dim]Examples:[/dim]
      [command]ag2 replay clear[/command]
    """
    if not SESSIONS_DIR.exists():
        console.print("[dim]No sessions to clear.[/dim]")
        raise typer.Exit(0)

    sessions = list(SESSIONS_DIR.glob("*.json"))
    for s in sessions:
        s.unlink()

    console.print(f"[success]Cleared {len(sessions)} session(s).[/success]")
