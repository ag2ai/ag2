"""AG2 Teams Live Dashboard — watch agents work in real-time.

Run with hardcoded agents:
    .venv/bin/pip install rich
    .venv/bin/python teams_docs/dashboard.py

Run with agents from a manifest file:
    .venv/bin/python teams_docs/dashboard.py --manifest teams_docs/sample_agentos_agent_manifests.json

Optionally filter which manifest agents to include:
    .venv/bin/python teams_docs/dashboard.py --manifest manifest.json --agents AtlassianAgent,SlackAgent

Type a goal and watch the live dashboard update as agents plan, work, and complete tasks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import termios
import tty
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

load_dotenv(os.path.expanduser("~/.env"))

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat.teams import Orchestrator, Team, build_workers_from_manifest, load_manifest

# ---------------------------------------------------------------------------
# Dashboard state
# ---------------------------------------------------------------------------


@dataclass
class TaskInfo:
    id: str
    subject: str
    status: str = "pending"  # pending | in_progress | completed
    owner: str = ""
    blocked_by: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0


@dataclass
class AgentInfo:
    name: str
    status: str = "idle"  # idle | working
    current_task: str = ""  # e.g. "#2: Implement CSS"


@dataclass
class DashboardState:
    tasks: dict[str, TaskInfo] = field(default_factory=dict)
    agents: dict[str, AgentInfo] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    phase: str = "init"
    round_number: int = 0
    total_turns: int = 0
    tasks_completed: int = 0
    finished: bool = False
    final_summary: str = ""
    success: bool = False
    # Token usage totals
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost: float = 0.0
    # Streaming output preview
    stream_buffer: str = ""
    stream_agent: str = ""

    MAX_LOG = 15

    def add_log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"{ts}  {msg}")
        if len(self.log) > self.MAX_LOG:
            self.log = self.log[-self.MAX_LOG :]


# ---------------------------------------------------------------------------
# Event → state updates
# ---------------------------------------------------------------------------


def update_state(state: DashboardState, event) -> None:
    """Apply an orchestration event to the dashboard state."""
    etype = event.type
    c = event.content

    if etype == "team_phase":
        state.phase = c.phase
        if c.round_number is not None:
            state.round_number = c.round_number
        label = c.phase.upper()
        if c.round_number:
            label += f" (round {c.round_number})"
        state.add_log(f"[bold cyan]Phase:[/] {label}")

    elif etype == "team_task_created":
        state.tasks[c.task_id] = TaskInfo(
            id=c.task_id,
            subject=c.subject,
            blocked_by=list(c.blocked_by),
        )
        blocked = f" [blocked by {c.blocked_by}]" if c.blocked_by else ""
        state.add_log(f"[green]+[/] task created  #{c.task_id}: {_trunc(c.subject, 40)}{blocked}")

    elif etype == "team_task_assigned":
        task = state.tasks.get(c.task_id)
        if task:
            task.owner = c.agent_name
            task.status = "in_progress"
        agent = state.agents.get(c.agent_name)
        if agent:
            agent.status = "working"
            agent.current_task = f"#{c.task_id}: {c.subject}"
        state.add_log(f"[yellow]→[/] task assigned #{c.task_id} → {c.agent_name}")

    elif etype == "team_task_completed":
        task = state.tasks.get(c.task_id)
        if task:
            task.status = "completed"
        state.tasks_completed = sum(1 for t in state.tasks.values() if t.status == "completed")
        # Mark agent idle if it has no more in_progress tasks
        agent = state.agents.get(c.agent_name)
        if agent:
            has_more = any(t.owner == c.agent_name and t.status == "in_progress" for t in state.tasks.values())
            if not has_more:
                agent.status = "idle"
                agent.current_task = ""
        # Update blocked_by for other tasks
        for t in state.tasks.values():
            if c.task_id in t.blocked_by:
                t.blocked_by.remove(c.task_id)
        unblocked = f" (unblocked: {c.unblocked})" if c.unblocked else ""
        state.add_log(f"[bold green]✓[/] task complete #{c.task_id}: {_trunc(c.subject, 35)}{unblocked}")

    elif etype == "team_agent_step_start":
        state.total_turns += 1
        state.stream_buffer = ""
        state.stream_agent = c.agent_name
        agent = state.agents.get(c.agent_name)
        if agent:
            agent.status = "working"
            if c.task_id:
                agent.current_task = f"#{c.task_id}: {c.task_subject or ''}"

    elif etype == "team_agent_step_complete":
        state.stream_buffer = ""
        state.stream_agent = ""
        tools = ", ".join(c.tools_called) if c.tools_called else "none"
        # Accumulate token usage
        usage = c.usage if isinstance(c.usage, dict) else {}
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        cost = usage.get("cost", 0.0)
        state.total_prompt_tokens += pt
        state.total_completion_tokens += ct
        state.total_cost += cost
        # Attribute usage to the task if this step was for one
        if c.task_id:
            task = state.tasks.get(c.task_id)
            if task:
                task.prompt_tokens += pt
                task.completion_tokens += ct
                task.cost += cost
        tokens_str = f" [{pt + ct:,} tok, ${cost:.4f}]" if pt + ct else ""
        state.add_log(f"[dim]  {c.agent_name} step done (tools: {tools}){tokens_str}[/]")

    elif etype == "team_agent_step_error":
        state.stream_buffer = ""
        state.stream_agent = ""
        # Mark agent idle so it can be retried on the next round
        agent = state.agents.get(c.agent_name)
        if agent:
            agent.status = "idle"
            agent.current_task = ""
        task_str = f" on #{c.task_id}" if c.task_id else ""
        state.add_log(f"[bold red]![/] {c.agent_name} stalled{task_str}: {_trunc(c.error, 50)}")

    elif etype == "team_handoff":
        # Source agent becomes idle, target becomes working
        src = state.agents.get(c.from_agent)
        if src:
            src.status = "idle"
            src.current_task = ""
        dst = state.agents.get(c.to_agent)
        if dst:
            dst.status = "working"
        state.add_log(f"[magenta]⇄[/] handoff {c.from_agent} → {c.to_agent}: {_trunc(c.message, 40)}")

    elif etype == "team_run_complete":
        state.finished = True
        state.success = c.success
        state.final_summary = c.summary
        state.tasks_completed = c.tasks_completed
        status = "[bold green]SUCCESS[/]" if c.success else "[bold red]INCOMPLETE[/]"
        state.add_log(f"{status} — {c.tasks_completed}/{c.tasks_total} tasks, {c.total_turns} turns")


def _trunc(s: str, n: int) -> str:
    s = s.replace("\n", " ")
    return s[:n] + "…" if len(s) > n else s


# ---------------------------------------------------------------------------
# Event → JSON record
# ---------------------------------------------------------------------------

# Fields to extract per event type.  For each type the value is a list of
# attribute names on event.content that we pull into the JSON record.
_EVENT_FIELDS: dict[str, list[str]] = {
    "team_phase": ["phase", "round_number", "detail"],
    "team_task_created": ["task_id", "subject", "description", "blocked_by", "created_by"],
    "team_task_assigned": ["task_id", "subject", "agent_name", "assigned_by"],
    "team_task_completed": ["task_id", "subject", "agent_name", "result", "unblocked"],
    "team_agent_step_start": ["agent_name", "task_id", "task_subject", "message_preview"],
    "team_agent_step_complete": [
        "agent_name",
        "task_id",
        "content",
        "tools_called",
        "tool_call_details",
        "usage",
    ],
    "team_agent_step_error": ["agent_name", "task_id", "task_subject", "error"],
    "team_handoff": ["from_agent", "to_agent", "message"],
    "team_run_complete": ["success", "total_turns", "tasks_completed", "tasks_total", "summary"],
}


def event_to_record(event) -> dict:
    """Convert an orchestration event to a JSON-serializable dict."""
    etype = event.type
    c = event.content
    record: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": etype,
    }
    for attr in _EVENT_FIELDS.get(etype, []):
        record[attr] = getattr(c, attr, None)
    return record


# ---------------------------------------------------------------------------
# Build rich layout
# ---------------------------------------------------------------------------

STATUS_ICON = {
    "pending": "[dim]○[/]",
    "in_progress": "[yellow]◉[/]",
    "completed": "[green]✓[/]",
}

AGENT_ICON = {
    "idle": "[dim]●[/]",
    "working": "[yellow]◉[/]",
}


class DashboardRenderable:
    """Wrapper that re-generates the layout on every Live refresh.

    rich.Live auto-refreshes at ``refresh_per_second`` by re-rendering its
    current renderable.  If that renderable is a static Layout, it never picks
    up changes to ``state`` (e.g. streaming tokens written by the executor
    thread).  By wrapping the state in a ``__rich__`` protocol object, each
    refresh re-calls ``build_layout`` and sees the latest ``stream_buffer``.
    """

    def __init__(self, state: DashboardState, goal: str) -> None:
        self.state = state
        self.goal = goal

    def __rich__(self) -> Layout:
        return build_layout(self.state, self.goal)


def build_layout(state: DashboardState, goal: str) -> Layout:
    """Assemble the full dashboard from current state."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_column(
        Layout(name="tasks", ratio=3),
        Layout(name="agents", ratio=2),
        Layout(name="stream", ratio=2),
        Layout(name="log", ratio=3),
    )

    # --- Header ---
    phase_display = state.phase.upper()
    if state.round_number and state.phase == "work":
        phase_display += f" (round {state.round_number})"
    header_text = Text.from_markup(
        f"  [bold]AG2 Teams Dashboard[/]    Phase: [bold cyan]{phase_display}[/]    Goal: [italic]{_trunc(goal, 50)}[/]"
    )
    layout["header"].update(Panel(header_text, style="bold blue"))

    # --- Tasks table ---
    task_table = Table(expand=True, show_header=True, header_style="bold", box=None, padding=(0, 1))
    task_table.add_column("", width=2)  # status icon
    task_table.add_column("ID", width=4)
    task_table.add_column("Subject", ratio=4)
    task_table.add_column("Owner", width=12)
    task_table.add_column("Status", width=14)
    task_table.add_column("Tokens", width=8, justify="right")
    task_table.add_column("Cost", width=8, justify="right")

    for task in state.tasks.values():
        icon = STATUS_ICON.get(task.status, "?")
        owner = task.owner or "[dim]—[/]"
        status_str = task.status
        if task.blocked_by:
            blocked_ids = ", ".join(f"#{b}" for b in task.blocked_by)
            status_str = f"[red]blocked [{blocked_ids}][/]"
        elif task.status == "in_progress":
            status_str = "[yellow]in_progress[/]"
        elif task.status == "completed":
            status_str = "[green]completed[/]"
        tok_total = task.prompt_tokens + task.completion_tokens
        tok_str = f"{tok_total:,}" if tok_total else "[dim]—[/]"
        cost_str = f"${task.cost:.4f}" if task.cost else "[dim]—[/]"
        task_table.add_row(icon, f"#{task.id}", _trunc(task.subject, 42), owner, status_str, tok_str, cost_str)

    if not state.tasks:
        task_table.add_row("[dim]…[/]", "", "[dim]Waiting for leader to create tasks[/]", "", "", "", "")

    layout["tasks"].update(Panel(task_table, title="[bold]TASKS[/]", border_style="green"))

    # --- Agents ---
    agents_text = Text()
    for agent in state.agents.values():
        icon = AGENT_ICON.get(agent.status, "?")
        line = f"  {icon}  {agent.name:12s}"
        if agent.status == "working" and agent.current_task:
            line += f"  working on {agent.current_task}"
        else:
            line += "  [dim]idle[/]"
        agents_text.append_text(Text.from_markup(line + "\n"))

    if not state.agents:
        agents_text = Text.from_markup("  [dim]No agents registered[/]")

    layout["agents"].update(Panel(agents_text, title="[bold]AGENTS[/]", border_style="cyan"))

    # --- Live output stream ---
    if state.stream_agent and state.stream_buffer:
        # Show last ~500 chars of streaming output to keep it readable
        buf = state.stream_buffer[-500:]
        stream_text = Text.from_markup(f"  [bold]{state.stream_agent}[/] [dim]is generating...[/]\n\n")
        stream_text.append(buf)
    elif state.stream_agent:
        stream_text = Text.from_markup(f"  [dim]{state.stream_agent} is thinking...[/]")
    else:
        stream_text = Text.from_markup("  [dim]Waiting for agent...[/]")

    layout["stream"].update(Panel(stream_text, title="[bold]LIVE OUTPUT[/]", border_style="magenta"))

    # --- Activity log ---
    log_text = Text()
    for line in state.log:
        log_text.append_text(Text.from_markup(line + "\n"))
    if not state.log:
        log_text = Text.from_markup("  [dim]Waiting for events…[/]")

    layout["log"].update(Panel(log_text, title="[bold]ACTIVITY LOG[/]", border_style="yellow"))

    # --- Footer / progress bar + tokens ---
    total = len(state.tasks) or 1
    done = state.tasks_completed
    pct = done / total
    bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
    tok_total = state.total_prompt_tokens + state.total_completion_tokens
    footer_text = Text.from_markup(
        f"  Progress [green]{bar}[/] {done}/{total} tasks    "
        f"Turns: {state.total_turns}    "
        f"Round: {state.round_number}    "
        f"Tokens: {tok_total:,}    "
        f"Cost: [bold]${state.total_cost:.4f}[/]"
    )
    layout["footer"].update(Panel(footer_text, style="dim"))

    return layout


# ---------------------------------------------------------------------------
# Team setup (reused from demo.py)
# ---------------------------------------------------------------------------


def _make_llm_config(model: str, max_tokens: int = 4096) -> LLMConfig:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY in ~/.env")
    return LLMConfig({
        "model": model,
        "api_key": api_key,
        "api_type": "anthropic",
        "temperature": 0.0,
        "max_tokens": max_tokens,
    })


def build_team() -> Team:
    team = Team("dashboard-demo", description="Live dashboard demo team")

    leader = ConversableAgent(
        name="leader",
        llm_config=_make_llm_config("claude-sonnet-4-5"),
        system_message=(
            "You are a project leader. Break goals into small, concrete tasks "
            "using create_task. Assign each task to the best team member with "
            "assign_task. Use blocked_by when tasks depend on earlier ones. "
            "Keep tasks focused — one clear deliverable each."
        ),
    )
    team.add_agent(leader, is_leader=True)

    writer = ConversableAgent(
        name="writer",
        llm_config=_make_llm_config("claude-haiku-4-5", max_tokens=8192),
        system_message=(
            "You are a skilled writer and researcher. When given a task, produce "
            "high-quality written content: articles, blog posts, documentation, "
            "copy, or research summaries. Use complete_task with your writing "
            "as the result."
        ),
        description="Writer & researcher — articles, blog posts, docs, research",
    )
    team.add_agent(writer)

    developer = ConversableAgent(
        name="developer",
        llm_config=_make_llm_config("claude-haiku-4-5", max_tokens=8192),
        system_message=(
            "You are a developer. When given a task, write code, technical specs, "
            "or structured output (HTML, CSS, JavaScript, Markdown, JSON, etc.). "
            "Use complete_task with your output as the result."
        ),
        description="Developer — code, technical specs, structured output",
    )
    team.add_agent(developer)

    return team


def build_team_from_manifest_file(
    manifest_path: str,
    agent_names: list[str] | None = None,
) -> Team:
    """Build a team from a manifest file.

    Creates a local leader agent and wraps each manifest agent in a local
    ConversableAgent that delegates work via ask_specialist() -> remote A2A.

    Args:
        manifest_path: Path to the agent manifest JSON file.
        agent_names: Optional list of agent names to include. If None, all agents.

    Returns:
        A Team ready to use with an Orchestrator.
    """
    manifest = load_manifest(manifest_path)

    if agent_names:
        available = (
            {a.name for a in manifest.a2a}
            | {a.name for a in manifest.agents}
            | {a.name for a in manifest.local_agents}
            | {a.name for a in manifest.reference_agents}
        )
        missing = set(agent_names) - available
        if missing:
            raise ValueError(f"Agents not found in manifest: {missing}. Available: {sorted(available)}")

    team = Team("manifest-team", description="Team built from agent manifest")

    # Leader is always local — needs LLM for planning and task management
    leader = ConversableAgent(
        name="leader",
        llm_config=_make_llm_config("claude-sonnet-4-5"),
        system_message=(
            "You are a project leader. Break goals into small, concrete tasks "
            "using create_task. Assign each task to the best team member with "
            "assign_task. Use blocked_by when tasks depend on earlier ones. "
            "Keep tasks focused — one clear deliverable each."
        ),
    )
    team.add_agent(leader, is_leader=True)

    # Workers from manifest — each gets a cheap local LLM (Haiku) for tool
    # orchestration + an ask_specialist tool for remote A2A delegation
    worker_config = _make_llm_config("claude-haiku-4-5", max_tokens=8192)
    workers = build_workers_from_manifest(
        manifest,
        worker_config,
        agent_names=agent_names,
    )

    for worker in workers:
        team.add_agent(worker)

    return team


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_dashboard(goal: str, team: Team, **orch_kwargs) -> None:
    """Run the orchestrator with a live-updating rich dashboard."""
    state = DashboardState()

    # Seed agent info from team
    for name in team.agents:
        if name != team._leader_name:
            state.agents[name] = AgentInfo(name=name)

    orch = Orchestrator(team, **orch_kwargs)
    console = Console()

    # Streaming callback — called from executor thread, appends to state buffer.
    # The Live refresh (4fps) picks up changes automatically.
    def on_token(text: str) -> None:
        state.stream_buffer += text

    # Prepare JSONL log file
    runs_dir = Path(__file__).parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = runs_dir / f"run_{ts_str}.jsonl"

    # Put terminal into cbreak mode to swallow arrow keys / scroll escape sequences
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    # Use a dynamic renderable so Live auto-refresh (4fps) picks up streaming
    # tokens written to state.stream_buffer by the executor thread.
    renderable = DashboardRenderable(state, goal)

    try:
        tty.setcbreak(fd)
        with (
            open(log_path, "w") as log_file,  # noqa: ASYNC230
            Live(renderable, console=console, refresh_per_second=4, screen=True) as live,
        ):
            # Write a header record with run metadata
            header = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": "run_start",
                "goal": goal,
                "team": team.name,
                "agents": list(team.agents.keys()),
            }
            log_file.write(json.dumps(header) + "\n")

            async for event in orch.run_stream(goal, stream=True, on_token=on_token):
                # Write to JSONL log
                log_file.write(json.dumps(event_to_record(event)) + "\n")
                log_file.flush()

                # Update state — the Live auto-refresh will pick up changes
                update_state(state, event)
                live.refresh()

            # Keep the final state visible for a moment
            await asyncio.sleep(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    # Print final summary below the dashboard
    console.print()
    if state.success:
        console.print("[bold green]Run completed successfully![/]")
    else:
        console.print("[bold red]Run finished (not all tasks completed).[/]")
    console.print()
    console.print(Panel(state.final_summary or "No summary.", title="Final Summary", border_style="blue"))

    # Token usage + cost breakdown
    console.print()
    cost_table = Table(title="Token Usage & Cost", show_header=True, header_style="bold")
    cost_table.add_column("", width=20)
    cost_table.add_column("Prompt", justify="right")
    cost_table.add_column("Completion", justify="right")
    cost_table.add_column("Total", justify="right")
    cost_table.add_column("Cost", justify="right")

    for task in state.tasks.values():
        tok = task.prompt_tokens + task.completion_tokens
        if tok:
            cost_table.add_row(
                f"Task #{task.id}",
                f"{task.prompt_tokens:,}",
                f"{task.completion_tokens:,}",
                f"{tok:,}",
                f"${task.cost:.4f}",
            )

    # Overhead = leader planning/summarization (not attributed to tasks)
    task_prompt = sum(t.prompt_tokens for t in state.tasks.values())
    task_completion = sum(t.completion_tokens for t in state.tasks.values())
    task_cost = sum(t.cost for t in state.tasks.values())
    overhead_prompt = state.total_prompt_tokens - task_prompt
    overhead_completion = state.total_completion_tokens - task_completion
    overhead_cost = state.total_cost - task_cost
    if overhead_prompt + overhead_completion:
        cost_table.add_row(
            "[dim]Leader[/]",
            f"[dim]{overhead_prompt:,}[/]",
            f"[dim]{overhead_completion:,}[/]",
            f"[dim]{overhead_prompt + overhead_completion:,}[/]",
            f"[dim]${overhead_cost:.4f}[/]",
        )

    cost_table.add_section()
    tok_total = state.total_prompt_tokens + state.total_completion_tokens
    cost_table.add_row(
        "[bold]TOTAL[/]",
        f"[bold]{state.total_prompt_tokens:,}[/]",
        f"[bold]{state.total_completion_tokens:,}[/]",
        f"[bold]{tok_total:,}[/]",
        f"[bold]${state.total_cost:.4f}[/]",
    )
    console.print(cost_table)

    console.print()
    console.print(f"[dim]Run log saved to:[/] {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AG2 Teams Live Dashboard")
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to an agent manifest JSON file. If provided, builds team from manifest.",
    )
    parser.add_argument(
        "--agents",
        type=str,
        default=None,
        help="Comma-separated list of agent names to include from the manifest.",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Goal to accomplish. If not provided, prompts interactively.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=6,
        help="Maximum orchestration rounds (default: 6).",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    console = Console()
    console.print()
    console.print("[bold blue]AG2 Teams Live Dashboard[/]")
    console.print("[dim]Watch agents plan and work in real-time.[/]")
    console.print()

    # Build team from manifest or hardcoded agents
    if args.manifest:
        agent_names = [n.strip() for n in args.agents.split(",")] if args.agents else None
        team = build_team_from_manifest_file(args.manifest, agent_names=agent_names)
        agent_list = [n for n in team.agents if n != team._leader_name]
        console.print(f"[dim]Loaded {len(agent_list)} agents from manifest: {', '.join(agent_list)}[/]")
    else:
        team = build_team()

    console.print()

    # Get goal
    if args.goal:
        goal = args.goal
    else:
        goal = console.input("[bold]Enter your goal:[/] ")
        if not goal.strip():
            goal = (
                "Create a simple maths quiz web page with 5 addition questions, "
                "a score counter, and a 'check answers' button"
            )
            console.print(f"[dim]Using default goal: {goal}[/]")

    await run_dashboard(goal, team, max_rounds=args.max_rounds, max_stalls=2)


if __name__ == "__main__":
    asyncio.run(main())
