"""ag2 arena — A/B test agent implementations.

Compare quality, cost, and speed across agent implementations, models,
or in interactive head-to-head mode with ELO ratings.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer
from rich.panel import Panel
from rich.table import Table

from ..core.runner import execute
from ..testing import CaseResult, EvalCase, EvalSuite, check_assertion, load_eval_suite
from ..ui import console
from ._shared import extract_cost

app = typer.Typer(
    help="A/B test agent implementations — compare quality, cost, and speed.",
    rich_markup_mode="rich",
)

LEADERBOARD_PATH = Path.home() / ".ag2" / "arena" / "leaderboard.json"


@dataclass
class ContenderResult:
    """Aggregated results for one contender across all eval cases."""

    name: str
    source: str
    case_results: list[CaseResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.case_results:
            return 0.0
        return sum(1 for r in self.case_results if r.passed) / len(self.case_results)

    @property
    def avg_score(self) -> float:
        if not self.case_results:
            return 0.0
        return sum(r.score for r in self.case_results) / len(self.case_results)

    @property
    def avg_elapsed(self) -> float:
        if not self.case_results:
            return 0.0
        return sum(r.elapsed for r in self.case_results) / len(self.case_results)

    @property
    def total_elapsed(self) -> float:
        return sum(r.elapsed for r in self.case_results)

    @property
    def total_cost(self) -> float:
        return sum(extract_cost(r.cost) for r in self.case_results if r.cost)


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------


def _run_single_case(agent_file: Path, case: EvalCase) -> CaseResult:
    """Run a single eval case against a fresh agent instance."""
    from ..core.discovery import discover

    start = time.time()

    try:
        discovered = discover(agent_file)
    except (ValueError, ImportError) as exc:
        return CaseResult(case=case, errors=[str(exc)], elapsed=time.time() - start)

    result = execute(discovered, case.input)

    assertion_results = [
        check_assertion(a, result.output, turns=result.turns, errors=result.errors) for a in case.assertions
    ]

    return CaseResult(
        case=case,
        assertion_results=assertion_results,
        output=result.output,
        turns=result.turns,
        errors=result.errors,
        elapsed=result.elapsed,
        cost=result.cost,
    )


def _run_contender(agent_file: Path, suites: list[EvalSuite], *, name: str | None = None) -> ContenderResult:
    """Run all eval cases for one contender."""
    contender_name = name or agent_file.stem
    results: list[CaseResult] = []
    for suite in suites:
        for case in suite.cases:
            results.append(_run_single_case(agent_file, case))
    return ContenderResult(name=contender_name, source=str(agent_file), case_results=results)


def _load_suites(eval_path: Path) -> list[EvalSuite]:
    """Load eval suites from a file or directory."""
    eval_path = eval_path.resolve()
    if eval_path.is_dir():
        yaml_files = sorted(eval_path.glob("*.yaml")) + sorted(eval_path.glob("*.yml"))
        if not yaml_files:
            console.print(f"[error]No YAML files found in {eval_path}[/error]")
            raise typer.Exit(1)
        return [load_eval_suite(f) for f in yaml_files]
    return [load_eval_suite(eval_path)]


def _resolve_contender_files(contenders: list[Path]) -> list[Path]:
    """Resolve contender paths — expand directories to .py files."""
    files: list[Path] = []
    for p in contenders:
        p = p.resolve()
        if p.is_dir():
            py_files = sorted(p.glob("*.py"))
            if not py_files:
                console.print(f"[error]No Python files found in {p}[/error]")
                raise typer.Exit(1)
            files.extend(py_files)
        elif p.exists():
            files.append(p)
        else:
            console.print(f"[error]File not found: {p}[/error]")
            raise typer.Exit(1)
    return files


# ---------------------------------------------------------------------------
# Winner determination
# ---------------------------------------------------------------------------


def _determine_case_winner(
    contender_results: dict[str, CaseResult],
) -> str | None:
    """Determine winner for a single case. Returns name or None for tie."""
    if not contender_results:
        return None

    passed = {n: r for n, r in contender_results.items() if r.passed}

    # If exactly one passed, they win
    if len(passed) == 1:
        return next(iter(passed))

    # Compare by score among the relevant group
    group = passed if passed else contender_results
    scores = {n: r.score for n, r in group.items()}
    max_score = max(scores.values())
    top = [n for n, s in scores.items() if s == max_score]

    if len(top) == 1:
        return top[0]

    # Tiebreak by speed (>10% difference)
    times = {n: group[n].elapsed for n in top}
    fastest = min(times, key=lambda n: times[n])
    slowest_t = max(times.values())
    if slowest_t > 0 and (slowest_t - times[fastest]) / slowest_t > 0.1:
        return fastest

    return None


# ---------------------------------------------------------------------------
# ELO rating system
# ---------------------------------------------------------------------------


def _compute_elo(rating_a: float, rating_b: float, winner: str | None, k: float = 32.0) -> tuple[float, float]:
    """Standard ELO computation. winner is 'a', 'b', or None for draw."""
    ea = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
    eb = 1.0 - ea

    if winner == "a":
        sa, sb = 1.0, 0.0
    elif winner == "b":
        sa, sb = 0.0, 1.0
    else:
        sa, sb = 0.5, 0.5

    return rating_a + k * (sa - ea), rating_b + k * (sb - eb)


def _load_leaderboard() -> dict[str, Any]:
    if LEADERBOARD_PATH.exists():
        return json.loads(LEADERBOARD_PATH.read_text())
    return {"agents": {}}


def _save_leaderboard(data: dict[str, Any]) -> None:
    LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEADERBOARD_PATH.write_text(json.dumps(data, indent=2))


def _update_leaderboard(name_a: str, name_b: str, winner: str | None) -> None:
    """Update ELO ratings after a match."""
    lb = _load_leaderboard()
    agents = lb["agents"]

    for name in (name_a, name_b):
        if name not in agents:
            agents[name] = {"elo": 1500.0, "wins": 0, "losses": 0, "ties": 0}

    ra, rb = agents[name_a]["elo"], agents[name_b]["elo"]

    if winner == name_a:
        w = "a"
        agents[name_a]["wins"] += 1
        agents[name_b]["losses"] += 1
    elif winner == name_b:
        w = "b"
        agents[name_b]["wins"] += 1
        agents[name_a]["losses"] += 1
    else:
        w = None
        agents[name_a]["ties"] += 1
        agents[name_b]["ties"] += 1

    new_a, new_b = _compute_elo(ra, rb, w)
    agents[name_a]["elo"] = round(new_a, 1)
    agents[name_b]["elo"] = round(new_b, 1)
    _save_leaderboard(lb)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _flat_cases(suites: list[EvalSuite]) -> list[EvalCase]:
    """Flatten all cases from all suites into a single list."""
    return [case for suite in suites for case in suite.cases]


def _display_comparison(contenders: list[ContenderResult], suites: list[EvalSuite]) -> dict[str, int]:
    """Display side-by-side comparison. Returns win counts per contender."""
    names = [c.name for c in contenders]
    cases = _flat_cases(suites)

    # Header
    console.print(
        Panel(
            f"[dim]Contenders: {' vs '.join(names)} | Eval cases: {len(cases)}[/dim]",
            title="[bold cyan]AG2 Arena[/bold cyan]",
            border_style="cyan",
            width=72,
        )
    )
    console.print()

    # Per-case table
    table = Table(show_edge=False, pad_edge=False, box=None, width=72)
    table.add_column("Case", style="dim", min_width=20)
    for name in names:
        table.add_column(name, justify="center", min_width=12)
    table.add_column("Winner", justify="center", min_width=12)

    wins: dict[str, int] = dict.fromkeys(names, 0)
    ties = 0

    for i, case in enumerate(cases):
        row: list[str] = [case.name]
        case_results: dict[str, CaseResult] = {}

        for c in contenders:
            r = c.case_results[i] if i < len(c.case_results) else None
            if r is None:
                row.append("[dim]-[/dim]")
                continue
            if r.passed:
                cell = "[success]\u2713[/success]"
                if r.score < 1.0:
                    cell += f" ({r.score:.2f})"
            else:
                cell = "[error]\u2717[/error]"
                if r.score > 0:
                    cell += f" ({r.score:.2f})"
            row.append(cell)
            case_results[c.name] = r

        winner = _determine_case_winner(case_results)
        if winner:
            row.append(f"[bold]{winner}[/bold]")
            wins[winner] += 1
        else:
            row.append("[dim]tie[/dim]")
            ties += 1

        table.add_row(*row)

    console.print(table)
    console.print()

    # Summary panel
    summary = Table(show_header=False, show_edge=False, pad_edge=False, box=None)
    summary.add_column(style="dim", min_width=16)
    for _ in names:
        summary.add_column(min_width=16)

    # Pass rate
    pass_rates = [c.pass_rate for c in contenders]
    rate_row: list[str] = ["Pass rate"]
    best_pr = max(pass_rates)
    for c in contenders:
        pct = f"{c.pass_rate * 100:.0f}%"
        if c.pass_rate == best_pr and pass_rates.count(best_pr) == 1:
            pct = f"[success]{pct}[/success]"
        rate_row.append(pct)
    summary.add_row(*rate_row)

    # Avg quality
    summary.add_row("Avg quality", *[f"{c.avg_score:.2f}" for c in contenders])

    # Avg time
    summary.add_row("Avg time", *[f"{c.avg_elapsed:.1f}s" for c in contenders])

    # Cost
    summary.add_row("Total cost", *[f"${c.total_cost:.4f}" for c in contenders])

    # Overall winner
    if len(contenders) == 2:
        a, b = contenders
        aw, bw = wins[a.name], wins[b.name]
        if aw > bw:
            winner_str = f"[bold]{a.name}[/bold] (better on {aw}, worse on {bw})"
        elif bw > aw:
            winner_str = f"[bold]{b.name}[/bold] (better on {bw}, worse on {aw})"
        else:
            winner_str = "[dim]Tie[/dim]"
    else:
        best = max(contenders, key=lambda c: wins[c.name])
        winner_str = f"[bold]{best.name}[/bold] ({wins[best.name]} wins)"

    summary.add_row("", *[""] * len(names))
    # Pad winner row to fill columns
    winner_row: list[str] = ["Winner:", winner_str]
    winner_row.extend([""] * (len(names) - 1))
    summary.add_row(*winner_row)

    console.print(Panel(summary, title="Summary", border_style="cyan", width=72))
    console.print()

    return wins


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("compare")
def arena_compare(
    contenders: list[Path] = typer.Argument(..., help="Agent files or directories to compare."),
    eval_file: Path = typer.Option(..., "--eval", "-e", help="Evaluation cases file (YAML) or directory."),
    output: str | None = typer.Option(None, "--output", "-o", help="Output format: json."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate cost without running."),
) -> None:
    """Compare agent implementations on the same eval suite.

    [dim]Examples:[/dim]
      [command]ag2 arena compare agent_v1.py agent_v2.py --eval tests/cases.yaml[/command]
      [command]ag2 arena compare agents/ --eval tests/ --output json[/command]
    """
    try:
        import autogen  # noqa: F401
    except ImportError:
        console.print("[error]ag2 is not installed.[/error]")
        console.print("Install with: [command]pip install ag2[/command]")
        raise typer.Exit(1)

    files = _resolve_contender_files(contenders)
    if len(files) < 2:
        console.print("[error]Need at least 2 contenders to compare.[/error]")
        raise typer.Exit(1)

    suites = _load_suites(eval_file)
    cases = _flat_cases(suites)

    if dry_run:
        console.print(
            f"\n[heading]Dry run:[/heading] {len(files)} contenders x {len(cases)} cases "
            f"= {len(files) * len(cases)} total runs"
        )
        console.print("[dim]Estimated cost depends on model and input length.[/dim]")
        raise typer.Exit(0)

    # Run all contenders
    results: list[ContenderResult] = []
    for f in files:
        console.print(f"[dim]Running {f.stem}...[/dim]")
        results.append(_run_contender(f, suites))

    wins = _display_comparison(results, suites)

    # Update leaderboard for pairwise matchups
    if len(results) == 2:
        a, b = results
        overall = a.name if wins[a.name] > wins[b.name] else b.name if wins[b.name] > wins[a.name] else None
        _update_leaderboard(a.name, b.name, overall)

    # JSON output
    if output == "json":
        json_data = {
            "contenders": [
                {
                    "name": c.name,
                    "source": c.source,
                    "pass_rate": c.pass_rate,
                    "avg_score": c.avg_score,
                    "avg_elapsed": round(c.avg_elapsed, 2),
                    "total_cost": c.total_cost,
                    "cases": [
                        {
                            "name": r.case.name,
                            "passed": r.passed,
                            "score": r.score,
                            "output": r.output[:500],
                            "elapsed": round(r.elapsed, 2),
                        }
                        for r in c.case_results
                    ],
                }
                for c in results
            ],
            "wins": wins,
        }
        print(json.dumps(json_data, indent=2, default=str))


@app.command("models")
def arena_models(
    agent_file: Path = typer.Argument(..., help="Agent file to test across models."),
    models: str = typer.Option(..., "--models", "-m", help="Comma-separated model names."),
    eval_file: Path = typer.Option(..., "--eval", "-e", help="Evaluation cases file (YAML) or directory."),
    output: str | None = typer.Option(None, "--output", "-o", help="Output format: json."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate cost without running."),
) -> None:
    """Compare same agent across different LLM models.

    [dim]Examples:[/dim]
      [command]ag2 arena models my_agent.py --models gpt-4o,claude-sonnet-4-6 --eval tests/cases.yaml[/command]
    """
    try:
        import autogen  # noqa: F401
    except ImportError:
        console.print("[error]ag2 is not installed.[/error]")
        raise typer.Exit(1)

    agent_path = Path(agent_file).resolve()
    if not agent_path.exists():
        console.print(f"[error]Agent file not found: {agent_path}[/error]")
        raise typer.Exit(1)

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if len(model_list) < 2:
        console.print("[error]Need at least 2 models to compare.[/error]")
        raise typer.Exit(1)

    suites = _load_suites(eval_file)
    cases = _flat_cases(suites)

    if dry_run:
        console.print(
            f"\n[heading]Dry run:[/heading] {len(model_list)} models x {len(cases)} cases "
            f"= {len(model_list) * len(cases)} total runs"
        )
        raise typer.Exit(0)

    # Run the agent with each model by setting an env var override
    import os

    results: list[ContenderResult] = []
    for model in model_list:
        console.print(f"[dim]Running with model {model}...[/dim]")
        # Use environment variable to override model
        old_val = os.environ.get("AG2_MODEL_OVERRIDE")
        os.environ["AG2_MODEL_OVERRIDE"] = model
        try:
            cr = _run_contender(agent_path, suites, name=model)
        finally:
            if old_val is None:
                os.environ.pop("AG2_MODEL_OVERRIDE", None)
            else:
                os.environ["AG2_MODEL_OVERRIDE"] = old_val
        results.append(cr)

    _display_comparison(results, suites)

    if output == "json":
        json_data = {
            "agent": str(agent_path),
            "models": [
                {
                    "model": c.name,
                    "pass_rate": c.pass_rate,
                    "avg_score": c.avg_score,
                    "avg_elapsed": round(c.avg_elapsed, 2),
                    "total_cost": c.total_cost,
                }
                for c in results
            ],
        }
        print(json.dumps(json_data, indent=2, default=str))


@app.command("interactive")
def arena_interactive(
    contenders: list[Path] = typer.Argument(..., help="Two agent files for head-to-head comparison."),
) -> None:
    """Interactive head-to-head with human judgment.

    Send the same message to both agents (anonymized as A/B).
    Pick the winner each round. ELO ratings are updated.

    [dim]Examples:[/dim]
      [command]ag2 arena interactive agent_v1.py agent_v2.py[/command]
    """
    try:
        import autogen  # noqa: F401
    except ImportError:
        console.print("[error]ag2 is not installed.[/error]")
        raise typer.Exit(1)

    files = _resolve_contender_files(contenders)
    if len(files) != 2:
        console.print("[error]Interactive mode requires exactly 2 contenders.[/error]")
        raise typer.Exit(1)

    from ..core.discovery import discover

    # Randomize assignment so users can't guess by position
    order = list(range(2))
    random.shuffle(order)
    labels = ["A", "B"]
    file_order = [files[order[0]], files[order[1]]]
    real_names = [f.stem for f in file_order]

    console.print(
        Panel(
            "[dim]Send the same message to both agents.\n"
            "Pick the winner for each round. Identities revealed at the end.[/dim]",
            title="[bold cyan]AG2 Arena \u2014 Interactive[/bold cyan]",
            border_style="cyan",
            width=60,
        )
    )
    console.print()

    scores = {"A": 0, "B": 0, "ties": 0}
    rounds = 0

    while True:
        try:
            user_input = console.input("[bold]You:[/bold] ")
        except (KeyboardInterrupt, EOFError):
            break

        user_input = user_input.strip()
        if not user_input or user_input.lower() in ("/quit", "/exit", "/q"):
            break

        # Run both agents
        for i, (label, agent_file) in enumerate(zip(labels, file_order)):
            try:
                discovered = discover(agent_file)
                result = execute(discovered, user_input)
                output = result.output or "[dim]No output[/dim]"
                meta = f"{result.turns} turns, {result.elapsed:.1f}s"
                if result.cost:
                    cost = extract_cost(result.cost)
                    if cost > 0:
                        meta += f", ${cost:.4f}"
            except Exception as exc:
                output = f"[error]Error: {exc}[/error]"
                meta = "error"

            console.print(
                Panel(
                    output,
                    title=f"Agent {label}",
                    subtitle=f"[dim]({meta})[/dim]",
                    border_style="blue" if label == "A" else "magenta",
                    width=60,
                )
            )
            console.print()

        # Get human judgment
        console.print(
            "  Which is better? [bold][A][/bold] Agent A  "
            "[bold][B][/bold] Agent B  [bold][T][/bold] Tie  "
            "[bold][S][/bold] Skip"
        )
        try:
            choice = console.input("  > ").strip().upper()
        except (KeyboardInterrupt, EOFError):
            break

        if choice == "A":
            scores["A"] += 1
            _update_leaderboard(real_names[0], real_names[1], real_names[0])
        elif choice == "B":
            scores["B"] += 1
            _update_leaderboard(real_names[0], real_names[1], real_names[1])
        elif choice == "T":
            scores["ties"] += 1
            _update_leaderboard(real_names[0], real_names[1], None)

        rounds += 1
        console.print(f"  [dim]Score: Agent A: {scores['A']}  Agent B: {scores['B']}  Ties: {scores['ties']}[/dim]\n")

    # Reveal identities
    if rounds > 0:
        console.print()
        console.print(
            Panel(
                f"  Agent A = [bold]{real_names[0]}[/bold] ({files[order[0]].name})\n"
                f"  Agent B = [bold]{real_names[1]}[/bold] ({files[order[1]].name})\n\n"
                f"  Final: A={scores['A']}  B={scores['B']}  Ties={scores['ties']}",
                title="Identity Reveal",
                border_style="cyan",
                width=60,
            )
        )


@app.command("leaderboard")
def arena_leaderboard() -> None:
    """Show the ELO leaderboard from past arena sessions.

    [dim]Examples:[/dim]
      [command]ag2 arena leaderboard[/command]
    """
    lb = _load_leaderboard()
    agents = lb.get("agents", {})

    if not agents:
        console.print("[dim]No arena results yet. Run some comparisons first.[/dim]")
        raise typer.Exit(0)

    # Sort by ELO descending
    ranked = sorted(agents.items(), key=lambda kv: kv[1]["elo"], reverse=True)

    table = Table(title="Agent Leaderboard", width=60)
    table.add_column("Rank", justify="right", style="dim", width=5)
    table.add_column("Agent", min_width=20)
    table.add_column("ELO", justify="right", width=8)
    table.add_column("W/L/T", justify="center", width=10)

    for rank, (name, stats) in enumerate(ranked, 1):
        elo = f"{stats['elo']:.0f}"
        wlt = f"{stats['wins']}/{stats['losses']}/{stats['ties']}"
        table.add_row(str(rank), name, elo, wlt)

    console.print()
    console.print(table)
    console.print()


@app.command("reset")
def arena_reset() -> None:
    """Reset the ELO leaderboard.

    [dim]Examples:[/dim]
      [command]ag2 arena reset[/command]
    """
    if LEADERBOARD_PATH.exists():
        LEADERBOARD_PATH.unlink()
        console.print("[success]Leaderboard reset.[/success]")
    else:
        console.print("[dim]No leaderboard to reset.[/dim]")
