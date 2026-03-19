"""ag2 test — agent evaluation and testing framework."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer
from rich.panel import Panel
from rich.table import Table

from ..testing import EvalCase, EvalSuite, check_assertion, load_eval_suite
from ..testing.assertions import AssertionResult
from ..ui import console

app = typer.Typer(
    help="Test, evaluate, and benchmark agents.",
    rich_markup_mode="rich",
)


@dataclass
class CaseResult:
    """Result of running a single eval case."""

    case: EvalCase
    assertion_results: list[AssertionResult] = field(default_factory=list)
    output: str = ""
    turns: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.assertion_results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.assertion_results if r.passed)

    @property
    def total_count(self) -> int:
        return len(self.assertion_results)


def _run_single_case(
    agent_file: Path,
    case: EvalCase,
) -> CaseResult:
    """Run a single eval case against a fresh agent instance."""
    errors: list[str] = []
    output = ""
    turns = 0

    start = time.time()

    try:
        import autogen

        # Discover agent fresh each time for isolation
        from ..core.discovery import discover

        discovered = discover(agent_file)

        if discovered.kind == "main" and discovered.main_fn is not None:
            import asyncio
            import inspect

            kwargs: dict[str, Any] = {}
            sig = inspect.signature(discovered.main_fn)
            if "message" in sig.parameters:
                kwargs["message"] = case.input

            result = discovered.main_fn(**kwargs)
            if asyncio.iscoroutine(result):
                result = asyncio.run(result)

            if hasattr(result, "chat_history"):
                history = result.chat_history or []
                turns = len(history)
                agent_msgs = [
                    m for m in history if m.get("role") != "user" and m.get("content")
                ]
                output = agent_msgs[-1]["content"] if agent_msgs else ""
            else:
                output = str(result)
                turns = 1

        elif discovered.kind == "agent":
            user_proxy = autogen.UserProxyAgent(
                name="user",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=0,
                code_execution_config=False,
            )
            result = user_proxy.initiate_chat(
                discovered.agent,
                message=case.input,
            )
            history = result.chat_history or []
            turns = len(history)
            agent_msgs = [
                m for m in history if m.get("role") != "user" and m.get("content")
            ]
            output = agent_msgs[-1]["content"] if agent_msgs else ""

        elif discovered.kind == "agents":
            groupchat = autogen.GroupChat(
                agents=discovered.agents,
                messages=[],
                max_round=10,
            )
            manager = autogen.GroupChatManager(groupchat=groupchat)
            user_proxy = autogen.UserProxyAgent(
                name="user",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=0,
                code_execution_config=False,
            )
            result = user_proxy.initiate_chat(manager, message=case.input)
            history = result.chat_history or []
            turns = len(history)
            agent_msgs = [
                m for m in history if m.get("role") != "user" and m.get("content")
            ]
            output = agent_msgs[-1]["content"] if agent_msgs else ""

    except Exception as exc:
        errors.append(str(exc))

    elapsed = time.time() - start

    # Evaluate assertions
    assertion_results = [
        check_assertion(a, output, turns=turns, errors=errors)
        for a in case.assertions
    ]

    return CaseResult(
        case=case,
        assertion_results=assertion_results,
        output=output,
        turns=turns,
        errors=errors,
        elapsed=elapsed,
    )


def _display_results(suite: EvalSuite, results: list[CaseResult]) -> None:
    """Display eval results with Rich formatting."""
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    total_assertions = sum(r.total_count for r in results)
    passed_assertions = sum(r.passed_count for r in results)
    total_time = sum(r.elapsed for r in results)

    # Header
    console.print(
        Panel(
            f"[dim]Suite: {suite.name} | Cases: {total} | Assertions: {total_assertions}[/dim]",
            title="[bold cyan]AG2 Test[/bold cyan]",
            border_style="cyan",
            width=60,
        )
    )
    console.print()

    # Per-case results
    for r in results:
        mark = "[success]✓[/success]" if r.passed else "[error]✗[/error]"
        assertions_str = f"{r.passed_count}/{r.total_count} assertions"
        console.print(
            f"  {mark} {r.case.name:30s} {assertions_str:20s} {r.elapsed:.1f}s"
        )

        # Show failures
        for ar in r.assertion_results:
            if not ar.passed:
                console.print(f"    [error]└─ FAIL:[/error] {ar.assertion_type}: {ar.message}")

    # Summary
    console.print()
    pct = (passed / total * 100) if total else 0
    style = "success" if passed == total else "warning" if passed > 0 else "error"
    apct = (passed_assertions / total_assertions * 100) if total_assertions else 0

    summary_table = Table(show_header=False, show_edge=False, pad_edge=False, box=None)
    summary_table.add_column(style="dim")
    summary_table.add_column()
    summary_table.add_row("Passed:", f"[{style}]{passed}/{total} ({pct:.0f}%)[/{style}]")
    summary_table.add_row("Assertions:", f"{passed_assertions}/{total_assertions} ({apct:.0f}%)")
    summary_table.add_row("Total time:", f"{total_time:.1f}s")

    console.print(
        Panel(summary_table, title="Results", border_style=style, width=60)
    )
    console.print()


@app.command("eval")
def test_eval(
    agent_file: Path = typer.Argument(..., help="Python file defining agent(s) to test."),
    eval_file: Path = typer.Option(..., "--eval", "-e", help="Evaluation cases file (YAML)."),
    models: str | None = typer.Option(None, "--models", help="Comma-separated models to compare."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate cost without running."),
    baseline: Path | None = typer.Option(None, "--baseline", help="Previous results for regression."),
) -> None:
    """Run evaluation suite against an agent.

    [dim]Examples:[/dim]
      [command]ag2 test eval my_agent.py --eval tests/cases.yaml[/command]
      [command]ag2 test eval my_agent.py --eval tests/ --models gpt-4o,claude-sonnet-4-6[/command]
      [command]ag2 test eval my_agent.py --eval tests/ --dry-run[/command]
    """
    try:
        import autogen  # noqa: F401
    except ImportError:
        console.print("[error]ag2 is not installed.[/error]")
        console.print("Install with: [command]pip install ag2[/command]")
        raise typer.Exit(1)

    agent_path = Path(agent_file).resolve()
    if not agent_path.exists():
        console.print(f"[error]Agent file not found: {agent_path}[/error]")
        raise typer.Exit(1)

    eval_path = Path(eval_file).resolve()

    # Load eval suite(s)
    if eval_path.is_dir():
        import yaml  # noqa: F401

        yaml_files = sorted(eval_path.glob("*.yaml")) + sorted(eval_path.glob("*.yml"))
        if not yaml_files:
            console.print(f"[error]No YAML files found in {eval_path}[/error]")
            raise typer.Exit(1)
        suites = [load_eval_suite(f) for f in yaml_files]
    else:
        suites = [load_eval_suite(eval_path)]

    if models:
        console.print("[warning]Multi-model comparison is coming soon.[/warning]")

    if dry_run:
        total_cases = sum(len(s.cases) for s in suites)
        console.print(f"\n[heading]Dry run:[/heading] {total_cases} case(s) across {len(suites)} suite(s)")
        console.print("[dim]Estimated cost depends on model and input length.[/dim]")
        raise typer.Exit(0)

    # Run each suite
    all_passed = True
    for suite in suites:
        console.print(f"\n[heading]Running:[/heading] {suite.name} ({len(suite.cases)} cases)\n")
        results = [_run_single_case(agent_path, case) for case in suite.cases]
        _display_results(suite, results)
        if not all(r.passed for r in results):
            all_passed = False

    if not all_passed:
        raise typer.Exit(1)


@app.command("bench")
def test_bench(
    agent_file: Path = typer.Argument(..., help="Python file defining agent(s) to benchmark."),
    suite: str = typer.Option(..., "--suite", "-s", help="Benchmark suite (gaia, humaneval, swe-bench-lite, or path)."),
) -> None:
    """Run standardized benchmarks against an agent.

    [dim]Examples:[/dim]
      [command]ag2 test bench my_agent.py --suite gaia[/command]
      [command]ag2 test bench my_agent.py --suite ./my_benchmarks/[/command]
    """
    console.print("[warning]ag2 test bench is coming soon.[/warning]")
    console.print(f"Suite: {suite}")
    console.print("See [command]cli/docs/test.md[/command] for the design.")
    raise typer.Exit(0)
