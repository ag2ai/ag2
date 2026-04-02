"""ag2 test — agent evaluation and testing framework."""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.panel import Panel
from rich.table import Table

from ..core.runner import execute
from ..testing import CaseResult, EvalCase, EvalSuite, check_assertion, load_eval_suite
from ..ui import console
from ._shared import extract_cost

app = typer.Typer(
    help="Test, evaluate, and benchmark agents.",
    rich_markup_mode="rich",
)


def _run_single_case(agent_file: Path, case: EvalCase) -> CaseResult:
    """Run a single eval case against a fresh agent instance."""
    from ..core.discovery import discover

    start = time.time()

    try:
        discovered = discover(agent_file)
    except (ValueError, ImportError) as exc:
        elapsed = time.time() - start
        return CaseResult(
            case=case,
            errors=[str(exc)],
            elapsed=elapsed,
        )

    result = execute(discovered, case.input)

    # Evaluate assertions
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
        mark = "[success]\u2713[/success]" if r.passed else "[error]\u2717[/error]"
        assertions_str = f"{r.passed_count}/{r.total_count} assertions"
        cost_str = ""
        if r.cost:
            case_cost = extract_cost(r.cost)
            if case_cost > 0:
                cost_str = f"  ${case_cost:.6f}"
        console.print(f"  {mark} {r.case.name:30s} {assertions_str:20s} {r.elapsed:.1f}s{cost_str}")

        # Show failures
        for ar in r.assertion_results:
            if not ar.passed:
                console.print(f"    [error]\u2514\u2500 FAIL:[/error] {ar.assertion_type}: {ar.message}")

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

    # Aggregate cost
    total_cost = sum(extract_cost(r.cost) for r in results if r.cost)
    total_tokens = 0
    for r in results:
        if r.cost and isinstance(r.cost, dict):
            for v in r.cost.get("usage_excluding_cached_inference", {}).values():
                if isinstance(v, dict) and "total_tokens" in v:
                    total_tokens += v["total_tokens"]
    if total_cost > 0:
        summary_table.add_row("Cost:", f"${total_cost:.6f} ({total_tokens} tokens)")

    console.print(Panel(summary_table, title="Results", border_style=style, width=60))
    console.print()


@app.command("eval")
def test_eval(
    agent_file: Path = typer.Argument(..., help="Python file defining agent(s) to test."),
    eval_file: Path = typer.Option(..., "--eval", "-e", help="Evaluation cases file (YAML)."),
    models: str | None = typer.Option(None, "--models", help="Comma-separated models to compare (coming soon)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate cost without running."),
    output: str | None = typer.Option(None, "--output", "-o", help="Output format: json."),
) -> None:
    """Run evaluation suite against an agent.

    [dim]Examples:[/dim]
      [command]ag2 test eval my_agent.py --eval tests/cases.yaml[/command]
      [command]ag2 test eval my_agent.py --eval tests/ --output json[/command]
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
    all_results: list[tuple[EvalSuite, list[CaseResult]]] = []

    for suite in suites:
        console.print(f"\n[heading]Running:[/heading] {suite.name} ({len(suite.cases)} cases)\n")
        results = [_run_single_case(agent_path, case) for case in suite.cases]
        all_results.append((suite, results))
        _display_results(suite, results)
        if not all(r.passed for r in results):
            all_passed = False

    # JSON output
    if output == "json":
        json_data = []
        for suite, results in all_results:
            suite_data = {
                "suite": suite.name,
                "cases": [
                    {
                        "name": r.case.name,
                        "passed": r.passed,
                        "output": r.output[:500],
                        "turns": r.turns,
                        "elapsed": round(r.elapsed, 2),
                        "cost": str(r.cost) if r.cost else None,
                        "assertions": [
                            {
                                "type": ar.assertion_type,
                                "passed": ar.passed,
                                "message": ar.message,
                            }
                            for ar in r.assertion_results
                        ],
                    }
                    for r in results
                ],
            }
            json_data.append(suite_data)
        print(json.dumps(json_data, indent=2, default=str))

    if not all_passed:
        raise typer.Exit(1)


@app.command("bench")
def test_bench(
    agent_file: Path = typer.Argument(..., help="Python file defining agent(s) to benchmark."),
    suite: str = typer.Option(
        ...,
        "--suite",
        "-s",
        help="Benchmark suite (gaia, humaneval, swe-bench-lite, or path).",
    ),
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
