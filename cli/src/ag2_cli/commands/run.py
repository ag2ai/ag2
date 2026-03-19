"""ag2 run / ag2 chat — run agents and interactive chat sessions."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import typer
from rich.panel import Panel

from ..ui import console


def _require_ag2() -> Any:
    """Import autogen or exit with a helpful error."""
    try:
        import autogen

        return autogen
    except ImportError:
        console.print("[error]ag2 is not installed.[/error]")
        console.print("Install it with: [command]pip install ag2[/command]")
        raise typer.Exit(1)


def _display_header(discovered: Any, verbose: bool = False) -> None:
    """Display a header panel before running."""
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
            title="[bold cyan]AG2 Run[/bold cyan]",
            border_style="cyan",
            width=60,
        )
    )
    console.print()


def _display_result(result: Any, verbose: bool = False) -> None:
    """Display the result of an agent run."""
    if result is None:
        return

    # Handle ChatResult objects
    if hasattr(result, "chat_history") and result.chat_history:
        for msg in result.chat_history:
            speaker = msg.get("name", msg.get("role", "agent"))
            content = msg.get("content", "")
            if not content:
                continue

            if speaker == "user" or msg.get("role") == "user":
                continue

            console.print(
                Panel(
                    content,
                    title=f"[bold]{speaker}[/bold]",
                    border_style="bright_cyan",
                    padding=(0, 1),
                )
            )

        if verbose and hasattr(result, "cost"):
            console.print(f"\n[dim]Cost: {result.cost}[/dim]")

    elif isinstance(result, str):
        console.print(result)


def _run_discovered(discovered: Any, message: str | None, verbose: bool) -> Any:
    """Execute a discovered agent."""
    ag2 = _require_ag2()
    from ..core.discovery import DiscoveredAgent

    d: DiscoveredAgent = discovered

    if d.kind == "main":
        fn = d.main_fn
        if fn is None:
            raise typer.Exit(1)
        # main() may accept a message parameter
        import inspect

        sig = inspect.signature(fn)
        kwargs: dict[str, Any] = {}
        if "message" in sig.parameters and message:
            kwargs["message"] = message

        result = fn(**kwargs)
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
        return result

    if d.kind == "agent":
        agent = d.agent
        if message is None:
            console.print("[error]--message / -m is required when running a single agent.[/error]")
            raise typer.Exit(1)

        user_proxy = ag2.UserProxyAgent(
            name="user",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False,
        )
        result = user_proxy.initiate_chat(agent, message=message)
        return result

    if d.kind == "agents":
        if message is None:
            console.print("[error]--message / -m is required when running a team.[/error]")
            raise typer.Exit(1)

        agents = d.agents
        try:
            from autogen.agentchat.group import run_group_chat
            from autogen.agentchat.group.patterns.pattern import AutoPattern

            pattern = AutoPattern(
                initial_agent=agents[0],
                agents=agents,
            )
            response = run_group_chat(
                pattern=pattern,
                messages=message,
                max_rounds=10,
            )
            # Consume the response
            last = None
            for event in response:
                last = event
            return last
        except ImportError:
            # Fallback to classic GroupChat
            groupchat = ag2.GroupChat(agents=agents, messages=[], max_round=10)
            manager = ag2.GroupChatManager(groupchat=groupchat)
            user_proxy = ag2.UserProxyAgent(
                name="user",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=0,
                code_execution_config=False,
            )
            result = user_proxy.initiate_chat(manager, message=message)
            return result

    console.print(f"[error]Unknown discovery kind: {d.kind}[/error]")
    raise typer.Exit(1)


def run_cmd(
    agent_file: Path = typer.Argument(..., help="Python file or YAML config defining agents."),
    message: str | None = typer.Option(None, "--message", "-m", help="Input message to send."),
    verbose: bool = typer.Option(False, "--verbose", "-V", help="Show detailed agent activity."),
) -> None:
    """Run an agent or team from a Python file or YAML config.

    [dim]Examples:[/dim]
      [command]ag2 run my_team.py[/command]
      [command]ag2 run my_team.py --message "Research quantum computing"[/command]
      [command]ag2 run team.yaml[/command]
    """
    _require_ag2()
    path = Path(agent_file).resolve()

    if not path.exists():
        console.print(f"[error]File not found: {path}[/error]")
        raise typer.Exit(1)

    # Read from stdin if no message provided and stdin has data
    if message is None and not sys.stdin.isatty():
        message = sys.stdin.read().strip()

    if path.suffix in (".yaml", ".yml"):
        from ..core.discovery import build_agents_from_yaml, load_yaml_config

        config = load_yaml_config(path)
        discovered = build_agents_from_yaml(config)
    else:
        from ..core.discovery import discover

        try:
            discovered = discover(path)
        except (ValueError, ImportError) as exc:
            console.print(f"[error]{exc}[/error]")
            raise typer.Exit(1)

    _display_header(discovered, verbose)

    try:
        result = _run_discovered(discovered, message, verbose)
        _display_result(result, verbose)
    except Exception as exc:
        console.print(f"\n[error]Error: {exc}[/error]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


def chat_cmd(
    agent_file: Path | None = typer.Argument(None, help="Python file defining agent(s)."),
    model: str | None = typer.Option(None, "--model", "-M", help="LLM model for ad-hoc chat."),
    system: str | None = typer.Option(None, "--system", "-s", help="System prompt for ad-hoc chat."),
    resume: str | None = typer.Option(None, "--resume", help="Resume a previous session by ID."),
    verbose: bool = typer.Option(False, "--verbose", "-V", help="Show detailed agent activity."),
) -> None:
    """Start an interactive terminal chat with an agent or team.

    [dim]Examples:[/dim]
      [command]ag2 chat my_agent.py[/command]
      [command]ag2 chat --model gpt-4o --system "You are a Python expert"[/command]
      [command]ag2 chat my_team.py --verbose[/command]
    """
    ag2 = _require_ag2()

    # Build or discover agent
    if agent_file is not None:
        path = Path(agent_file).resolve()
        if not path.exists():
            console.print(f"[error]File not found: {path}[/error]")
            raise typer.Exit(1)

        if path.suffix in (".yaml", ".yml"):
            from ..core.discovery import build_agents_from_yaml, load_yaml_config

            config = load_yaml_config(path)
            discovered = build_agents_from_yaml(config)
        else:
            from ..core.discovery import discover

            try:
                discovered = discover(path)
            except (ValueError, ImportError) as exc:
                console.print(f"[error]{exc}[/error]")
                raise typer.Exit(1)
    elif model:
        # Ad-hoc chat: create a single agent on the fly
        llm_config = ag2.LLMConfig(api_type="openai", model=model)
        system_msg = system or "You are a helpful assistant."
        with llm_config:
            agent = ag2.AssistantAgent(name="assistant", system_message=system_msg)
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
        console.print("  [command]ag2 chat --model gpt-4o --system \"You are a Python expert\"[/command]")
        raise typer.Exit(1)

    # Display chat header
    if discovered.kind == "agents":
        info = f"Team: {', '.join(discovered.agent_names)}"
    else:
        info = f"Agent: {', '.join(discovered.agent_names)}"

    console.print(
        Panel(
            f"[dim]{info}[/dim]\n[dim]Type /quit to exit[/dim]",
            title="[bold cyan]AG2 Chat[/bold cyan]",
            border_style="cyan",
            width=60,
        )
    )
    console.print()

    # Interactive chat loop
    turn_count = 0
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

        turn_count += 1
        try:
            result = _run_discovered(discovered, user_input, verbose)
            _display_result(result, verbose)
        except Exception as exc:
            console.print(f"[error]Error: {exc}[/error]")
            if verbose:
                console.print_exception()

    console.print(f"\n[dim]Session ended after {turn_count} turn(s).[/dim]")
