---
name: ag2-shell-tool
description: Give an AG2 beta `Agent` the ability to run shell commands. Covers `LocalShellTool` (client-side `subprocess`, works with any provider) and the provider-native `ShellTool` (Anthropic / OpenAI execution). Use when the user wants the Agent to execute commands, build/test code, manage files, or operate on a workspace. Always pair with sandboxing — `allowed`, `blocked`, `ignore`, or `readonly`.
license: Apache-2.0
---

# Shell tools

## When to use

Two distinct tools, both named "shell" — pick deliberately:

| Need | Use | Why |
|---|---|---|
| Works with any model provider; full control over what runs and where | `LocalShellTool` | Client-side `subprocess`. You own the sandbox. |
| Provider-managed sandbox (container, network policy) on Anthropic / OpenAI | `ShellTool` | Server-side execution. No local subprocess. |

**`LocalShellTool` is the workhorse**. Reach for it unless you specifically need provider-managed isolation and you're on Anthropic or OpenAI.

## 60-second recipe — `LocalShellTool`

```python
from autogen.beta import Agent
from autogen.beta.config import AnthropicConfig
from autogen.beta.tools import LocalShellTool

agent = Agent(
    "coder",
    "You write and run Python code.",
    config=AnthropicConfig(model="claude-sonnet-4-6"),
    tools=[LocalShellTool()],
)

reply = await agent.ask("Write a hello world script and run it.")
print(await reply.content())
```

With no arguments, `LocalShellTool` creates a temporary working directory (prefixed `ag2_shell_`) and cleans it up when the process exits. Pass a path to use a specific directory:

```python
from pathlib import Path
LocalShellTool("/tmp/my_project")
LocalShellTool(Path("/tmp/my_project"))
```

When a path is given, the directory is created if it does not exist and is **not** deleted on exit. Inspect the resolved working directory via `tool.workdir`.

## Sandboxing (`LocalShellEnvironment`)

For anything beyond a throwaway demo, use `LocalShellEnvironment` and lock down what the agent can do. Filtering is applied in this order on every call:

1. `allowed` — if set, the command must match at least one prefix.
2. `blocked` — if set, the command must not match any prefix.
3. `ignore` — literal path tokens in the command are checked against gitignore-style patterns; matches return `"Access denied: <path>"`.
4. Execute via `subprocess.run`.

```python
from autogen.beta.tools import LocalShellTool
from autogen.beta.tools.shell import LocalShellEnvironment

sh = LocalShellTool(
    LocalShellEnvironment(
        path="/tmp/my_project",
        allowed=["python", "uv run", "git"],
        blocked=["rm -rf", "curl", "wget"],
        ignore=["**/.env", "*.key", "secrets/**"],
        timeout=30,
        max_output=50_000,
    )
)
```

### Read-only mode

For inspection-only access (`cat`, `head`, `tail`, `ls`, `grep`, `find`, `git log`, `git diff`, `git status`, …):

```python
from autogen.beta.tools import LocalShellTool
from autogen.beta.tools.shell import LocalShellEnvironment

sh = LocalShellTool(LocalShellEnvironment(path="/my/codebase", readonly=True))
```

Pass an explicit `allowed=[...]` to override the built-in read-only allowlist.

### `LocalShellEnvironment` parameter reference

| Parameter | Default | Description |
|---|---|---|
| `path` | `None` | Working dir. `None` → temp dir, deleted on exit |
| `cleanup` | `None` | `None` → auto (`True` when `path=None`, `False` otherwise). Deletes `path` on process exit |
| `allowed` | `None` | Whitelist of command prefixes. `None` → all commands allowed |
| `blocked` | `None` | Blacklist of command prefixes |
| `ignore` | `None` | Gitignore-style path patterns; matches block the command |
| `readonly` | `False` | When `True` and `allowed` unset, restricts to a built-in read-only list |
| `env` | `None` | Extra env vars merged into each command |
| `timeout` | `60` | Per-command timeout in seconds (returns `"Command timed out after Ns [exit code: 124]"`) |
| `max_output` | `100_000` | Max characters returned (truncated output is suffixed `[truncated: …]`) |

## Stateful multi-turn workspaces

Files persist in `workdir` across `ask()` calls, so the agent can build on prior work:

```python
from autogen.beta.tools import LocalShellTool
from autogen.beta.tools.shell import LocalShellEnvironment

sh = LocalShellTool(LocalShellEnvironment(path="/tmp/counter_demo"))
agent = Agent("coder", "You manage files.", config=config, tools=[sh])

reply1 = await agent.ask("Create counter.txt with value 0")
reply2 = await reply1.ask("Increment the counter by 1")
reply3 = await reply2.ask("Read the counter and tell me the value")
```

## Provider-native `ShellTool` (Anthropic / OpenAI)

```python
from autogen.beta.tools import ShellTool

agent = Agent("devops", config=AnthropicConfig(model="claude-sonnet-4-6"), tools=[ShellTool()])
```

OpenAI lets you configure the execution environment:

```python
from autogen.beta.config import OpenAIResponsesConfig
from autogen.beta.tools import ShellTool
from autogen.beta.tools.builtin.shell import ContainerAutoEnvironment, NetworkPolicy

agent = Agent(
    "devops",
    config=OpenAIResponsesConfig(model="gpt-4.1"),
    tools=[
        ShellTool(
            environment=ContainerAutoEnvironment(
                network_policy=NetworkPolicy(allowed_domains=["pypi.org"]),
            ),
        ),
    ],
)
```

Environment options:

| Environment | Description |
|---|---|
| `ContainerAutoEnvironment` | Provider-managed container with optional network policy |
| `ContainerReferenceEnvironment` | Reference an existing container by ID |

`ShellTool` is **not supported on Gemini** — the request will raise `UnsupportedToolError`.

## `LocalShellTool` vs `ShellTool`

| | `LocalShellTool` | `ShellTool` |
|---|---|---|
| **Execution** | Client-side `subprocess` | Provider-side container |
| **Provider support** | Any provider | Anthropic, OpenAI |
| **Environment control** | Full (`allowed`, `blocked`, `ignore`, `readonly`, …) | Limited (provider-dependent) |
| **Local FS access** | Yes (you choose what's exposed) | No |
| **Network control** | Via `blocked` / `allowed` patterns | OpenAI: `NetworkPolicy` |
| **Import** | `from autogen.beta.tools import LocalShellTool` (env: `from autogen.beta.tools.shell import LocalShellEnvironment`) | `from autogen.beta.tools import ShellTool` |

## Going deeper

- `website/docs/beta/tools/local_shell.mdx` — full `LocalShellTool` reference, command-filtering semantics.
- `website/docs/beta/tools/builtin_tools.mdx#shell` — provider-native `ShellTool` setup and environment configs.
- For **human-approval gating before each shell call**, layer `approval_required()` middleware (see `ag2-hitl`).

## Common pitfalls

- **Forgetting sandboxing in production** — `LocalShellTool()` with no environment runs anything anywhere with a 60s timeout. Set `allowed`, `blocked`, or `readonly` for any non-trivial use.
- **`ignore` only checks literal paths in the command string** — variable substitution, command substitution (`` `cat secrets.key` ``), and dynamic glob expansion are not inspected. Layer in `blocked=["cat", "less"]` if you also want to block readers.
- **Trying to use `ShellTool` on Gemini** — unsupported, will raise. Use `LocalShellTool` instead.
- **Using a hardcoded path that another process is also touching** — multiple agents sharing `/tmp/my_project` will race. Use `tempfile.mkdtemp(prefix="...")` for parallel runs.
- **Expecting `ShellTool` to access local files** — it doesn't; it runs in the provider's container. Use `LocalShellTool` for anything on your filesystem.
- **Trusting the LLM with shell access** — even sandboxed, write `prompt`s that scope what's allowed and consider pairing with `approval_required()` for destructive operations.
