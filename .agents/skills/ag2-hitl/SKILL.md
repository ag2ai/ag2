---
name: ag2-hitl
description: Pause an AG2 beta `Agent` mid-run to collect human input via `context.input()`, or gate a tool call with `approval_required()` middleware. Use when the user wants the agent to ask for confirmation, request missing info (passwords, API keys, data), or have a human approve sensitive / irreversible / expensive tool calls (sending emails, deleting records, payments).
license: Apache-2.0
---

# Human-in-the-loop

## When to use

- The agent should **ask for confirmation** before doing something risky.
- The agent needs **information from the user mid-conversation** (a password, an API key, missing context).
- A specific **tool call should require human approval** before it runs (irreversible / expensive / sensitive).
- Quality assurance — show a draft, get human edits/approval before finalising.

Two distinct mechanisms — pick by intent:

| Need | Use |
|---|---|
| Tool asks an open question and waits for a typed answer | `context.input()` from inside the tool + `hitl_hook` on the agent |
| Approve / deny a specific tool call before its body runs | `approval_required()` tool middleware |

## Pattern 1 — `context.input()` for open questions

A tool requests input via `Context.input(prompt, timeout=...)`. The agent must have a `hitl_hook` that knows how to collect that input.

```python
from autogen.beta import Agent, Context, tool
from autogen.beta.events import HumanInputRequest, HumanMessage

@tool
async def execute_query(context: Context) -> str:
    answer = await context.input(
        "Are you sure you want to run this query? (yes/no)",
        timeout=60.0,
    )
    if answer.strip().lower() != "yes":
        return "Query cancelled."
    return "Query executed successfully."

def hitl_hook(event: HumanInputRequest) -> HumanMessage:
    print(f"Agent asks: {event.content}")
    return HumanMessage(content=input("Your answer: "))

agent = Agent("dba", tools=[execute_query], hitl_hook=hitl_hook)
```

The hook receives a `HumanInputRequest` (containing the prompt) and must return a `HumanMessage`. Both `def` and `async def` hooks are supported.

You can also register the hook after construction:

```python
agent = Agent("dba", tools=[execute_query])

@agent.hitl_hook
async def async_hitl_hook(event: HumanInputRequest) -> HumanMessage:
    answer = await collect_from_ui(event.content)
    return HumanMessage(content=answer)
```

The decorator overrides any hook set in the constructor.

The hook participates in dependency injection — `Context`, `Inject`, `Variable`, `Depends` work the same as in tools.

If `context.input()` is called and no hook is registered, the framework raises `HumanInputNotProvidedError`.

## Pattern 2 — `approval_required()` for specific tool calls

Gate a single tool with the built-in approval middleware. The user is prompted before the tool body runs and can approve or deny.

```python
from autogen.beta import Agent, tool
from autogen.beta.config import OpenAIConfig
from autogen.beta.middleware import approval_required

@tool(middleware=[approval_required()])
def delete_account(user_id: str) -> str:
    """Deletes a user account by ID permanently."""
    return f"Account {user_id} deleted."

agent = Agent(
    "support",
    config=OpenAIConfig(model="gpt-4o-mini"),
    tools=[delete_account],
    hitl_hook=lambda event: input(event.content),
)
```

When the agent calls `delete_account`, the user sees:

```
Agent tries to call tool:
`delete_account`, {"user_id": "abc-123"}
Please approve or deny this request.
Y/N?
```

Typing **y** lets the tool run. Anything else denies it; the agent receives the denial message and can adjust.

`approval_required()` calls `context.input()` under the hood, so it **also requires a `hitl_hook`** — without one you'll get a runtime error.

### Custom prompt

```python
@tool(middleware=[approval_required(
    message="⚠️ Run `{tool_name}` with {tool_arguments}? (y/n)",
    denied_message="Operation blocked by user.",
)])
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to the given address."""
    ...
```

`{tool_name}` and `{tool_arguments}` are interpolated.

## Pairing both patterns

For a tool that both gathers input mid-run and requires approval:

```python
@tool(middleware=[approval_required()])
async def schedule_report(name: str, context: Context) -> str:
    """Schedule a report — asks the user for the cadence, then runs after approval."""
    cadence = await context.input("How often? (daily / weekly / monthly)")
    return f"Scheduled '{name}' on {cadence} cadence."
```

The approval middleware runs first (outermost). Once approved, the tool body executes and `context.input()` triggers a second human interaction.

## Going deeper

- Source docs: `website/docs/beta/context/human_in_the_loop.mdx` (`context.input`, `hitl_hook`), `website/docs/beta/tools/approval_required.mdx` (`approval_required` middleware).
- Tool middleware in general — `website/docs/beta/tools/tool_middleware.mdx`. See also `ag2-middleware` for agent-wide HITL interception via `BaseMiddleware.on_human_input()`.
- HITL hooks support dependency injection identically to tools — see `../ag2-add-custom-tool/references/dependency_injection.md`.

## Common pitfalls

- **`approval_required()` without a `hitl_hook`** — the middleware calls `context.input()`, so the agent needs a hook. You'll see `HumanInputNotProvidedError` otherwise.
- **Forgetting to handle the denial path** — `context.input()` returns whatever the hook returns. If you only branch on "yes", any other answer (including silence/default) lets the operation continue. Always validate.
- **Sync `input()` in an async UI** — `input()` blocks the event loop. Use an async hook (`async def`) and an async input collector (web socket, message queue) for any non-CLI app.
- **No timeout** — `context.input(prompt)` can wait forever. Pass `timeout=60.0` (seconds) for any production path.
- **Decorator hook overrides constructor hook silently** — if you set both, the decorator wins. Pick one place.
- **Expecting `HumanMessage` to flow into the conversation history automatically** — it does for the requesting tool's return value, but mid-run inputs collected via `ctx.input()` are not separate user turns. They live in the tool's scope.
