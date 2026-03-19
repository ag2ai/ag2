# ag2 replay

> Replay, debug, and branch agent conversations.

## Problem

When a multi-agent conversation goes wrong, you can't step through it
to find where. You can't re-run it with different parameters to see what
would have changed. You can't compare two runs side by side.

This is like debugging without a debugger — printf-only.

## Commands

```bash
# List recorded sessions
ag2 replay --list

# Replay a session (uses cached LLM responses for determinism)
ag2 replay session_abc123

# Replay with a different model
ag2 replay session_abc123 --model claude-sonnet-4-6

# Interactive step-through mode
ag2 replay session_abc123 --step

# Branch from a specific turn
ag2 replay session_abc123 --branch-at 5 --message "Try a different approach"

# Compare two sessions side by side
ag2 replay --compare session_abc123 session_def456

# Export a session for sharing
ag2 replay session_abc123 --export transcript.md
ag2 replay session_abc123 --export transcript.html
```

## Session Recording

Sessions are automatically recorded by `ag2 run` and `ag2 chat`. Each
session captures:

```json
{
  "id": "session_abc123",
  "timestamp": "2026-03-18T10:30:00Z",
  "agent_file": "my_team.py",
  "agents": ["researcher", "critic", "writer"],
  "model": "gpt-4o",
  "turns": [
    {
      "turn": 1,
      "speaker": "user",
      "message": "Research quantum computing trends",
      "timestamp": "2026-03-18T10:30:01Z"
    },
    {
      "turn": 2,
      "speaker": "researcher",
      "message": "I'll search for recent developments...",
      "tool_calls": [
        {
          "tool": "web_search",
          "args": {"query": "quantum computing 2026"},
          "result": "...",
          "duration_ms": 1200
        }
      ],
      "tokens": {"input": 450, "output": 280},
      "cost": 0.008,
      "timestamp": "2026-03-18T10:30:04Z",
      "llm_response_raw": { "...cached for deterministic replay..." }
    }
  ],
  "summary": {
    "total_turns": 8,
    "total_tokens": 4500,
    "total_cost": 0.12,
    "duration_seconds": 23.5,
    "outcome": "completed"
  }
}
```

Sessions are stored in `~/.ag2/sessions/` with automatic cleanup of
sessions older than 30 days.

## Interactive Step-Through (`--step`)

```
╭─ AG2 Replay ─ session_abc123 ──────────────────────╮
│ Agents: researcher, critic, writer                  │
│ Recorded: 2026-03-18 10:30 | 8 turns | $0.12       │
╰─────────────────────────────────────────────────────╯

Turn 1/8 — user
  "Research quantum computing trends"

  [Enter] next | [i]nspect state | [j]ump to turn | [e]dit & re-run | [q]uit
  >

Turn 2/8 — researcher
  "I'll search for recent developments in quantum computing..."

  🔧 web_search("quantum computing 2026")
     → 8 results (1.2s)

  Tokens: 450 in / 280 out | Cost: $0.008

  [Enter] next | [i]nspect state | [j]ump to turn | [e]dit & re-run | [q]uit
  > i

  ╭─ State at Turn 2 ────────────────────────────────╮
  │ Speaker: researcher                               │
  │ History: 2 messages                               │
  │ Context variables: {}                             │
  │ Tools called: web_search (1x)                     │
  │ Cumulative tokens: 730                            │
  │ Cumulative cost: $0.008                           │
  ╰───────────────────────────────────────────────────╯
```

## Branching (`--branch-at`)

Re-run from a specific turn with a modified message:

```bash
ag2 replay session_abc123 --branch-at 5 --message "Focus on error correction, not hardware"
```

This:
1. Replays turns 1-4 using cached responses (instant)
2. Injects the new message at turn 5
3. Continues with live LLM calls from turn 5 onward
4. Saves the branched run as a new session

This is like `git branch` for conversations — explore alternative paths
without losing the original.

## Comparison (`--compare`)

```bash
ag2 replay --compare session_abc session_def
```

```
╭─ Comparison ────────────────────────────────────────────────╮
│              session_abc          session_def                │
│ Model        gpt-4o              claude-sonnet-4-6          │
│ Turns        8                   6                          │
│ Cost         $0.12               $0.08                      │
│ Time         23.5s               18.2s                      │
│ Outcome      completed           completed                  │
╰─────────────────────────────────────────────────────────────╯

Turn-by-turn diff:
  Turn 2: Both called web_search — same results
  Turn 3: session_abc critic asked for more detail
           session_def went straight to writing
  Turn 4: session_abc researcher did a follow-up search
           session_def writer produced first draft
  ...

Key differences:
  - session_def was 2 turns shorter (writer was more proactive)
  - session_abc had more thorough research (2 search calls vs 1)
  - session_def cost 33% less
```

## Export Formats

### Markdown
```bash
ag2 replay session_abc123 --export transcript.md
```
Clean, readable conversation transcript with agent labels, tool calls,
and metadata.

### HTML
```bash
ag2 replay session_abc123 --export transcript.html
```
Styled HTML with collapsible tool calls, syntax-highlighted code blocks,
and a cost summary. Shareable as a standalone file.

### JSON
```bash
ag2 replay session_abc123 --export raw.json
```
Full session data for programmatic analysis.

## Implementation Notes

### Recording
Hook into AG2's event system to capture all messages, tool calls, and
LLM responses. The beta framework's `Stream` API is ideal for this.
For classic agents, use `ConversableAgent.register_hook()`.

### Deterministic Replay
Cache raw LLM responses (not just the text — the full API response
including tool calls). On replay, intercept the LLM client and return
cached responses. This makes replay instant and deterministic.

### Storage
Use SQLite for the session database (query by date, agent file, model).
Store large blobs (LLM responses) as compressed JSON. The `~/.ag2/sessions/`
directory should have a reasonable size limit with automatic cleanup.

## Dependencies
- `ag2` — required
- `rich` — already in CLI deps
- `sqlite3` — stdlib
