# ag2 watch

> Live monitoring dashboard for running agents — terminal UI.

## Problem

When agents are running, you want to see what's happening in real time:
which agent is speaking, what tools are being called, how many tokens
are being consumed, what the current cost is. Currently you get a wall
of unformatted text.

## Commands

```bash
# Run and monitor an agent with live dashboard
ag2 watch my_team.py --message "Analyze Q4 earnings"

# Monitor a running ag2 serve instance
ag2 watch --port 8000

# Monitor with specific detail level
ag2 watch my_team.py -m "..." --detail full     # everything
ag2 watch my_team.py -m "..." --detail compact  # summary only
```

## Dashboard Layout

```
╭─ AG2 Watch ─ research-team ─────────────────────────────────╮
│ Status: RUNNING | Elapsed: 12.3s | Turn 4/10                │
╰──────────────────────────────────────────────────────────────╯

╭─ Agent Activity ─────────────────────────────────────────────╮
│                                                              │
│  [user] → researcher → critic → writer → (active)           │
│                                                              │
│  ┌─ writer (speaking) ──────────────────────────────────┐    │
│  │ Based on the research and feedback, here is the      │    │
│  │ revised report on quantum computing trends...        │    │
│  │ ▌ (streaming)                                        │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
╰──────────────────────────────────────────────────────────────╯

╭─ Tools ──────────────────────────────────────────────────────╮
│  web_search     ✓ ✓ ✗ ✓   (3/4 success, avg 1.2s)          │
│  arxiv_fetch    ✓ ✓        (2/2 success, avg 2.1s)          │
╰──────────────────────────────────────────────────────────────╯

╭─ Metrics ────────────────────────────────────────────────────╮
│  Tokens    Input: 3,420   Output: 1,890   Total: 5,310      │
│  Cost      $0.08 (gpt-4o)                                   │
│  Turns     4/10                                              │
│  Agents    researcher: 2 turns | critic: 1 | writer: 1      │
╰──────────────────────────────────────────────────────────────╯
```

## Implementation Notes

### Rich Live Display
Use `rich.live.Live` for auto-refreshing terminal output. Compose the
layout using `rich.layout.Layout` with multiple panels.

### Event Stream
Hook into AG2's event system to capture real-time activity. The beta
framework's `Stream` API provides event-by-event updates. For classic
agents, use registered hooks.

### Metrics Collection
Track per-agent metrics using AG2's `token_count_utils` and the
`runtime_logging` module. Compute cost using model-specific pricing tables.

### Complexity Note
This is a Rich Live display, NOT a Textual TUI app. It's a one-way
display that updates in place — no interactive widgets needed. The
user can Ctrl+C to stop.
