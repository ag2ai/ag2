# AG2 CLI Test Playground

Demo agents for testing CLI commands. Requires `GOOGLE_GEMINI_API_KEY` in environment.

Load env first: `source /path/to/ag2/.env`

## Files

| File | Purpose |
|------|---------|
| `single_agent.py` | Single agent with `agent` variable |
| `main_agent.py` | Agent with `main()` entry point |
| `team.py` | Two-agent team with `agents` list |
| `agent_concise.py` | Concise style agent (for arena) |
| `agent_detailed.py` | Detailed style agent (for arena) |
| `agent_coder.py` | Coding-focused agent (for arena) |
| `eval_cases.yaml` | Basic eval suite for `ag2 test eval` |
| `arena_eval.yaml` | Arena-specific eval suite (5 cases) |
| `sample_api.yaml` | OpenAPI spec for `ag2 proxy openapi` |
| `scripts/` | Shell scripts for `ag2 proxy scripts` |

## Commands to try

### ag2 run
```bash
# Run with main() entry point
ag2 run test_playground/main_agent.py -m "What is the capital of Japan?"

# Run with discovered agent variable
ag2 run test_playground/single_agent.py -m "Explain quantum computing in one sentence."

# JSON output (for piping)
ag2 run test_playground/main_agent.py -m "Hello" --json

# With max turns
ag2 run test_playground/single_agent.py -m "Write a haiku" --max-turns 3
```

### ag2 chat
```bash
# Interactive chat with an agent
ag2 chat test_playground/single_agent.py

# Ad-hoc chat (no file needed)
ag2 chat --model gemini-3-flash-preview --system "You are a pirate."
```

### ag2 test eval
```bash
# Run eval suite
ag2 test eval test_playground/main_agent.py --eval test_playground/eval_cases.yaml

# JSON output for CI
ag2 test eval test_playground/main_agent.py --eval test_playground/eval_cases.yaml --output json
```

### ag2 serve
```bash
# REST API
ag2 serve test_playground/single_agent.py --port 8000
# Then: curl -X POST http://localhost:8000/chat -H 'Content-Type: application/json' -d '{"message":"Hello"}'

# MCP server (for Claude Desktop, Cursor, etc.)
ag2 serve test_playground/single_agent.py --protocol mcp --port 8001

# A2A protocol
ag2 serve test_playground/single_agent.py --protocol a2a --port 8002

# With ngrok tunnel
ag2 serve test_playground/single_agent.py --ngrok
```

### ag2 arena
```bash
# Head-to-head: concise vs detailed agent
ag2 arena compare test_playground/agent_concise.py test_playground/agent_detailed.py \
  --eval test_playground/arena_eval.yaml

# Tournament: all 3 agents compete
ag2 arena compare test_playground/agent_concise.py test_playground/agent_detailed.py \
  test_playground/agent_coder.py --eval test_playground/arena_eval.yaml

# Dry run (estimate cost)
ag2 arena compare test_playground/agent_concise.py test_playground/agent_detailed.py \
  --eval test_playground/arena_eval.yaml --dry-run

# JSON output for further analysis
ag2 arena compare test_playground/agent_concise.py test_playground/agent_detailed.py \
  --eval test_playground/arena_eval.yaml --output json

# Interactive head-to-head (human judge, anonymous A/B)
ag2 arena interactive test_playground/agent_concise.py test_playground/agent_detailed.py

# View ELO leaderboard (accumulates across sessions)
ag2 arena leaderboard
```

### ag2 proxy
```bash
# Wrap git CLI as AG2 tools
ag2 proxy cli git --subcommands "status,log,diff" --preview
ag2 proxy cli git --subcommands "status,log,diff" --output test_playground/git_tools.py

# Wrap an OpenAPI spec as AG2 tools
ag2 proxy openapi test_playground/sample_api.yaml --preview
ag2 proxy openapi test_playground/sample_api.yaml --output test_playground/api_tools.py

# Filter to specific endpoints
ag2 proxy openapi test_playground/sample_api.yaml --endpoints "list_tasks,create_task" --preview

# Wrap Python module functions
ag2 proxy module json --functions "dumps,loads" --preview
ag2 proxy module os.path --functions "exists,join,basename" --preview

# Wrap shell scripts as AG2 tools
ag2 proxy scripts test_playground/scripts/ --preview
ag2 proxy scripts test_playground/scripts/ --output test_playground/script_tools.py
```

### ag2 replay
```bash
# List recorded sessions (after running with --record)
ag2 replay list

# Replay a session (use session ID from the list)
ag2 replay show <session_id>

# Interactive step-through (Enter=next, p=prev, g N=goto, q=quit)
ag2 replay step <session_id>

# Export as markdown, JSON, or HTML
ag2 replay export <session_id> --format md
ag2 replay export <session_id> --format json --output session.json
ag2 replay export <session_id> --format html --output session.html

# Compare two sessions side-by-side
ag2 replay compare <session_id_1> <session_id_2>

# Branch from a specific turn and re-run
ag2 replay branch <session_id> --at 3 -m "Try a different approach"

# Clean up
ag2 replay delete <session_id>
ag2 replay clear
```
