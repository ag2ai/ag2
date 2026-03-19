# AG2 CLI Test Playground

Demo agents for testing CLI commands. Requires `GEMINI_API_KEY` in environment.

Load env first: `source /path/to/ag2/.env`

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
