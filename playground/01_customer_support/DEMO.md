# Customer Support Agent

**Category 1: Single Actor + Tools** — the simplest AG2 network demo.

One actor with five domain tools, no network, no scheduling, no observers. This is the "hello world" of the framework: give an actor tools and a system prompt, then `await actor.ask(message)`.

## What it shows

- Creating tools with the `@tool` decorator
- Wiring tools into an `Actor`
- Subscribing to the `MemoryStream` to observe tool calls and model responses in real time
- The full request lifecycle: user message -> model reasoning -> tool calls -> tool results -> final response

## Prerequisites

```bash
# Activate the beta virtual environment
source .venv-beta/bin/activate

# Set your Gemini API key
export GOOGLE_API_KEY="your-key-here"  # pragma: allowlist secret
```

## Running the demo

```bash
cd playground/01_customer_support

# Scenario 1 (default): Order status inquiry
python main.py

# Scenario 2: Refund request for a defective product
python main.py --scenario 2

# Scenario 3: Product availability + return policy
python main.py --scenario 3

# Custom message
python main.py "Can I cancel order ORD-9102?"

# Use a different model
python main.py --model gemini-3-flash-preview --scenario 2
```

## What to watch for

- **Tool call logging** — yellow lines show each tool the model invokes and its arguments. Watch how the model chains tools (e.g., looks up an order before processing a refund).
- **Multi-tool scenarios** — scenario 3 triggers both `check_inventory` and `search_faq` in a single conversation turn.
- **Realistic mock data** — orders have tracking numbers, dates, and status notes; inventory includes stock counts and shipping estimates.
- **Final response quality** — the model synthesizes tool results into a coherent, customer-friendly reply with specific details (IDs, dates, amounts).

## Tools

| Tool | Purpose |
|---|---|
| `search_faq` | Searches a hardcoded FAQ knowledge base by keyword matching |
| `lookup_order` | Returns order details — status, tracking, dates, notes |
| `check_inventory` | Returns stock levels, warehouse, and shipping estimates |
| `process_refund` | Processes a refund and returns a confirmation with refund ID |
| `escalate_ticket` | Creates a support escalation ticket with priority |
