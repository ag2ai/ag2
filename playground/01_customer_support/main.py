"""Customer Support Agent — Single Actor + Tools demo.

Category 1: The simplest AG2 network demo. One actor with domain tools,
no network, no scheduling, no observers.

Usage:
    python main.py                          # default: order status inquiry
    python main.py --scenario 2             # refund request
    python main.py --scenario 3             # product availability
    python main.py "Where is my package?"   # custom message
    python main.py --model gemini-3-flash-preview # specify model
"""

from __future__ import annotations

import argparse
import asyncio
import random
import string
from datetime import datetime, timedelta

from autogen.beta.config.gemini import GeminiConfig
from autogen.beta.events import ModelResponse, ToolCallEvent, ToolResultEvent
from autogen.beta.network import Actor
from autogen.beta.stream import MemoryStream
from autogen.beta.tools.final import tool

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
CYAN = "\033[36m"
RED = "\033[31m"
MAGENTA = "\033[35m"
WHITE = "\033[97m"


def _ts() -> str:
    """Return a compact HH:MM:SS timestamp."""
    return datetime.now().strftime("%H:%M:%S")


def _header(title: str, scenario: int, message: str) -> None:
    width = 64
    print()
    print(f"{CYAN}{BOLD}{'=' * width}{RESET}")
    print(f"{CYAN}{BOLD}  Customer Support Agent — {title}{RESET}")
    print(f"{CYAN}  Scenario {scenario} | Single Actor + Tools{RESET}")
    print(f"{CYAN}{BOLD}{'=' * width}{RESET}")
    print(f"\n{DIM}[{_ts()}]{RESET} {WHITE}Customer message:{RESET}")
    print(f"  {message}\n")
    print(f"{DIM}{'─' * width}{RESET}")


# ---------------------------------------------------------------------------
# FAQ knowledge base
# ---------------------------------------------------------------------------
FAQ_DB = [
    {
        "question": "What is your return policy?",
        "answer": (
            "We offer a 30-day return policy for most items. Electronics have a "
            "15-day return window. Items must be in original packaging and unused. "
            "Refunds are processed within 5-7 business days after we receive the return."
        ),
        "tags": ["return", "refund", "policy", "exchange"],
    },
    {
        "question": "How long does shipping take?",
        "answer": (
            "Standard shipping: 5-7 business days. Express shipping: 2-3 business days. "
            "Same-day delivery available in select metro areas for orders placed before 2 PM. "
            "Free shipping on orders over $50."
        ),
        "tags": ["shipping", "delivery", "time", "free"],
    },
    {
        "question": "How do I track my order?",
        "answer": (
            "You can track your order by logging into your account and visiting "
            "'My Orders', or use the tracking number sent to your email. "
            "Tracking updates may take 24 hours after shipment."
        ),
        "tags": ["track", "order", "status", "tracking"],
    },
    {
        "question": "What payment methods do you accept?",
        "answer": (
            "We accept Visa, MasterCard, American Express, PayPal, Apple Pay, "
            "Google Pay, and Shop Pay. Afterpay and Klarna available for orders "
            "over $35."
        ),
        "tags": ["payment", "pay", "credit", "card", "paypal"],
    },
    {
        "question": "Do you offer warranty on electronics?",
        "answer": (
            "All electronics come with a 1-year manufacturer warranty. Extended "
            "warranty (2 additional years) is available for purchase at checkout. "
            "Warranty covers manufacturing defects but not accidental damage."
        ),
        "tags": ["warranty", "electronics", "defect", "protection"],
    },
    {
        "question": "How do I cancel an order?",
        "answer": (
            "Orders can be cancelled within 1 hour of placement. Go to 'My Orders' "
            "and click 'Cancel Order'. If the order has already shipped, you'll need "
            "to initiate a return instead."
        ),
        "tags": ["cancel", "order", "cancellation"],
    },
]


# ---------------------------------------------------------------------------
# Mock order database
# ---------------------------------------------------------------------------
ORDER_DB: dict[str, dict] = {
    "ORD-7821": {
        "customer": "Alex Thompson",
        "product": "UltraBook Pro 16-inch Laptop",
        "price": 1299.99,
        "status": "processing",
        "ordered_date": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
        "estimated_delivery": (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d"),
        "tracking": None,
        "notes": "Awaiting stock from warehouse. Expected to ship within 24 hours.",
    },
    "ORD-6234": {
        "customer": "Jordan Rivera",
        "product": "ProSound Wireless Headphones XR-500",
        "price": 149.99,
        "status": "delivered",
        "ordered_date": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
        "delivered_date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
        "tracking": "FX-98234571-US",
        "notes": "Delivered to front door.",
    },
    "ORD-9102": {
        "customer": "Sam Chen",
        "product": "Ergonomic Office Chair",
        "price": 349.00,
        "status": "shipped",
        "ordered_date": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"),
        "estimated_delivery": (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"),
        "tracking": "FX-77412983-US",
        "notes": "In transit — last scan: regional distribution center.",
    },
}


# ---------------------------------------------------------------------------
# Mock inventory database
# ---------------------------------------------------------------------------
INVENTORY_DB: dict[str, dict] = {
    "ultrabook pro 16": {
        "name": "UltraBook Pro 16-inch Laptop",
        "sku": "UBP-16-2026",
        "price": 1299.99,
        "in_stock": 23,
        "warehouse": "West Coast DC",
        "next_restock": None,
        "ships_within": "1-2 business days",
    },
    "proSound wireless headphones": {
        "name": "ProSound Wireless Headphones XR-500",
        "sku": "PSW-XR500",
        "price": 149.99,
        "in_stock": 156,
        "warehouse": "East Coast DC",
        "next_restock": None,
        "ships_within": "same day",
    },
    "ergonomic office chair": {
        "name": "Ergonomic Office Chair — Mesh Pro",
        "sku": "EOC-MP-BLK",
        "price": 349.00,
        "in_stock": 0,
        "warehouse": "Central DC",
        "next_restock": (datetime.now() + timedelta(days=12)).strftime("%Y-%m-%d"),
        "ships_within": "ships after restock",
    },
    "mechanical keyboard": {
        "name": "MechType RGB Mechanical Keyboard",
        "sku": "MTK-RGB-01",
        "price": 89.99,
        "in_stock": 312,
        "warehouse": "West Coast DC",
        "next_restock": None,
        "ships_within": "same day",
    },
    "4k monitor": {
        "name": "ClearView 4K 27-inch Monitor",
        "sku": "CV-4K27",
        "price": 429.99,
        "in_stock": 8,
        "warehouse": "East Coast DC",
        "next_restock": None,
        "ships_within": "1-2 business days",
    },
}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
async def search_faq(query: str) -> str:
    """Search the FAQ knowledge base for answers to common customer questions.

    Args:
        query: The customer's question or keywords to search for.
    """
    query_lower = query.lower()
    matches = []

    for entry in FAQ_DB:
        score = 0
        for tag in entry["tags"]:
            if tag in query_lower:
                score += 2
        # Also check if query words appear in the question text
        for word in query_lower.split():
            if len(word) > 3 and word in entry["question"].lower():
                score += 1
        if score > 0:
            matches.append((score, entry))

    matches.sort(key=lambda x: x[0], reverse=True)

    if not matches:
        return "No FAQ entries found matching the query. Consider escalating to a human agent for this question."

    results = []
    for _, entry in matches[:3]:
        results.append(f"Q: {entry['question']}\nA: {entry['answer']}")

    return "\n\n---\n\n".join(results)


@tool
async def lookup_order(order_id: str) -> str:
    """Look up the status and details of a customer order by order ID.

    Args:
        order_id: The order identifier (e.g., ORD-7821).
    """
    order_id = order_id.upper().strip()
    order = ORDER_DB.get(order_id)

    if not order:
        return f"Order {order_id} not found. Please verify the order ID and try again."

    lines = [
        f"Order: {order_id}",
        f"Customer: {order['customer']}",
        f"Product: {order['product']}",
        f"Price: ${order['price']:.2f}",
        f"Status: {order['status'].upper()}",
        f"Ordered: {order['ordered_date']}",
    ]

    if order.get("tracking"):
        lines.append(f"Tracking: {order['tracking']}")
    if order.get("estimated_delivery"):
        lines.append(f"Est. Delivery: {order['estimated_delivery']}")
    if order.get("delivered_date"):
        lines.append(f"Delivered: {order['delivered_date']}")
    if order.get("notes"):
        lines.append(f"Notes: {order['notes']}")

    return "\n".join(lines)


@tool
async def check_inventory(product_name: str) -> str:
    """Check inventory and stock levels for a product.

    Args:
        product_name: The name or keyword of the product to check.
    """
    product_lower = product_name.lower()

    # Try exact key match first, then fuzzy match
    best_match = None
    best_score = 0

    for key, product in INVENTORY_DB.items():
        score = 0
        for word in product_lower.split():
            if len(word) > 2 and word in key.lower():
                score += 1
            if len(word) > 2 and word in product["name"].lower():
                score += 1
        if score > best_score:
            best_score = score
            best_match = product

    if not best_match:
        return f"No product matching '{product_name}' found in inventory."

    stock_status = "IN STOCK" if best_match["in_stock"] > 0 else "OUT OF STOCK"
    lines = [
        f"Product: {best_match['name']}",
        f"SKU: {best_match['sku']}",
        f"Price: ${best_match['price']:.2f}",
        f"Status: {stock_status} ({best_match['in_stock']} units)",
        f"Warehouse: {best_match['warehouse']}",
        f"Ships Within: {best_match['ships_within']}",
    ]

    if best_match.get("next_restock"):
        lines.append(f"Next Restock: {best_match['next_restock']}")

    return "\n".join(lines)


@tool
async def process_refund(order_id: str, reason: str) -> str:
    """Process a refund for a customer order.

    Args:
        order_id: The order identifier to refund.
        reason: The reason for the refund.
    """
    order_id = order_id.upper().strip()
    order = ORDER_DB.get(order_id)

    if not order:
        return f"Cannot process refund: Order {order_id} not found."

    if order["status"] not in ("delivered", "shipped"):
        return (
            f"Cannot process refund: Order {order_id} has status '{order['status']}'. "
            f"Only delivered or shipped orders are eligible for refund."
        )

    refund_id = "RFD-" + "".join(random.choices(string.digits, k=6))
    refund_amount = order["price"]

    return (
        f"Refund processed successfully.\n"
        f"Refund ID: {refund_id}\n"
        f"Order: {order_id}\n"
        f"Amount: ${refund_amount:.2f}\n"
        f"Reason: {reason}\n"
        f"Status: APPROVED — refund will appear in 5-7 business days.\n"
        f"A confirmation email has been sent to the customer."
    )


@tool
async def escalate_ticket(issue_summary: str, priority: str = "medium") -> str:
    """Escalate an issue to the human support team by creating a ticket.

    Args:
        issue_summary: A concise summary of the issue for the support team.
        priority: Ticket priority — 'low', 'medium', 'high', or 'urgent'.
    """
    priority = priority.lower().strip()
    if priority not in ("low", "medium", "high", "urgent"):
        priority = "medium"

    ticket_id = "TKT-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    eta_map = {"low": "48 hours", "medium": "24 hours", "high": "4 hours", "urgent": "1 hour"}

    return (
        f"Escalation ticket created.\n"
        f"Ticket ID: {ticket_id}\n"
        f"Priority: {priority.upper()}\n"
        f"Summary: {issue_summary}\n"
        f"Expected Response: within {eta_map[priority]}\n"
        f"A support specialist will follow up via email."
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
SCENARIOS: dict[int, tuple[str, str]] = {
    1: (
        "Order Status Inquiry",
        "Hi, I ordered a laptop (order #ORD-7821) 5 days ago and it still "
        "says processing. Can you check what's going on?",
    ),
    2: (
        "Refund Request",
        "I received my wireless headphones (order #ORD-6234) yesterday but "
        "they won't connect to any device. I've tried everything. I want a refund.",
    ),
    3: (
        "Product Availability",
        "Do you have the UltraBook Pro 16 in stock? I need it delivered by "
        "Friday. Also, what's your return policy for electronics?",
    ),
}


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a friendly and professional customer support agent for TechMart, \
an online electronics and accessories store.

Guidelines:
- Greet the customer warmly and acknowledge their concern.
- Use the available tools to look up real data before answering. Never guess \
  at order statuses, inventory, or policies.
- When a customer reports a defective product, first look up the order to \
  confirm the details, then offer a refund or replacement.
- If inventory is checked and the product is in stock, include the shipping \
  estimate in your response.
- Cite specific details (order IDs, tracking numbers, dates, prices) in \
  your responses.
- If you cannot resolve an issue, escalate it with an appropriate priority.
- Keep responses concise but thorough. Use a helpful, empathetic tone.
- End with a clear summary of actions taken and next steps for the customer.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Customer Support Agent — AG2 Single Actor + Tools Demo",
    )
    parser.add_argument(
        "message",
        nargs="?",
        default=None,
        help="Custom customer message (overrides scenario)",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Predefined scenario (default: 1 — order status inquiry)",
    )
    parser.add_argument(
        "--model",
        default="gemini-3-flash-preview",
        help="Gemini model name (default: gemini-3-flash-preview)",
    )
    args = parser.parse_args()

    # Resolve the message
    if args.message:
        scenario_title = "Custom Query"
        message = args.message
    else:
        scenario_title, message = SCENARIOS[args.scenario]

    _header(scenario_title, args.scenario, message)

    # Create the actor
    actor = Actor(
        name="support",
        prompt=SYSTEM_PROMPT,
        config=GeminiConfig(model=args.model, temperature=0.3),
        tools=[search_faq, lookup_order, check_inventory, process_refund, escalate_ticket],
    )

    # Set up the stream with event logging
    stream = MemoryStream()

    async def _log_event(event: ToolCallEvent | ToolResultEvent | ModelResponse) -> None:
        if isinstance(event, ToolCallEvent):
            print(
                f"  {DIM}[{_ts()}]{RESET} {YELLOW}TOOL  "
                f"{event.name}({event.serialized_arguments}){RESET}"
            )
        elif isinstance(event, ToolResultEvent):
            content = event.content.replace("\n", " ")
            truncated = content[:120] + "..." if len(content) > 120 else content
            print(f"  {DIM}[{_ts()}]{RESET} {MAGENTA}  ->  {truncated}{RESET}")
        elif isinstance(event, ModelResponse) and event.content:
            preview = event.content[:200].replace("\n", " ")
            print(f"  {DIM}[{_ts()}]{RESET} {GREEN}MODEL {preview}...{RESET}")

    stream.where(ToolCallEvent).subscribe(_log_event)
    stream.where(ToolResultEvent).subscribe(_log_event)
    stream.where(ModelResponse).subscribe(_log_event)

    # Ask the actor
    print(f"{DIM}[{_ts()}]{RESET} {WHITE}Sending to actor...{RESET}\n")

    reply = await actor.ask(message, stream=stream)

    # Print the final response
    print(f"\n{DIM}{'─' * 64}{RESET}")
    print(f"{DIM}[{_ts()}]{RESET} {GREEN}{BOLD}Final Response:{RESET}\n")
    print(f"{GREEN}{reply.body}{RESET}")
    print(f"\n{DIM}{'─' * 64}{RESET}")
    print(f"{DIM}Done.{RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())
