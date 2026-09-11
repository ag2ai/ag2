"""Serve an MCP App: an agent plus an interactive document a host renders.

Run it over stdio and point an MCP Apps host (or the MCP Inspector) at it::

    python examples/mcp/server_apps.py

The document is static and registered as a resource — the host reads it *in
parallel with* the call, so it cannot be built from the call's arguments. Each
call's data arrives inside the frame as the result's ``structuredContent``.
Contrast ``server_ui.py``, which builds HTML from arguments inside the handler;
the two mechanisms are alternatives.
"""

import asyncio
from dataclasses import dataclass

from ag2 import Agent
from ag2.config import AnthropicConfig
from ag2.mcp import MCPApp, MCPServer

CATALOG = {"42": ("Espresso cup", 12), "43": ("Pour-over kettle", 48)}

# One document, two tools. `ag2ui` is the injected runtime: it performs the
# handshake a host requires before it will accept any message from the frame.
CARD = """<!doctype html>
<html>
  <head>
    <style>
      body { font-family: system-ui, sans-serif; margin: 0; padding: 20px; }
      #name { font-size: 20px; font-weight: 600; }
      button { margin-top: 12px; padding: 8px 14px; }
    </style>
  </head>
  <body>
    <div id="name">Loading…</div>
    <div id="price"></div>
    <button id="buy" disabled>Add to cart</button>
    <div id="status"></div>
    <script>
      var current = null;

      ag2ui.onToolResult("show_item", function (result) {
        current = result.structuredContent;
        document.getElementById("name").textContent = current.name;
        document.getElementById("price").textContent = "$" + current.price;
        document.getElementById("buy").disabled = false;
      });

      ag2ui.onToolResult("add_to_cart", function (result) {
        document.getElementById("status").textContent = result.structuredContent.status;
      });

      document.getElementById("buy").addEventListener("click", function () {
        ag2ui.callTool("add_to_cart", { item_id: current.item_id });
      });
    </script>
  </body>
</html>
"""

shop = MCPApp(
    "ui://shop/card",
    CARD,
    title="Product card",
    description="A product card that can add its item to the cart.",
    prefers_border=True,
)


@dataclass
class Item:
    """The card's payload — and, via its schema, the tool's ``outputSchema``."""

    item_id: str
    name: str
    price: int

    def __str__(self) -> str:
        # What a text-only client and the model see; the card reads the dump.
        return f"{self.name} — ${self.price}"


@dataclass
class CartLine:
    status: str

    def __str__(self) -> str:
        return self.status


@shop.tool
async def show_item(item_id: str) -> Item:
    """Show the product card for an item."""
    name, price = CATALOG.get(item_id, ("Unknown item", 0))
    return Item(item_id=item_id, name=name, price=price)


@shop.tool(visibility=["app"])
async def add_to_cart(item_id: str) -> CartLine:
    """Add an item to the cart. Called by the card's button, not by the model."""
    name, _ = CATALOG.get(item_id, ("Unknown item", 0))
    return CartLine(status=f"Added {name} to your cart ✓")


agent = Agent(
    name="shopkeeper",
    prompt="You are a concise shop assistant.",
    config=AnthropicConfig(model="claude-sonnet-5"),
)

server = MCPServer(agent, apps=[shop])


if __name__ == "__main__":
    asyncio.run(server.run_stdio())
