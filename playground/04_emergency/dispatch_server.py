#!/usr/bin/env python3
"""Dispatch server — always-on 911 dispatch center.

Long-running server that accepts emergency calls via HTTP,
connects to medical/police/fire servers, and coordinates response.

Usage:
    python playground/04_emergency/dispatch_server.py
    python playground/04_emergency/dispatch_server.py --model gemini-3-flash-preview
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aiohttp import web

from _shared import BOLD, DIM, GREEN, MAGENTA, PORTS, RED, RESET, YELLOW, make_dispatch, subscribe_hub_logging

from autogen.beta.network import Hub, TelemetryPlugin


async def main() -> None:
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "gemini-3.1-pro-preview"
    port = PORTS["dispatch"]

    hub = Hub(max_delegation_depth=4, plugins=[TelemetryPlugin()])
    await hub.register(
        make_dispatch(model),
        capabilities=["dispatch", "triage", "coordination"],
        description="911 dispatcher - coordinates emergency response",
    )
    subscribe_hub_logging(hub, label="DISPATCH")

    # Header
    print()
    print(f"  {YELLOW}{BOLD}{'=' * 60}{RESET}")
    print(f"  {YELLOW}{BOLD}  911 DISPATCH CENTER{RESET}")
    print(f"  {YELLOW}{BOLD}{'=' * 60}{RESET}")
    print(f"  {DIM}Port: {port}  |  Model: {model}{RESET}")
    print()

    # Auto-connect to medical + police
    connections: dict[str, str] = {}
    for name, srv_port in [("medical", PORTS["medical"]), ("police", PORTS["police"])]:
        endpoint = f"http://localhost:{srv_port}"
        try:
            agents = await hub.connect(endpoint)
            connections[name] = endpoint
            print(f"  {GREEN}Connected to {name} ({srv_port}): {', '.join(agents)}{RESET}")
        except Exception as e:
            print(f"  {YELLOW}Warning: {name} server not available — {e}{RESET}")
            print(f"  {DIM}  Start it: python playground/04_emergency/{name}_server.py{RESET}")

    # Track incidents
    incident_counter = 0

    # --- HTTP handlers ---

    async def handle_emergency(request: web.Request) -> web.Response:
        nonlocal incident_counter
        data = await request.json()
        message = data.get("message", "")
        if not message:
            return web.json_response({"status": "error", "reason": "Missing 'message'"}, status=400)

        incident_counter += 1
        iid = f"INC-{incident_counter:04d}"

        print()
        print(f"  {RED}{BOLD}{'─' * 60}{RESET}")
        print(f"  {RED}{BOLD}INCOMING [{iid}]{RESET}")
        print(f"  {RED}{BOLD}{'─' * 60}{RESET}")
        preview = message[:200].replace("\n", " ")
        print(f"  {DIM}{preview}{'...' if len(message) > 200 else ''}{RESET}")
        print()

        t0 = time.monotonic()
        try:
            reply = await hub.ask("dispatch", message)
            elapsed = time.monotonic() - t0
            result = reply.body or ""

            print()
            print(f"  {GREEN}{BOLD}RESOLVED [{iid}]{RESET}  {DIM}({elapsed:.1f}s){RESET}")
            print()

            return web.json_response({
                "status": "ok",
                "incident_id": iid,
                "result": result,
                "elapsed": round(elapsed, 1),
            })
        except Exception as e:
            return web.json_response(
                {"status": "error", "incident_id": iid, "reason": str(e)}, status=500,
            )

    async def handle_connect(request: web.Request) -> web.Response:
        data = await request.json()
        endpoint = data.get("endpoint", "")
        if not endpoint:
            return web.json_response({"status": "error", "reason": "Missing 'endpoint'"}, status=400)
        try:
            agents = await hub.connect(endpoint)
            label = data.get("label", endpoint)
            connections[label] = endpoint

            print()
            print(f"  {GREEN}{BOLD}NEW SERVICE CONNECTED:{RESET} {endpoint}")
            print(f"  {GREEN}Agents: {', '.join(agents)}{RESET}")
            print()

            return web.json_response({"status": "ok", "agents": agents})
        except Exception as e:
            return web.json_response({"status": "error", "reason": str(e)}, status=500)

    async def handle_status(request: web.Request) -> web.Response:
        agents = await hub.discover()
        return web.json_response({
            "status": "running",
            "agents": [
                {"name": a.name, "capabilities": a.capabilities, "description": a.description}
                for a in agents
            ],
            "connections": connections,
            "incident_count": incident_counter,
        })

    # --- Start server ---

    app = web.Application()
    app.router.add_post("/emergency", handle_emergency)
    app.router.add_post("/connect", handle_connect)
    app.router.add_get("/status", handle_status)

    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    # Show ready state
    print()
    agents = await hub.discover()
    print(f"  {DIM}Registered actors:{RESET}")
    for a in agents:
        caps = ", ".join(a.capabilities)
        print(f"    {a.name} [{caps}]")
    print()
    print(f"  {DIM}Endpoints:{RESET}")
    print(f"    POST /emergency  — send an emergency call")
    print(f"    POST /connect    — connect a new service")
    print(f"    GET  /status     — show connected services")
    print()
    print(f"  {YELLOW}Serving on http://0.0.0.0:{port} — Ctrl+C to stop{RESET}")
    print(f"  {GREEN}Ready — waiting for emergency calls...{RESET}")
    print()

    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
        await hub.close()


if __name__ == "__main__":
    asyncio.run(main())
