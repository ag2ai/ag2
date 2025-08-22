# api/ws.py
import asyncio
import json
from datetime import datetime, date
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from dependencies import get_websocket_manager
from managers.connection import WebSocketManager
from managers.groupchat import AgentChat
import os
from fastapi import HTTPException, Depends, status

router = APIRouter()

@router.websocket("/chat/{chat_id}")
async def run_websocket(
    websocket: WebSocket,
    chat_id: str,
    ws_manager: WebSocketManager = Depends(get_websocket_manager),
):
    """WebSocket endpoint for run communication"""
    chat = AgentChat(chat_id=chat_id)
    input_func = ws_manager._create_input_func(chat_id)
    chat.set_input_function(input_func)
    # Connect websocket
    connected = await ws_manager.connect(websocket, chat_id)
    if not connected:
        await websocket.close(code=4002, reason="Failed to establish connection")
        return

    try:
        while True:
            try:
                raw_message = await websocket.receive_text()
                message = json.loads(raw_message)

                if message.get("type") == "start":
                    # Handle start message
                    if message.get("task"):
                        asyncio.create_task(ws_manager.start_chat_stream(session_id=chat_id,initial_message=message,chat=chat))
                    else:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "error": "Invalid start message format",
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )

                elif message.get("type") == "stop":
                    reason = message.get("reason") or "User requested stop/cancellation"
                    await ws_manager.stop_session(chat_id, reason=reason)
                    # break

                elif message.get("type") == "ping":
                    await websocket.send_json(
                        {"type": "pong", "timestamp": datetime.utcnow().isoformat()}
                    )

                elif message.get("type") == "input_response":
                    # Handle input response from client
                    response = message.get("response")
                    if response is not None:
                        await ws_manager.handle_input_response(chat_id, response)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": "Invalid message format",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

    except WebSocketDisconnect:
        raise
    except Exception as e:
        raise
    finally:
        await ws_manager.disconnect(chat_id)
