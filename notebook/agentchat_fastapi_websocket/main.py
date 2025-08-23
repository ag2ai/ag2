import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import nest_asyncio
import uvicorn
import ws
from dependencies import cleanup_managers, init_managers
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware import Middleware
from fastapi.responses import HTMLResponse

# from routers import apis
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse

# Initialize application
load_dotenv()

middleware = [
    Middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application
    Handles startup and shutdown events
    """
    nest_asyncio.apply()  # 🧠 Patch the loop here
    os.environ["HOME"] = os.path.expanduser("~")
    try:
        await init_managers()
    except Exception:
        raise

    yield  # App is running and handling requests here

    # Shutdown: Close database connections and cleanup
    try:
        await cleanup_managers()
        print("✅ Managers cleaned up successfully")

    except Exception:
        raise


app = FastAPI(title="Autogen WebSocket", redoc_url=None, lifespan=lifespan, middleware=middleware)

app.include_router(
    ws.router,
    prefix="/ws",
    tags=["websocket"],
    responses={404: {"description": "Not found"}},
)


@app.get(path="autogen", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url="redoc")


# WEBSOCKET Swagger Docs


@app.get("/", response_class=HTMLResponse)
async def serve_html():
    html_content = Path("templates/index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "ok", "message": "Service is running"}


@app.post("/chats/new_chat")
async def new_chat():
    """
    Health check endpoint
    """
    return {"status": "ok", "chat_id": str(uuid.uuid4())}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
