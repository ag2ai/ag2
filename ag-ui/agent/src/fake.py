from fastapi import FastAPI

from autogen.ag_ui import AGUIStream

from .ag import agent

app = FastAPI()

ag_ui = AGUIStream(agent)
app.mount("/chat", ag_ui.build_asgi())
