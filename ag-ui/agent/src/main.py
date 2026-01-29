from agent import ProverbsState, StateDeps, agent
from pydantic_ai.ui.ag_ui.app import AGUIApp

app = AGUIApp(
    agent=agent,
    deps=StateDeps(ProverbsState()),
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
