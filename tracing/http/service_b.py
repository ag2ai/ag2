import asyncio

import uvicorn
from fastapi import FastAPI
from instrumentation import setup_instrumentation

app = FastAPI()

# Setup OTel instrumentation
setup_instrumentation("service-b", app)


@app.get("/data")
async def get_data():
    # Simulate some work
    await asyncio.sleep(0.1)
    return {"data": "Important data from Service B"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
