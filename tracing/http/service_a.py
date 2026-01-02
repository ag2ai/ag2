import httpx
import uvicorn
from fastapi import FastAPI
from instrumentation import setup_instrumentation

app = FastAPI()

# Setup OTel instrumentation
setup_instrumentation("service-a", app)


@app.get("/")
async def root():
    async with httpx.AsyncClient() as client:
        # Call Service B
        response = await client.get("http://localhost:8001/data")
    return {"message": "Hello from Service A", "service_b_response": response.json()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
