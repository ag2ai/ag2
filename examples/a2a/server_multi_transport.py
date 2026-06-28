import asyncio

import uvicorn

from ag2 import Agent
from ag2.a2a import A2AServer, build_card
from ag2.config import AnthropicConfig

agent = Agent(
    name="claude",
    config=AnthropicConfig(model="claude-sonnet-4-6"),
)


async def main() -> None:
    server = A2AServer(agent)

    card = build_card(
        agent,
        url="http://127.0.0.1:8000",
        transports=("jsonrpc", "rest", "grpc"),
        rest_url="http://127.0.0.1:8001",
        grpc_url="127.0.0.1:50051",
    )

    jsonrpc_app = server.build_jsonrpc(url="http://127.0.0.1:8000", card=card)
    rest_app = server.build_rest(url="http://127.0.0.1:8001", card=card)
    grpc_server = server.build_grpc(bind="127.0.0.1:50051", grpc_url="127.0.0.1:50051", card=card)

    await grpc_server.start()
    await asyncio.gather(
        uvicorn.Server(uvicorn.Config(jsonrpc_app, host="127.0.0.1", port=8000)).serve(),
        uvicorn.Server(uvicorn.Config(rest_app, host="127.0.0.1", port=8001)).serve(),
        grpc_server.wait_for_termination(),
    )


if __name__ == "__main__":
    asyncio.run(main())
