import {
    CopilotRuntime,
    createCopilotEndpoint,
    InMemoryAgentRunner,
} from "@copilotkit/runtime/v2";
import { handle } from "hono/vercel";


import { HttpAgent } from "@ag-ui/client";
const agent = new HttpAgent({ url: "http://localhost:8008/chat" });

// import { A2AAgent } from "@ag-ui/a2a";
// import { A2AClient } from "@a2a-js/sdk/client";
// const a2aClient = new A2AClient("http://localhost:8008");
// const agent = new A2AAgent({ a2aClient });

const runtime = new CopilotRuntime({
    agents: {
        default: agent,
    },
    runner: new InMemoryAgentRunner(),
});

const app = createCopilotEndpoint({
    runtime,
    basePath: "/api/copilotkit",
});

export const GET = handle(app);
export const POST = handle(app);
