# 🔌 AG2AI WebSocket Agentic Workflow UI & Backend

An interactive WebSocket-based UI and FastAPI backend for orchestrating agentic workflows using [AG2AI Autogen](https://github.com/ag2ai/ag2). This project demonstrates real-time, multi-agent collaboration for solving problems through a WebSocket-powered interface — ideal for tasks like data analysis, EDA, and more.

![App UI Screenshot](./docs/screenshot.png)

---

## 📘 How to Use the WebSocket UI

To learn how to interact with the UI step-by-step, check out the full guide:

➡️ [Usage Guide (UI Walkthrough)](./docs/USAGE_GUIDE.md)


---

## 📦 Overview

This repository includes:

- A clean **frontend UI** to interact with WebSockets visually — more intuitive than Postman or raw clients.
- A **FastAPI backend** that manages real-time WebSocket communication and coordinates multiple agents via AG2AI Autogen.
- A custom **orchestrator agent** (`agent_aligner`) that manages execution flow, ensuring orderly agent coordination.

---

## ✨ Features

### ✅ Frontend

- Interactive WebSocket client with formatted JSON display
- Message blocks styled for clarity and separation
- UUID-based client tracking
- Send/receive messages with live updates
- Manual message construction and quick templates

### ✅ Backend

- Built with **FastAPI** and **async WebSocket** handling
- Modular architecture using manager classes
- Custom `AgentChat` class for group chat orchestration
- Real-time message streaming to frontend
- Manual user input integration during live chat
- Environment-based configuration via `.env`

---

## 🧩 Problem & Solution

### ❌ The Problem We Faced

While working with WebSocket-based agent systems using AG2AI/Autogen, we encountered several major bottlenecks that affected productivity and developer experience:

- **Postman and raw WebSocket clients are not interactive**  
  These tools make it hard to follow multi-agent conversations. They lack formatting, which slows down debugging and understanding the data flow.

- **Reading agent messages is time-consuming**  
  When working with multiple agents, reviewing each step (especially during prompt tuning or alignment) becomes tedious and error-prone.

- **Lack of message formatting**  
  JSON responses from agents are dumped as raw strings, making them hard to read and troubleshoot — especially when nested or streamed.

- **Frontend development was not feasible**  
  Building a fully custom UI in frameworks like React or Vue would add significant overhead and distract from core system development.

- **No streamlined session management**  
  Keeping track of WebSocket sessions and switching between different chats was a manual and error-prone task.

---

### ✅ The Solution We Implemented

To overcome these challenges, we built a minimal yet powerful **interactive WebSocket UI**, paired with a **FastAPI backend**, enabling seamless development and debugging of agent workflows.

Key benefits:

- **Clean, interactive WebSocket communication**  
  Live messages stream directly to the browser with proper formatting and role-based separation.

- **Well-structured message display**  
  All messages are styled in blocks and automatically formatted as JSON, making it easy to inspect agent responses.

- **Faster prompt tuning & agent alignment**  
  Developers can instantly see how agents respond, helping fine-tune prompts with clarity and speed.

- **Quick session switching**  
  Chat sessions can be created and reused easily, improving workflow efficiency during development and testing.

- **Minimal development effort**  
  A lightweight HTML/JS UI replaces the need for building a full-fledged frontend framework — saving time while still improving UX significantly.

---

This setup dramatically reduced the friction in debugging, testing, and managing agentic workflows — allowing us to focus on what matters: building smart and responsive agents.

---

## 🤖 Agents Overview

This system supports the following agents for structured task completion:

- **`planner_agent`**: Produces a step-by-step execution plan (no code).
- **`code_writer`**: Converts the plan into working `python` code.
- **`code_executor`**: Executes code in the local runtime environment.
- **`debugger`**: Detects and resolves runtime errors; retries code.
- **`process_completion`**: Summarizes results and guides the next steps.
- `agent_aligner`: Coordinates the overall workflow, ensuring agents operate in the correct sequence. Enforces the execution flow (Plan → Confirm → Write → Confirm → Execute) to maintain structure, avoid loops, and ensure safe progression.


---

## 📁 Project Structure

```

.
├── managers/
│   ├── cancellation_token.py     # Manages cancellation signals
│   ├── connection.py             # Handles socket connections
│   ├── groupchat.py              # AgentChat orchestration logic
│   ├── prompts.py                # Prompt templates and roles
│
├── templates/
│   └── index.html                # Frontend UI (WebSocket client)
│
├── .env                          # API keys and environment variables
├── dependencies.py               # AG2AI agent setup and configuration
├── helpers.py                    # Utility functions
├── main.py                       # FastAPI server entry point
├── ws.py                         # WebSocket route handler
├── requirements.txt              # Python dependencies
└── Readme.md                     # You're reading it!

````

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Suryaaa-Rathore/websocket-ag2ai.git
cd websocket-ag2ai
````

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Your OpenAI API Key

Create a `.env` file in the root directory:

```
OPENAI_API_KEY=your-openai-api-key
```

Or export the variable directly:

```bash
export OPENAI_API_KEY=your-openai-api-key
```

### 5. Run the Server

```bash
python main.py
```

### 6. Access the Frontend

Open your browser and go to:

```
http://localhost:8000
```

---

## 🔍 Discoverability Tags

* FastAPI WebSocket Manager
* AG2AI Autogen Orchestrator
* Real-time agent workflows
* WebSocket frontend UI
* AI agent orchestration
* Agentic problem solving with Python
* Multi-agent system with streaming responses
* Custom group chat with Autogen

---