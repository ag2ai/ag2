---
title: "Deploy AG2 Agent to Google Cloud Platform"
sidebarTitle: "Deploy to GCP"
---

Author: [Priyanshu Deshmukh](https://github.com/priyansh4320)

This guide walks you through deploying an AG2 conversational agent to Google Cloud Platform using Cloud Run. Your agent will be exposed as a REST API using FastAPI and can scale automatically based on traffic.

## Prerequisites

Before you begin, ensure you have:

- **Google Cloud Platform Account** with billing enabled
- **Google Cloud SDK (gcloud CLI)** installed
- **Python 3.13+** installed
- **Gemini API Key** from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Docker**

## Step 1: Set Up Your Local Environment

### Install Dependencies

First, create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file in your project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

**Important:** Never commit your `.env` file to version control.

## Step 2: Test Locally

Before deploying, test your agent locally to ensure everything works:

```bash hl_lines="51-53"
# Run the FastAPI server
python agent.py
```

The server will start on `http://localhost:8080`. Test it with:

```bash
# Health check
curl http://localhost:8080/

# Chat endpoint
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "A joke about NYC.", "max_turn": 1}'
```

## Step 3: Set Up Google Cloud Project

### Create a New Project

You can create a project via the console or using the gcloud CLI:

```bash
gcloud projects create YOUR_PROJECT_ID --name="AG2 Agent Project"
gcloud config set project YOUR_PROJECT_ID
```

### Enable Required APIs

Enable the necessary Google Cloud APIs:

```bash
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com
```

### Authenticate

Set up authentication for your local machine:

```bash
gcloud auth login
gcloud auth application-default login
gcloud auth configure-docker
```

### Create Artifact Registry Repository

Create a Docker repository to store your container images:

```bash
gcloud artifacts repositories create ag2-agent-repo2 \
  --repository-format=docker \
  --location=us-central1 \
  --description="Docker repository for AG2 agent"
```

## Step 4: Configure Deployment Files

### Update deploy.sh

Edit `deploy.sh` and update the project-specific variables:

```bash hl_lines="4-7"
PROJECT_ID="your-actual-project-id"
REGION="us-central1"
IMAGE_NAME="ag2-agent"
REPOSITORY="ag2-agent-repo2"  # Match your Artifact Registry repo name
```

### Update cloudbuild.yaml (Optional)

If using Cloud Build, update the substitutions:

```yaml hl_lines="17-20"
substitutions:
  _REGION: 'us-central1'
  _REPOSITORY: 'ag2-agent-repo2'
  _IMAGE_NAME: 'ag2-agent'
```

## Step 5: Deploy to Cloud Run

### Quick Deployment

The simplest way to deploy is using the provided script:

```bash
# Set your Gemini API key
export GEMINI_API_KEY=your_gemini_api_key_here

# Make script executable and run
chmod +x deploy.sh
./deploy.sh
```

The script will:
1. Build your Docker image
2. Push it to Artifact Registry
3. Deploy to Cloud Run
4. Display your service URL

### Manual Deployment

If you prefer manual control, deploy step by step:

```bash
PROJECT_ID="your-project-id"
REGION="us-central1"
REPOSITORY="ag2-agent-repo2"
IMAGE_NAME="ag2-agent"

# Build and push image
gcloud builds submit --tag ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest

# Deploy to Cloud Run
gcloud run deploy ${IMAGE_NAME} \
  --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=${GEMINI_API_KEY} \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600 \
  --port 8080
```

## Step 6: Verify Deployment

### Get Your Service URL

```bash
gcloud run services describe ag2-agent --region=us-central1 --format="value(status.url)"
```

### Test the Deployed Service

Test your deployed agent:

```bash
# Health check
curl https://your-service-url.run.app/

# Chat endpoint
curl -X POST https://your-service-url.run.app/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "A joke about NYC.", "max_turn": 1}'
```

### Example Output

When you call the chat endpoint, you should receive a response like:

```json
{
  "response": "Why did the New Yorker go to therapy? Because they had too many issues!"
}
```

## Complete Deployment Example

Here are all the files you need for a complete deployment:

### agent.py

```python
from autogen import ConversableAgent, LLMConfig
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

load_dotenv()

app = FastAPI()

Gemini_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure LLM
llm_config = LLMConfig(
    config_list={
        "model": "gemini-2.5-flash",
        "api_type": "google",
        "api_key": Gemini_API_KEY,
    }
)

# Initialize agent
assistant = ConversableAgent(
    "assistant",
    system_message="You are a helpful assistant",
    llm_config=llm_config,
    human_input_mode="TERMINATE"
)

class MessageRequest(BaseModel):
    message: str
    max_turn: int = 1

@app.get("/")
def health_check():
    return {"status": "healthy"}

@app.post("/chat")
def chat(request: MessageRequest):
    try:
        response = assistant.run(
            messages=request.message,
            max_turn=request.max_turn
        )
        result = response.process()
        return {"response": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

### requirements.txt

```text
ag2[openai]>=0.9.9,<0.10.0
ag2[gemini]
python-dotenv>=1.0.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
```

### Dockerfile

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY agent.py .

# Create coding directory for code execution
RUN mkdir -p coding

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "agent.py"]
```

### cloudbuild.yaml

```yaml
steps:
  # Build the container image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', '${_REGION}-docker.pkg.dev/$PROJECT_ID/${_REPOSITORY}/${_IMAGE_NAME}:$COMMIT_SHA', '.']

  # Push the container image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '${_REGION}-docker.pkg.dev/$PROJECT_ID/${_REPOSITORY}/${_IMAGE_NAME}:$COMMIT_SHA']

  # Tag as latest
  - name: 'gcr.io/cloud-builders/docker'
    args: ['tag', '${_REGION}-docker.pkg.dev/$PROJECT_ID/${_REPOSITORY}/${_IMAGE_NAME}:$COMMIT_SHA', '${_REGION}-docker.pkg.dev/$PROJECT_ID/${_REPOSITORY}/${_IMAGE_NAME}:latest']

  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '${_REGION}-docker.pkg.dev/$PROJECT_ID/${_REPOSITORY}/${_IMAGE_NAME}:latest']

substitutions:
  _REGION: 'us-central1'
  _REPOSITORY: 'ag2-agent-repo2'
  _IMAGE_NAME: 'ag2-agent'

images:
  - '${_REGION}-docker.pkg.dev/$PROJECT_ID/${_REPOSITORY}/${_IMAGE_NAME}:$COMMIT_SHA'
  - '${_REGION}-docker.pkg.dev/$PROJECT_ID/${_REPOSITORY}/${_IMAGE_NAME}:latest'
```

### deploy.sh

```bash
#!/bin/bash

# Set your GCP project ID
PROJECT_ID="your-project-id"
REGION="us-central1"
IMAGE_NAME="ag2-agent"
REPOSITORY="ag2-agent-repo2"  # Artifact Registry repo name

# Set the project
gcloud config set project ${PROJECT_ID}

# Build and push the image to Artifact Registry
echo "Building Docker image..."
gcloud builds submit --tag ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest

# Deploy to Cloud Run (Recommended for web APIs)
echo "Deploying to Cloud Run..."
gcloud run deploy ${IMAGE_NAME} \
  --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=${GEMINI_API_KEY} \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600 \
  --port 8080

echo "Deployment complete!"
echo "Get your service URL:"
gcloud run services describe ${IMAGE_NAME} --region=${REGION} --format="value(status.url)"
```

## Troubleshooting

### Container Failed to Start

If you see an error about the container not listening on port 8080, ensure your `agent.py` uses the PORT environment variable:

```python hl_lines="51-53"
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

### Authentication Errors

If you encounter authentication issues:

```bash
gcloud auth login
gcloud auth application-default login
gcloud auth configure-docker
```

### View Logs

Check your deployment logs:

```bash
gcloud run services logs read ag2-agent --region=us-central1
```

Your Agent is now deployed on vertexai, thank you
