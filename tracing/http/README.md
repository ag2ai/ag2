# OpenTelemetry Tracing Example

This example demonstrates how to set up OpenTelemetry tracing with FastAPI services, communicating via HTTPX, and visualizing traces in Grafana using Tempo as the backend.

## Prerequisites

- Docker and Docker Compose
- Python 3.8+

## Setup

1.  **Start Infrastructure**

    Start Grafana and Tempo using Docker Compose:

    ```bash
    cd tracing
    docker-compose up -d
    ```

    - **Grafana**: http://localhost:3000
    - **Tempo**: http://localhost:3200 (ingestion), http://localhost:4317 (OTLP gRPC), http://localhost:4318 (OTLP HTTP)

2.  **Install Python Dependencies**

    It is recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run Services**

    You need to run two services in separate terminals.

    **Terminal 1 (Service B - The Backend):**
    ```bash
    python service_b.py
    ```
    Service B runs on `http://localhost:8001`.

    **Terminal 2 (Service A - The Frontend):**
    ```bash
    python service_a.py
    ```
    Service A runs on `http://localhost:8000`.

## Generating Traces

Make a request to Service A:

```bash
curl http://localhost:8000/
```

Service A will call Service B. Both services are instrumented, and the trace context is propagated via HTTP headers.

## Viewing Traces

1.  Open Grafana at [http://localhost:3000](http://localhost:3000).
2.  Go to **Explore** (compass icon on the left).
3.  Select **Tempo** from the data source dropdown.
4.  You can search for traces or look at recent ones.
    - Click on "Search" tab in Query type.
    - Click "Run query" to see recent traces.
5.  Click on a Trace ID to view the waterfall visualization. You should see spans from `service-a` calling `service-b`.

## Components

- **Service A**: FastAPI service listening on port 8000. It makes an HTTP call to Service B using `httpx`.
- **Service B**: FastAPI service listening on port 8001.
- **Tempo**: Distributed tracing backend.
- **Grafana**: Visualization tool.

