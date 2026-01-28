# MICA - Materials Intelligence Co-Analyst

An AI-powered multi-agent system for critical materials supply chain analysis, developed for the Department of Energy.

---

## Quick Start (5 minutes)

### 1. Start the MICA Backend

```bash
cd backend
pip install -r requirements.txt
python -m mica.api.main
# Backend runs at http://localhost:8000
```

### 2. Start the Pipelines Server

```bash
cd ui
python pipelines_server.py
# Pipelines server runs at http://localhost:9099
```

### 3. Configure Open WebUI

In Open WebUI, go to **Settings → Connections → OpenAI API**:

| Field | Value |
|-------|-------|
| API Base URL | `http://host.docker.internal:9099/v1` |
| API Key | `mica-key` (any value works) |

Click **Save**. The "MICA" model will appear in the model dropdown.

### 4. Set Credentials (First Time)

```bash
# For Argo (ANL internal)
curl -X POST http://localhost:8000/api/v1/credentials \
  -H "Content-Type: application/json" \
  -d '{"provider": "argo", "credential": "YOUR_USERNAME"}'

# For Gemini
curl -X POST http://localhost:8000/api/v1/credentials \
  -H "Content-Type: application/json" \
  -d '{"provider": "gemini", "credential": "YOUR_API_KEY"}'
```

### 5. Start Chatting

Select "mica-analyst" from the model dropdown and ask your question.

---

## Overview

MICA is a LangGraph-based agentic framework that serves as an intelligent co-analyst for exploring research pathways, identifying cost-competitive sourcing strategies, analyzing market dynamics, and prioritizing investments in domestic supply chain capabilities.

### Key Features

- **Multi-Agent Architecture**: Orchestration agent with specialized sub-agents for web search, document analysis, data processing, and report generation
- **Human-in-the-Loop (HITL)**: Two-phase approval workflow - user approves plan before execution, reviews results before completion
- **Comprehensive Logging**: Every query creates a session folder with complete audit trail
- **Multiple LLM Backends**: Supports Argo (internal lab) and Google Gemini models
- **Open WebUI Integration**: User-friendly chat interface via Open WebUI pipes

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Open WebUI (Docker)                         │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ HTTP/WebSocket
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MICA Backend (FastAPI)                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Orchestration Agent (LangGraph)                 │  │
│  │  - Receives query → preliminary research → proposes plan     │  │
│  │  - Waits for user approval                                   │  │
│  │  - Delegates to sub-agents after approval                    │  │
│  │  - Compiles final report for user review                     │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────── MCP Tools ──────────────────────────┐    │
│  │  Web Search │ PDF/RAG │ Excel │ Code │ Simulation │ Report │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌──────────────────── Logging System ────────────────────────┐    │
│  │  /sessions/{session_id}/ - Complete audit trail            │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.10+ (for local development)
- One of:
  - Argo access (VPN + ARGO_USERNAME)
  - Google API key (GOOGLE_API_KEY)

### Using Docker (Recommended)

1. **Clone and configure:**
   ```bash
   cd /path/to/MICA
   cp .env.example .env
   # Edit .env with your credentials
   ```

2. **Start services:**
   ```bash
   docker-compose up -d
   ```

3. **Access the UI:**
   - Open WebUI: http://localhost:3000
   - API Docs: http://localhost:8000/docs

### Local Development

1. **Install dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   export MICA_LLM_PROVIDER=argo  # or gemini
   export ARGO_USERNAME=your_username  # if using Argo
   export GOOGLE_API_KEY=your_key  # if using Gemini
   ```

3. **Run the backend:**
   ```bash
   python -m mica.api.main
   ```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MICA_LLM_PROVIDER` | LLM provider (argo/gemini) | `argo` |
| `MICA_DEFAULT_MODEL` | Default model ID | `claudeopus4` |
| `ARGO_USERNAME` | Argo API username | - |
| `GOOGLE_API_KEY` | Google API key | - |
| `MICA_SESSION_DIR` | Session storage path | `./sessions` |
| `MICA_LOG_LEVEL` | Log level | `INFO` |

See `.env.example` for complete configuration options.

## Usage

### Via Open WebUI

1. Navigate to http://localhost:3000
2. Select "MICA" from the model dropdown
3. Enter your analysis query
4. Review and approve the proposed plan
5. Wait for execution and review results

### Via API

```python
import httpx

# Submit a query
response = httpx.post("http://localhost:8000/api/v1/query", json={
    "query": "What are the supply chain risks for rare earth magnets?",
    "user_id": "analyst1"
})
session_id = response.json()["session_id"]

# Check status and get plan
status = httpx.get(f"http://localhost:8000/api/v1/session/{session_id}")
plan = httpx.get(f"http://localhost:8000/api/v1/session/{session_id}/plan")

# Approve the plan
httpx.post(f"http://localhost:8000/api/v1/session/{session_id}/approve", json={
    "session_id": session_id,
    "approved": True
})

# Get results
results = httpx.get(f"http://localhost:8000/api/v1/session/{session_id}")
```

## MCP Tools

MICA includes the following MCP-compatible tools:

| Tool | Description |
|------|-------------|
| `web_search` | Web search with federal document focus |
| `pdf_rag` | PDF parsing and semantic search (RAG) |
| `excel_handler` | Excel file read/write operations |
| `code_agent` | Python code execution for analysis |
| `simulation` | Supply chain simulation (placeholder) |
| `doc_generator` | PDF report generation |

## Project Structure

```
MICA/
├── backend/
│   ├── mica/
│   │   ├── api/          # FastAPI routes and WebSocket
│   │   ├── agents/       # LangGraph orchestration
│   │   ├── llm/          # LLM providers (Argo, Gemini)
│   │   ├── mcp_tools/    # MCP-compatible tools
│   │   └── logging/      # Session logging
│   ├── sessions/         # Session data storage
│   ├── Dockerfile
│   └── requirements.txt
├── ui/
│   ├── pipes/            # Open WebUI pipe
│   └── Dockerfile
├── docs/
│   └── deployment.md
├── docker-compose.yml
└── .env.example
```

## Workflow States

```
INITIAL → RESEARCHING → PLAN_PROPOSED → AWAITING_APPROVAL →
EXECUTING → COMPLETED → AWAITING_FEEDBACK
```

- **AWAITING_APPROVAL**: User must approve/reject the plan
- **AWAITING_FEEDBACK**: User can provide feedback or follow-up queries

## Development

### Running Tests

```bash
cd backend
pip install -e ".[dev]"
pytest
```

### Code Style

```bash
black mica/
ruff check mica/
mypy mica/
```

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Acknowledgments

Developed by the MICA Team for the Department of Energy.
