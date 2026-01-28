# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MICA (Materials Intelligence Co-Analyst) is an AI-powered multi-agent system for critical materials supply chain analysis, built for the Department of Energy. It uses LangGraph for orchestration and implements a human-in-the-loop workflow.

## Common Commands

### Running the Backend
```bash
cd backend
pip install -r requirements.txt
python -m mica.api.main
# Backend runs at http://localhost:8000
```

### Development with Hot Reload
```bash
cd backend
uvicorn mica.api.main:app --reload --port 8000
```

### Running Tests
```bash
cd backend
pip install -e ".[dev]"
pytest
```

### Code Style
```bash
cd backend
black mica/
ruff check mica/
mypy mica/
```

### Running UI Pipeline Server
```bash
cd ui
python pipelines_server.py
# Runs at http://localhost:9099
```

## Architecture

### Core Workflow (LangGraph State Machine)
The workflow in `backend/mica/agents/graph.py` follows this state machine:
```
INITIAL → RESEARCHING → PLAN_PROPOSED → AWAITING_APPROVAL →
EXECUTING → COMPLETED → AWAITING_FEEDBACK
```

Key interrupt points:
- `await_approval`: Pauses for user to approve/reject the analysis plan
- `await_feedback`: Pauses after results for user feedback/follow-ups

### Key Components

**State Management** (`backend/mica/agents/state.py`):
- `AgentState`: TypedDict that flows through all workflow nodes
- `WorkflowStatus`: Enum tracking workflow state
- `AnalysisPlan`: Parses LLM-generated plans into structured steps

**Orchestrator** (`backend/mica/agents/orchestrator.py`):
- `conduct_preliminary_research()`: Initial query analysis, detects simple vs complex queries
- `generate_plan()`: Creates structured analysis plans with tool assignments
- `execute_plan()`: Runs plan steps, invokes MCP tools
- `generate_final_summary()`: Synthesizes results into report

**LLM Integration** (`backend/mica/llm/`):
- Factory pattern in `factory.py` creates LLM instances
- Supports two providers: Argo (ANL internal) and Gemini (Google)
- Credentials managed via `backend/mica/credentials.py`

**MCP Tools** (`backend/mica/mcp_tools/`):
- `web_search`: Federal document-focused web search
- `pdf_rag`: PDF parsing with ChromaDB for semantic search
- `excel_handler`: Excel/CSV file operations
- `code_agent`: Python code execution
- `doc_generator`: PDF report generation
- `simulation`: Placeholder for supply chain models

### API Layer
- FastAPI app in `backend/mica/api/main.py`
- REST routes in `backend/mica/api/routes.py`
- WebSocket support in `backend/mica/api/websocket.py`

### UI Integration
- Open WebUI pipe in `ui/pipes/mica_pipe.py` connects to backend API
- Handles approval/feedback flow through polling

## Configuration

Configuration uses Pydantic settings in `backend/mica/config.py`. Key environment variables:
- `MICA_LLM_PROVIDER`: `argo` or `gemini`
- `MICA_DEFAULT_MODEL`: Model ID (e.g., `claudeopus4`)
- `ARGO_USERNAME`: For Argo provider
- `GOOGLE_API_KEY`: For Gemini provider
- `MICA_SESSION_DIR`: Session storage path (default: `./sessions`)

## Session Logging

Each query creates a session folder under `backend/sessions/{session_id}/` containing:
- `metadata.json`: Session metadata
- `query.txt`: Original query
- `plan.json`: Analysis plan
- `agent_logs/`: Per-agent execution logs
