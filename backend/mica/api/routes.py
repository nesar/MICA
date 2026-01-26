"""
MICA API Routes

REST API endpoints for the MICA system.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from ..agents import (
    AgentState,
    WorkflowStatus,
    create_initial_state,
    get_workflow,
    get_plan_summary,
)
from ..config import config
from ..llm import get_available_models
from ..logging import SessionLogger, get_session_logger, register_session
from ..mcp_tools import tool_registry

logger = logging.getLogger(__name__)

router = APIRouter(tags=["mica"])


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for submitting a query."""

    query: str = Field(..., description="The user's analysis query")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    model: Optional[str] = Field(None, description="LLM model to use")
    session_id: Optional[str] = Field(None, description="Existing session ID to continue")


class QueryResponse(BaseModel):
    """Response model for query submission."""

    session_id: str
    status: str
    message: str
    plan: Optional[List[Dict[str, Any]]] = None
    plan_summary: Optional[str] = None


class ApprovalRequest(BaseModel):
    """Request model for plan approval."""

    session_id: str = Field(..., description="Session ID")
    approved: bool = Field(..., description="Whether to approve the plan")
    feedback: Optional[str] = Field(None, description="Optional feedback")


class ApprovalResponse(BaseModel):
    """Response model for approval."""

    session_id: str
    status: str
    message: str


class FeedbackRequest(BaseModel):
    """Request model for user feedback."""

    session_id: str = Field(..., description="Session ID")
    feedback: str = Field(..., description="User feedback")
    follow_up_query: Optional[str] = Field(None, description="Follow-up query")


class SessionStatus(BaseModel):
    """Session status response."""

    session_id: str
    status: str
    current_step: Optional[str]
    plan: Optional[List[Dict[str, Any]]]
    results: Optional[Dict[str, Any]]
    errors: List[Dict[str, str]]
    final_summary: Optional[str]
    report_path: Optional[str]


# In-memory session state storage (use Redis in production)
_sessions: Dict[str, AgentState] = {}


def get_session(session_id: str) -> AgentState:
    """Get a session by ID."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return _sessions[session_id]


async def run_workflow_until_interrupt(session_id: str):
    """Run the workflow until an interrupt point."""
    state = _sessions[session_id]
    workflow = get_workflow()

    try:
        # Run workflow with config
        config_dict = {"configurable": {"thread_id": session_id}}

        # Stream until interrupt
        async for event in workflow.astream(state, config_dict):
            # Update stored state
            if isinstance(event, dict):
                for key, value in event.items():
                    if key in state:
                        _sessions[session_id] = value
                        state = value

        logger.info(f"Workflow paused at: {state.get('current_step')}")

    except Exception as e:
        logger.error(f"Workflow error: {e}")
        state["status"] = WorkflowStatus.FAILED
        state["errors"].append({
            "error": str(e),
            "context": "workflow_execution",
            "timestamp": datetime.utcnow().isoformat(),
        })
        _sessions[session_id] = state


# API Endpoints
@router.post("/query", response_model=QueryResponse)
async def submit_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
):
    """
    Submit a new analysis query.

    This starts the MICA workflow:
    1. Creates a new session
    2. Runs preliminary research
    3. Generates an analysis plan
    4. Returns the plan for user approval
    """
    # Create or retrieve session
    session_id = request.session_id or str(uuid.uuid4())

    if session_id in _sessions:
        # Continuing existing session
        state = _sessions[session_id]
        if state["status"] not in [WorkflowStatus.AWAITING_APPROVAL, WorkflowStatus.AWAITING_FEEDBACK]:
            raise HTTPException(
                status_code=400,
                detail=f"Session {session_id} is not awaiting input (status: {state['status']})",
            )
    else:
        # Create new session
        state = create_initial_state(
            session_id=session_id,
            query=request.query,
            user_id=request.user_id,
        )
        _sessions[session_id] = state

        # Create session logger
        session_logger = SessionLogger.create_session(session_id)
        session_logger.log_query(request.query, request.user_id, request.model)
        register_session(session_logger)

    # Run workflow in background until it hits the approval interrupt
    background_tasks.add_task(run_workflow_until_interrupt, session_id)

    return QueryResponse(
        session_id=session_id,
        status="processing",
        message="Query submitted. Workflow will generate a plan for your approval.",
    )


@router.get("/session/{session_id}", response_model=SessionStatus)
async def get_session_status(session_id: str):
    """
    Get the current status of a session.

    Returns the workflow status, current plan, and any results.
    """
    state = get_session(session_id)

    return SessionStatus(
        session_id=session_id,
        status=state["status"].value if isinstance(state["status"], WorkflowStatus) else state["status"],
        current_step=state.get("current_step"),
        plan=[dict(s) for s in state.get("plan", [])],
        results=state.get("tool_results"),
        errors=state.get("errors", []),
        final_summary=state.get("final_summary"),
        report_path=state.get("final_report_path"),
    )


@router.get("/session/{session_id}/plan")
async def get_session_plan(session_id: str):
    """Get the current plan for a session."""
    state = get_session(session_id)

    return {
        "session_id": session_id,
        "status": state["status"].value if isinstance(state["status"], WorkflowStatus) else state["status"],
        "plan": [dict(s) for s in state.get("plan", [])],
        "plan_summary": get_plan_summary(state),
        "reasoning": state.get("plan_reasoning"),
    }


@router.post("/session/{session_id}/approve", response_model=ApprovalResponse)
async def approve_plan(
    session_id: str,
    request: ApprovalRequest,
    background_tasks: BackgroundTasks,
):
    """
    Approve or reject the proposed analysis plan.

    If approved, the workflow continues with execution.
    If rejected with feedback, a revised plan may be generated.
    """
    state = get_session(session_id)

    if state["status"] != WorkflowStatus.AWAITING_APPROVAL:
        raise HTTPException(
            status_code=400,
            detail=f"Session is not awaiting approval (status: {state['status']})",
        )

    # Update state with approval decision
    state["approved"] = request.approved
    state["approval_feedback"] = request.feedback
    _sessions[session_id] = state

    # Log approval
    session_logger = get_session_logger(session_id)
    if session_logger:
        session_logger.log_approval(request.approved, request.feedback)

    # Continue workflow
    background_tasks.add_task(run_workflow_until_interrupt, session_id)

    if request.approved:
        message = "Plan approved. Execution starting..."
    else:
        message = "Plan rejected." + (" Revising based on feedback..." if request.feedback else "")

    return ApprovalResponse(
        session_id=session_id,
        status="processing" if request.approved else "revising",
        message=message,
    )


@router.post("/session/{session_id}/feedback")
async def submit_feedback(
    session_id: str,
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
):
    """
    Submit feedback on the analysis results.

    Can optionally include a follow-up query for additional analysis.
    """
    state = get_session(session_id)

    if state["status"] != WorkflowStatus.AWAITING_FEEDBACK:
        raise HTTPException(
            status_code=400,
            detail=f"Session is not awaiting feedback (status: {state['status']})",
        )

    # Update state with feedback
    if "metadata" not in state:
        state["metadata"] = {}
    state["metadata"]["feedback"] = request.feedback
    state["metadata"]["follow_up_query"] = request.follow_up_query
    _sessions[session_id] = state

    if request.follow_up_query:
        # Continue with follow-up
        background_tasks.add_task(run_workflow_until_interrupt, session_id)
        return {
            "session_id": session_id,
            "status": "processing",
            "message": "Follow-up query submitted. Processing...",
        }
    else:
        # Mark as complete
        state["status"] = WorkflowStatus.COMPLETED
        return {
            "session_id": session_id,
            "status": "completed",
            "message": "Thank you for your feedback. Session completed.",
        }


@router.get("/session/{session_id}/report")
async def get_report(session_id: str):
    """
    Get the generated report for a session.

    Returns the report path or indicates if not yet available.
    """
    state = get_session(session_id)

    report_path = state.get("final_report_path")

    if report_path:
        return {
            "session_id": session_id,
            "report_available": True,
            "report_path": report_path,
        }
    else:
        return {
            "session_id": session_id,
            "report_available": False,
            "status": state["status"].value if isinstance(state["status"], WorkflowStatus) else state["status"],
        }


@router.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    sessions = []
    for session_id, state in _sessions.items():
        sessions.append({
            "session_id": session_id,
            "status": state["status"].value if isinstance(state["status"], WorkflowStatus) else state["status"],
            "query": state.get("query", "")[:100],
            "created_at": state.get("created_at"),
        })

    return {"sessions": sessions, "count": len(sessions)}


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_id in _sessions:
        del _sessions[session_id]
        return {"message": f"Session {session_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")


# Utility endpoints
@router.get("/models")
async def list_models():
    """List available LLM models."""
    return get_available_models()


@router.get("/tools")
async def list_tools():
    """List available analysis tools."""
    return {
        "tools": tool_registry.get_all_schemas(),
    }


@router.get("/config")
async def get_config():
    """Get current configuration (non-sensitive)."""
    return {
        "llm_provider": config.llm.llm_provider,
        "default_model": config.llm.default_model,
        "search_provider": config.search.search_provider,
        "log_level": config.logging.log_level,
    }
