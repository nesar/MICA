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


async def run_workflow_until_interrupt(session_id: str, state_update: dict = None):
    """Run the workflow until an interrupt point.

    Args:
        session_id: The session ID
        state_update: Optional dict of state updates for resumption (e.g., approval)
    """
    state = _sessions[session_id]
    workflow = get_workflow()

    # Preserve important state that shouldn't be overwritten by workflow events
    preserved_keys = {}
    if state_update:
        preserved_keys = state_update.copy()

    try:
        # Run workflow with config
        config_dict = {"configurable": {"thread_id": session_id}}

        # Determine what to pass to the workflow
        if state_update is not None:
            # Resuming from interrupt - update the checkpoint state first
            logger.info(f"[{session_id}] Resuming workflow with update: {state_update}")

            # Update the checkpointed state before resuming
            try:
                workflow.update_state(config_dict, state_update)
                logger.info(f"[{session_id}] Checkpoint state updated")
            except Exception as e:
                logger.warning(f"[{session_id}] Failed to update checkpoint: {e}")

            # Also ensure our local state has the update
            state.update(state_update)
            _sessions[session_id] = state

            # Resume with None (continue from checkpoint)
            workflow_input = None
        else:
            # Initial run - pass the full state
            workflow_input = state
            logger.info(f"[{session_id}] Starting workflow")

        # Stream until interrupt - LangGraph returns {node_name: state} dicts
        async for event in workflow.astream(workflow_input, config_dict):
            if isinstance(event, dict):
                # LangGraph events are {node_name: updated_state}
                for node_name, updated_state in event.items():
                    if isinstance(updated_state, dict):
                        # Merge the updated state, but preserve important keys
                        state.update(updated_state)
                        # Re-apply preserved keys (like approved) that shouldn't be overwritten
                        if preserved_keys:
                            state.update(preserved_keys)
                        _sessions[session_id] = state
                        logger.info(f"[{session_id}] Node '{node_name}' completed, status: {state.get('status')}, approved: {state.get('approved')}")

        # Get the final state from the workflow checkpoint to ensure sync
        try:
            final_state_snapshot = workflow.get_state(config_dict)
            if final_state_snapshot and final_state_snapshot.values:
                final_workflow_state = final_state_snapshot.values
                # Update our state with the final workflow state
                state.update(final_workflow_state)
                # Re-apply preserved keys
                if preserved_keys:
                    state.update(preserved_keys)
                logger.info(f"[{session_id}] Synced final state from checkpoint: status={state.get('status')}")
        except Exception as e:
            logger.warning(f"[{session_id}] Could not get final state from checkpoint: {e}")

        # Always update the final state
        _sessions[session_id] = state
        logger.info(f"[{session_id}] Workflow paused at: {state.get('current_step')}, status: {state.get('status')}")

    except Exception as e:
        logger.error(f"Workflow error: {e}", exc_info=True)
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

    if state["status"] not in [WorkflowStatus.AWAITING_APPROVAL, WorkflowStatus.PLAN_PROPOSED]:
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

    # Continue workflow - pass only the approval update for proper resumption
    state_update = {
        "approved": request.approved,
        "approval_feedback": request.feedback,
    }
    background_tasks.add_task(run_workflow_until_interrupt, session_id, state_update)

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
        # Continue with follow-up - pass the metadata update for proper checkpoint resumption
        state_update = {
            "metadata": state["metadata"].copy(),  # Include the follow_up_query
        }
        background_tasks.add_task(run_workflow_until_interrupt, session_id, state_update)
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
    from ..credentials import credentials_manager

    return {
        "llm_provider": config.llm.llm_provider,
        "default_model": config.llm.default_model,
        "search_provider": config.search.search_provider,
        "log_level": config.logging.log_level,
        "credentials": credentials_manager.get_credentials_status(),
    }


# =============================================================================
# Credentials Endpoints
# =============================================================================

class CredentialsRequest(BaseModel):
    """Request model for setting credentials."""
    provider: str = Field(..., description="Provider: 'argo' or 'gemini'")
    credential: str = Field(..., description="Username for Argo, API key for Gemini")


class CredentialsResponse(BaseModel):
    """Response model for credentials operations."""
    success: bool
    provider: str
    message: str


@router.get("/credentials")
async def get_credentials_status():
    """
    Check which credentials are configured.

    Returns the status of each LLM provider's credentials without exposing
    the actual values. Use this to determine if credentials need to be set.
    """
    from ..credentials import credentials_manager

    status = credentials_manager.get_credentials_status()

    return {
        "credentials": status,
        "requires_setup": not (
            status["argo"]["configured"] or status["gemini"]["configured"]
        ),
        "message": (
            "At least one LLM provider must be configured."
            if not (status["argo"]["configured"] or status["gemini"]["configured"])
            else "Credentials configured."
        ),
    }


@router.post("/credentials", response_model=CredentialsResponse)
async def set_credentials(request: CredentialsRequest):
    """
    Set credentials for an LLM provider.

    For Argo: provide your Argo username
    For Gemini: provide your Google API key

    Credentials are stored in memory and will need to be re-entered
    if the server restarts.
    """
    from ..credentials import credentials_manager

    provider = request.provider.lower()

    if provider == "argo":
        credentials_manager.set_argo_username(request.credential)
        return CredentialsResponse(
            success=True,
            provider="argo",
            message="Argo username set successfully.",
        )
    elif provider in ("gemini", "google"):
        credentials_manager.set_google_api_key(request.credential)
        return CredentialsResponse(
            success=True,
            provider="gemini",
            message="Google API key set successfully.",
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider: {provider}. Use 'argo' or 'gemini'.",
        )


@router.delete("/credentials/{provider}")
async def clear_credentials(provider: str):
    """Clear credentials for a specific provider."""
    from ..credentials import credentials_manager

    provider = provider.lower()

    if provider == "argo":
        credentials_manager.clear_argo_credentials()
        return {"message": "Argo credentials cleared."}
    elif provider in ("gemini", "google"):
        credentials_manager.clear_google_credentials()
        return {"message": "Gemini credentials cleared."}
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider: {provider}. Use 'argo' or 'gemini'.",
        )


@router.post("/credentials/test")
async def test_credentials():
    """
    Test if the configured credentials work.

    Makes a simple API call to verify the credentials are valid.
    """
    from langchain_core.messages import HumanMessage
    from ..credentials import credentials_manager
    from ..llm import create_llm

    results = {}

    # Test Argo if configured
    if credentials_manager.has_argo_credentials():
        try:
            llm = create_llm(provider="argo")
            messages = [HumanMessage(content="Say 'Hello from MICA!' in exactly 5 words.")]
            response = llm.invoke(messages)
            results["argo"] = {
                "success": True,
                "message": "Argo credentials verified.",
                "response_preview": response.content[:100] if response.content else None,
            }
        except Exception as e:
            results["argo"] = {
                "success": False,
                "message": f"Argo test failed: {str(e)}",
            }
    else:
        results["argo"] = {
            "success": False,
            "message": "Argo credentials not configured.",
        }

    # Test Gemini if configured
    if credentials_manager.has_google_credentials():
        try:
            llm = create_llm(provider="gemini")
            messages = [HumanMessage(content="Say 'Hello from MICA!' in exactly 5 words.")]
            response = llm.invoke(messages)
            results["gemini"] = {
                "success": True,
                "message": "Gemini credentials verified.",
                "response_preview": response.content[:100] if response.content else None,
            }
        except Exception as e:
            results["gemini"] = {
                "success": False,
                "message": f"Gemini test failed: {str(e)}",
            }
    else:
        results["gemini"] = {
            "success": False,
            "message": "Gemini credentials not configured.",
        }

    return {"results": results}
