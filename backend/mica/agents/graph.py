"""
MICA LangGraph Workflow Definition

Defines the state machine and workflow for the MICA orchestration system.
The workflow follows a human-in-the-loop pattern:

1. Receive query → Preliminary research
2. Propose plan → Wait for user approval
3. Execute plan (agents run automatically)
4. Present results → Wait for feedback
"""

import logging
from typing import Literal

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState, WorkflowStatus
from .orchestrator import (
    conduct_preliminary_research,
    generate_plan,
    execute_plan,
    generate_final_summary,
    handle_feedback,
)

logger = logging.getLogger(__name__)


def should_continue_after_research(state: AgentState) -> Literal["propose_plan", "error"]:
    """Determine next step after preliminary research."""
    if state["errors"]:
        return "error"
    return "propose_plan"


def should_continue_after_plan(
    state: AgentState,
) -> Literal["await_approval", "error"]:
    """Determine next step after plan generation."""
    if state["errors"]:
        return "error"
    if not state["plan"]:
        return "error"
    return "await_approval"


def should_continue_after_approval(
    state: AgentState,
) -> Literal["execute", "revise_plan", "end"]:
    """Determine next step based on user approval."""
    if state["approved"] is None:
        # Still waiting
        return "end"
    if state["approved"]:
        return "execute"
    else:
        # User rejected, may want to revise
        if state["approval_feedback"]:
            return "revise_plan"
        return "end"


def should_continue_after_execution(
    state: AgentState,
) -> Literal["summarize", "error"]:
    """Determine next step after plan execution."""
    # Check for critical errors
    failed_steps = [s for s in state["plan"] if s["status"] == "failed"]
    if len(failed_steps) == len(state["plan"]):
        return "error"
    return "summarize"


def should_continue_after_summary(
    state: AgentState,
) -> Literal["await_feedback", "end"]:
    """Determine if we should wait for user feedback."""
    return "await_feedback"


def should_continue_after_feedback(
    state: AgentState,
) -> Literal["execute", "end"]:
    """Determine next step based on user feedback."""
    # If user provides new instructions, we might need to re-execute
    if state.get("metadata", {}).get("follow_up_query"):
        return "execute"
    return "end"


# Node functions that wrap orchestrator functions
def research_node(state: AgentState) -> AgentState:
    """Node: Conduct preliminary research."""
    logger.info(f"[{state['session_id']}] Starting preliminary research")
    state["status"] = WorkflowStatus.RESEARCHING
    state["current_step"] = "research"
    return conduct_preliminary_research(state)


def plan_node(state: AgentState) -> AgentState:
    """Node: Generate analysis plan."""
    logger.info(f"[{state['session_id']}] Generating analysis plan")
    state["current_step"] = "planning"
    return generate_plan(state)


def await_approval_node(state: AgentState) -> AgentState:
    """Node: Wait for user approval (interrupt point)."""
    logger.info(f"[{state['session_id']}] Awaiting user approval")
    state["status"] = WorkflowStatus.AWAITING_APPROVAL
    state["current_step"] = "awaiting_approval"
    # This is an interrupt point - the workflow will pause here
    return state


def execute_node(state: AgentState) -> AgentState:
    """Node: Execute the analysis plan."""
    logger.info(f"[{state['session_id']}] Executing analysis plan")
    state["status"] = WorkflowStatus.EXECUTING
    state["current_step"] = "executing"
    return execute_plan(state)


def summarize_node(state: AgentState) -> AgentState:
    """Node: Generate final summary and report."""
    logger.info(f"[{state['session_id']}] Generating final summary")
    state["current_step"] = "summarizing"
    return generate_final_summary(state)


def await_feedback_node(state: AgentState) -> AgentState:
    """Node: Wait for user feedback (interrupt point)."""
    logger.info(f"[{state['session_id']}] Awaiting user feedback")
    state["status"] = WorkflowStatus.AWAITING_FEEDBACK
    state["current_step"] = "awaiting_feedback"
    return state


def feedback_node(state: AgentState) -> AgentState:
    """Node: Handle user feedback."""
    logger.info(f"[{state['session_id']}] Processing user feedback")
    state["current_step"] = "processing_feedback"
    return handle_feedback(state)


def complete_node(state: AgentState) -> AgentState:
    """Node: Mark workflow as complete."""
    logger.info(f"[{state['session_id']}] Workflow complete")
    state["status"] = WorkflowStatus.COMPLETED
    state["current_step"] = "completed"
    return state


def error_node(state: AgentState) -> AgentState:
    """Node: Handle workflow errors."""
    logger.error(f"[{state['session_id']}] Workflow error: {state['errors']}")
    state["status"] = WorkflowStatus.FAILED
    state["current_step"] = "error"
    return state


def revise_plan_node(state: AgentState) -> AgentState:
    """Node: Revise plan based on user feedback."""
    logger.info(f"[{state['session_id']}] Revising plan based on feedback")
    state["current_step"] = "revising"
    # Re-generate plan with feedback context
    if state["approval_feedback"]:
        state["clarifications"].append(state["approval_feedback"])
    return generate_plan(state)


def create_workflow() -> StateGraph:
    """
    Create the MICA workflow graph.

    Returns:
        Compiled StateGraph with checkpointing enabled
    """
    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("research", research_node)
    workflow.add_node("propose_plan", plan_node)
    workflow.add_node("await_approval", await_approval_node)
    workflow.add_node("execute", execute_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("await_feedback", await_feedback_node)
    workflow.add_node("handle_feedback", feedback_node)
    workflow.add_node("complete", complete_node)
    workflow.add_node("error", error_node)
    workflow.add_node("revise_plan", revise_plan_node)

    # Set entry point
    workflow.set_entry_point("research")

    # Add edges with conditional routing
    workflow.add_conditional_edges(
        "research",
        should_continue_after_research,
        {
            "propose_plan": "propose_plan",
            "error": "error",
        },
    )

    workflow.add_conditional_edges(
        "propose_plan",
        should_continue_after_plan,
        {
            "await_approval": "await_approval",
            "error": "error",
        },
    )

    workflow.add_conditional_edges(
        "await_approval",
        should_continue_after_approval,
        {
            "execute": "execute",
            "revise_plan": "revise_plan",
            "end": END,
        },
    )

    # Revise plan goes back to await approval
    workflow.add_edge("revise_plan", "await_approval")

    workflow.add_conditional_edges(
        "execute",
        should_continue_after_execution,
        {
            "summarize": "summarize",
            "error": "error",
        },
    )

    workflow.add_conditional_edges(
        "summarize",
        should_continue_after_summary,
        {
            "await_feedback": "await_feedback",
            "end": "complete",
        },
    )

    workflow.add_conditional_edges(
        "await_feedback",
        should_continue_after_feedback,
        {
            "execute": "execute",
            "end": "complete",
        },
    )

    workflow.add_edge("handle_feedback", "complete")
    workflow.add_edge("complete", END)
    workflow.add_edge("error", END)

    return workflow


def compile_workflow(checkpointer=None):
    """
    Compile the workflow with optional checkpointing.

    Args:
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Compiled workflow ready for execution
    """
    workflow = create_workflow()

    if checkpointer is None:
        checkpointer = MemorySaver()

    # Compile with interrupt points for HITL
    compiled = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["await_approval", "await_feedback"],
    )

    return compiled


# Global compiled workflow instance
_workflow = None


def get_workflow():
    """Get the compiled workflow instance."""
    global _workflow
    if _workflow is None:
        _workflow = compile_workflow()
    return _workflow


def reset_workflow():
    """Reset the workflow instance (for testing)."""
    global _workflow
    _workflow = None
