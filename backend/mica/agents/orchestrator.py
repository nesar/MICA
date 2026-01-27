"""
MICA Orchestration Agent

The main orchestration logic for the MICA system. This module contains
the core functions that power each node in the LangGraph workflow.

The orchestrator:
1. Conducts preliminary research to understand the query
2. Generates an analysis plan with specific steps
3. Executes the plan by invoking appropriate tools
4. Compiles results into a final summary and report
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..config import config
from ..llm import create_llm
from ..logging import SessionLogger, register_session
from ..mcp_tools import (
    WebSearchTool,
    PDFRAGTool,
    ExcelHandlerTool,
    CodeAgentTool,
    SimulationTool,
    DocumentGeneratorTool,
    tool_registry,
)
from .state import (
    AgentState,
    WorkflowStatus,
    PlanStep,
    AnalysisPlan,
    add_error,
    add_artifact,
    update_plan_step,
)

logger = logging.getLogger(__name__)

# System prompts for the orchestrator
ORCHESTRATOR_SYSTEM_PROMPT = """You are MICA (Materials Intelligence Co-Analyst), an AI-powered
analyst specialized in critical materials supply chain analysis for the Department of Energy.

Your role is to:
1. Understand complex queries about critical materials supply chains
2. Develop comprehensive analysis plans
3. Synthesize information from multiple sources
4. Provide actionable insights for policy and investment decisions

Focus areas include:
- Rare earth elements and permanent magnets
- Battery materials (lithium, cobalt, nickel, graphite)
- Semiconductor materials
- Supply chain risks and vulnerabilities
- Domestic production capacity
- Trade flows and import dependencies
- Cost analysis and market dynamics

Always ground your analysis in available data and cite sources.
When uncertain, clearly state limitations and assumptions.
"""

RESEARCH_PROMPT = """Based on the user's query, conduct preliminary research to understand:
1. What specific information is being requested
2. What data sources and analyses would be most relevant
3. Any clarifying questions that might be needed

Query: {query}

Previous context: {context}

Provide a brief summary of your understanding and initial findings.
Focus on identifying the key aspects that need to be addressed.
"""

PLAN_PROMPT = """Based on the user's query and preliminary research, create a detailed analysis plan.

Query: {query}

Preliminary Research:
{research_summary}

User clarifications: {clarifications}

Create a step-by-step plan with specific actions. For each step, specify:
1. What tool/analysis to use
2. What specific inputs are needed
3. What output is expected

Available tools:
- web_search: Search for information from federal documents and web sources
- pdf_rag: Search and extract information from PDF documents
- excel_handler: Read and analyze Excel data files
- code_agent: Run statistical analysis and create visualizations
- simulation: Run supply chain simulations (GCMat, RELOG)
- doc_generator: Generate PDF reports

Format your plan as a numbered list of specific, actionable steps.
"""

SUMMARY_PROMPT = """Synthesize the analysis results into a comprehensive summary.

Original Query: {query}

Analysis Results:
{results}

Provide:
1. Executive Summary (2-3 key findings)
2. Detailed Findings (organized by topic)
3. Limitations and Caveats
4. Recommendations (if applicable)

Be specific and cite data sources where relevant.
"""


def get_llm():
    """Get the configured LLM instance."""
    try:
        return create_llm()
    except Exception as e:
        logger.error(f"Failed to create LLM: {e}")
        raise


def get_session_logger(state: AgentState) -> Optional[SessionLogger]:
    """Get or create a session logger for the state."""
    session_id = state["session_id"]

    try:
        # Try to load existing session
        session = SessionLogger.load_session(session_id)
    except FileNotFoundError:
        # Create new session
        session = SessionLogger.create_session(session_id)
        session.log_query(state["query"], state.get("user_id"))
        register_session(session)

    return session


def conduct_preliminary_research(state: AgentState) -> AgentState:
    """
    Conduct preliminary research to understand the query.

    This step:
    1. Parses the user's query
    2. Performs initial web searches for context
    3. Identifies relevant data sources
    4. Summarizes initial findings
    """
    logger.info(f"[{state['session_id']}] Conducting preliminary research")

    session = get_session_logger(state)

    try:
        llm = get_llm()

        # Build context from any previous interactions
        context = ""
        if state["clarifications"]:
            context = "\n".join(state["clarifications"])

        # Generate research prompt
        prompt = RESEARCH_PROMPT.format(
            query=state["query"],
            context=context or "No previous context.",
        )

        # Get LLM response
        messages = [
            SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = llm.invoke(messages)
        research_summary = response.content

        # Optionally do a quick web search
        web_search = WebSearchTool(session_logger=session)
        search_result = web_search.execute({
            "query": state["query"],
            "num_results": 5,
            "federal_only": True,
        })

        # Compile preliminary research
        state["preliminary_research"] = {
            "summary": research_summary,
            "web_search_results": search_result.data if search_result.data else [],
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Update messages
        state["messages"].append(HumanMessage(content=state["query"]))
        state["messages"].append(AIMessage(content=research_summary))

        # Log research
        if session:
            with session.agent_logger("orchestrator") as agent_log:
                agent_log.log("Preliminary research completed")
                agent_log.log_output(state["preliminary_research"])

        logger.info(f"[{state['session_id']}] Preliminary research completed")

    except Exception as e:
        logger.error(f"[{state['session_id']}] Research error: {e}")
        state = add_error(state, str(e), "preliminary_research")

    return state


def generate_plan(state: AgentState) -> AgentState:
    """
    Generate an analysis plan based on the query and research.

    This creates a structured plan with specific steps and tool assignments.
    """
    logger.info(f"[{state['session_id']}] Generating analysis plan")

    session = get_session_logger(state)

    try:
        llm = get_llm()

        # Build the planning prompt
        research_summary = state["preliminary_research"].get("summary", "")
        clarifications = "\n".join(state["clarifications"]) if state["clarifications"] else "None"

        prompt = PLAN_PROMPT.format(
            query=state["query"],
            research_summary=research_summary,
            clarifications=clarifications,
        )

        messages = state["messages"].copy()
        messages.append(HumanMessage(content=prompt))

        response = llm.invoke(messages)
        plan_text = response.content

        # Parse the plan
        plan = AnalysisPlan.from_llm_response(plan_text)
        steps, reasoning = plan.to_state_format()

        state["plan"] = steps
        state["plan_reasoning"] = reasoning
        state["status"] = WorkflowStatus.PLAN_PROPOSED

        # Add to messages
        state["messages"].append(AIMessage(content=plan_text))

        # Log plan
        if session:
            session.log_plan({
                "reasoning": reasoning,
                "steps": [dict(s) for s in steps],
                "estimated_tools": plan.estimated_tools,
            })

        logger.info(f"[{state['session_id']}] Generated plan with {len(steps)} steps")

    except Exception as e:
        logger.error(f"[{state['session_id']}] Plan generation error: {e}")
        state = add_error(state, str(e), "plan_generation")

    return state


def execute_plan(state: AgentState) -> AgentState:
    """
    Execute the approved analysis plan.

    Runs each step in sequence, invoking the appropriate tools.
    """
    logger.info(f"[{state['session_id']}] Executing analysis plan")

    session = get_session_logger(state)

    # Tool mapping
    tools = {
        "web_search": WebSearchTool(session_logger=session),
        "pdf_rag": PDFRAGTool(session_logger=session),
        "excel_handler": ExcelHandlerTool(session_logger=session),
        "code_agent": CodeAgentTool(session_logger=session),
        "simulation": SimulationTool(session_logger=session),
        "doc_generator": DocumentGeneratorTool(session_logger=session),
    }

    for step in state["plan"]:
        if step["status"] == "completed":
            continue  # Skip already completed steps

        step_id = step["step_id"]
        tool_name = step["tool"]

        logger.info(f"[{state['session_id']}] Executing step {step_id}: {tool_name}")

        # Update step status
        state = update_plan_step(state, step_id, "running")

        try:
            tool = tools.get(tool_name)

            if tool is None:
                # Use orchestrator LLM for generic steps
                result = _execute_with_llm(state, step, session)
            else:
                # Execute tool
                inputs = step.get("inputs", {})

                # Add context from previous steps if needed
                inputs = _enrich_inputs(inputs, step, state)

                result = tool.execute(inputs)

                if result.status.value == "success":
                    state["tool_results"][step_id] = result.data
                    state["intermediate_outputs"].append({
                        "step_id": step_id,
                        "tool": tool_name,
                        "output": result.data,
                    })
                    state = update_plan_step(state, step_id, "completed", output=result.data)
                else:
                    state = update_plan_step(
                        state, step_id, "failed", error=result.error or result.message
                    )

            # Log execution
            if session:
                with session.agent_logger(tool_name) as agent_log:
                    agent_log.log(f"Executed step {step_id}")
                    if tool:
                        agent_log.log_output(result.to_dict() if hasattr(result, 'to_dict') else str(result))

        except Exception as e:
            logger.error(f"[{state['session_id']}] Step {step_id} error: {e}")
            state = update_plan_step(state, step_id, "failed", error=str(e))
            state = add_error(state, str(e), f"step_{step_id}")

    logger.info(f"[{state['session_id']}] Plan execution completed")
    return state


def _execute_with_llm(
    state: AgentState,
    step: PlanStep,
    session: Optional[SessionLogger],
) -> Any:
    """Execute a step using the LLM directly."""
    llm = get_llm()

    prompt = f"""Execute this analysis step:

Step: {step['description']}

Context:
Query: {state['query']}
Previous results: {state['intermediate_outputs'][-3:] if state['intermediate_outputs'] else 'None'}

Provide a thorough analysis for this step.
"""

    messages = [
        SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)

    state["tool_results"][step["step_id"]] = response.content
    state["intermediate_outputs"].append({
        "step_id": step["step_id"],
        "tool": "orchestrator",
        "output": response.content,
    })
    update_plan_step(state, step["step_id"], "completed", output=response.content)

    return response.content


def _enrich_inputs(
    inputs: Dict[str, Any],
    step: PlanStep,
    state: AgentState,
) -> Dict[str, Any]:
    """Enrich tool inputs with context from previous steps."""
    # For web_search, ensure we have a proper query
    if step["tool"] == "web_search":
        # Check if we have queries (plural) from plan parsing
        if "queries" in inputs and inputs["queries"]:
            # Use the first query from the list, or combine them
            queries = inputs["queries"]
            if isinstance(queries, list) and queries:
                # Use first specific query, or combine top queries
                inputs["query"] = queries[0] if len(queries) == 1 else " ".join(queries[:2])
        elif "query" not in inputs:
            # Fall back to step description, but make it more search-friendly
            description = step.get("description", state["query"])
            # Clean up the description to be a better search query
            # Remove generic phrases that don't help search
            description = description.replace("Research ", "").replace("Investigate ", "")
            description = description.replace("Identify ", "").replace("Verify ", "")
            inputs["query"] = description

    # Add data from previous steps if referenced
    # This is a simplified implementation
    return inputs


def generate_final_summary(state: AgentState) -> AgentState:
    """
    Generate the final summary and report.

    Synthesizes all results into a comprehensive output.
    """
    logger.info(f"[{state['session_id']}] Generating final summary")

    session = get_session_logger(state)

    try:
        llm = get_llm()

        # Compile results
        results_text = ""
        for output in state["intermediate_outputs"]:
            results_text += f"\n## {output['step_id']} ({output['tool']})\n"
            if isinstance(output["output"], dict):
                results_text += str(output["output"])[:2000]
            else:
                results_text += str(output["output"])[:2000]
            results_text += "\n"

        prompt = SUMMARY_PROMPT.format(
            query=state["query"],
            results=results_text,
        )

        messages = [
            SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = llm.invoke(messages)
        summary = response.content

        state["final_summary"] = summary
        state["messages"].append(AIMessage(content=summary))

        # Generate report if we have substantial results
        if len(state["intermediate_outputs"]) >= 2:
            doc_gen = DocumentGeneratorTool(session_logger=session)

            sections = [
                {
                    "title": "Executive Summary",
                    "content": summary.split("\n\n")[0] if "\n\n" in summary else summary[:500],
                },
                {
                    "title": "Analysis",
                    "content": summary,
                },
            ]

            # Add results sections
            for output in state["intermediate_outputs"][:5]:
                sections.append({
                    "title": f"Results: {output['step_id']}",
                    "content": str(output["output"])[:3000],
                })

            report_result = doc_gen.execute({
                "title": f"MICA Analysis: {state['query'][:50]}...",
                "sections": sections,
                "metadata": {
                    "Session ID": state["session_id"],
                    "Query": state["query"],
                    "Generated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                },
            })

            if report_result.data and "path" in report_result.data:
                state["final_report_path"] = report_result.data["path"]
                state = add_artifact(
                    state,
                    "final_report.pdf",
                    "report",
                    report_result.data["path"],
                )

        # Finalize session logging
        if session:
            session.finalize(
                status="completed",
                summary=summary[:500],
            )

        logger.info(f"[{state['session_id']}] Final summary generated")

    except Exception as e:
        logger.error(f"[{state['session_id']}] Summary generation error: {e}")
        state = add_error(state, str(e), "summary_generation")

    return state


def handle_feedback(state: AgentState) -> AgentState:
    """
    Handle user feedback after results presentation.

    Can trigger follow-up analysis or refinements.
    """
    logger.info(f"[{state['session_id']}] Handling user feedback")

    feedback = state.get("metadata", {}).get("feedback", "")

    if not feedback:
        # No feedback to handle
        return state

    # Add feedback to clarifications for potential follow-up
    state["clarifications"].append(f"User feedback: {feedback}")

    # Check if follow-up is requested
    follow_up_query = state.get("metadata", {}).get("follow_up_query")
    if follow_up_query:
        state["query"] = follow_up_query
        # Reset for new analysis
        state["plan"] = []
        state["tool_results"] = {}
        state["intermediate_outputs"] = []

    return state


def get_plan_summary(state: AgentState) -> str:
    """Get a human-readable summary of the current plan."""
    if not state["plan"]:
        return "No plan generated yet."

    lines = ["## Analysis Plan\n"]

    for i, step in enumerate(state["plan"], 1):
        status_icon = {
            "pending": "⏳",
            "running": "🔄",
            "completed": "✅",
            "failed": "❌",
        }.get(step["status"], "❓")

        lines.append(f"{i}. {status_icon} [{step['tool']}] {step['description']}")

    return "\n".join(lines)
