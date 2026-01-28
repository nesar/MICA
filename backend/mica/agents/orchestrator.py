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


def is_simple_query(query: str) -> bool:
    """
    Detect if a query is simple and can be answered directly without full planning.

    Simple queries include:
    - Questions about MICA's capabilities/tools
    - Basic definitions or explanations
    - Simple factual questions that don't require multi-step analysis
    """
    query_lower = query.lower().strip()

    # Queries about MICA itself
    mica_queries = [
        "what tools", "available tools", "list tools", "show tools",
        "what can you do", "your capabilities", "how do you work",
        "what are you", "who are you", "help", "what is mica",
    ]
    if any(pattern in query_lower for pattern in mica_queries):
        return True

    # Simple definition questions
    if query_lower.startswith(("what is ", "define ", "explain ")) and len(query.split()) < 10:
        return True

    # Very short queries (likely simple)
    if len(query.split()) < 5 and "?" in query:
        return True

    return False


def answer_simple_query(query: str, llm, context: str = "") -> str:
    """
    Generate a direct answer for simple queries without multi-step planning.

    Args:
        query: The user's query
        llm: The LLM instance to use
        context: Previous conversation context for follow-up questions
    """
    # Special handling for tool queries
    query_lower = query.lower()
    if any(w in query_lower for w in ["tool", "capabilities", "can you do"]):
        return """# MICA Available Tools

MICA has access to the following analytical tools:

## 1. Web Search (`web_search`)
Searches the web with focus on federal government documents (DOE, USGS, EPA, etc.)
- Use for: Finding official reports, statistics, policy documents
- Data sources: energy.gov, usgs.gov, commerce.gov, epa.gov

## 2. PDF/Document Analysis (`pdf_rag`)
Extracts and analyzes information from PDF documents using RAG
- Use for: Analyzing uploaded reports, extracting specific data from documents

## 3. Excel Handler (`excel_handler`)
Reads and processes Excel/CSV data files
- Use for: Analyzing spreadsheet data, trade statistics, production numbers

## 4. Code Agent (`code_agent`)
Executes Python code for data analysis and visualization
- Use for: Statistical analysis, creating charts, data processing

## 5. Document Generator (`doc_generator`)
Creates formatted PDF reports
- Use for: Generating final analysis reports with visualizations

## 6. Simulation (`simulation`) [Limited]
Placeholder for supply chain simulation models
- Future: GCMat, RELOG integration

---
For complex supply chain analysis, MICA will create a detailed plan and ask for your approval before executing."""

    # For other simple queries, use LLM directly
    from langchain_core.messages import SystemMessage, HumanMessage

    # Build prompt with context if available
    context_section = ""
    if context:
        context_section = f"""
IMPORTANT - Previous conversation context:
{context}

The user is asking a follow-up question. Use the context above to understand what they're referring to.
"""

    prompt = f"""Answer this question directly and concisely. This is a simple informational query.
{context_section}
Current Question: {query}

Provide a clear, helpful answer. If this relates to critical materials or supply chains,
include relevant context. Keep the response focused and under 500 words.

If this is a follow-up question, make sure to answer in the context of the previous discussion."""

    messages = [
        SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    return response.content


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
    1. Checks if query is simple (can be answered directly)
    2. For complex queries: performs research and prepares for planning
    3. For simple queries: generates direct answer and marks for fast-track
    """
    logger.info(f"[{state['session_id']}] Conducting preliminary research")

    session = get_session_logger(state)
    query = state["query"]

    try:
        llm = get_llm()

        # Build context from any previous interactions (for follow-ups)
        context = ""
        if state["clarifications"]:
            context = "\n".join(state["clarifications"])

        # Check if this is a simple query that doesn't need full planning
        if is_simple_query(query):
            logger.info(f"[{state['session_id']}] Simple query detected - fast-track response")

            # Generate direct answer with context for follow-ups
            direct_answer = answer_simple_query(query, llm, context)

            # Mark as simple query in metadata
            state["metadata"]["simple_query"] = True
            state["final_summary"] = direct_answer

            # Set preliminary research
            state["preliminary_research"] = {
                "summary": direct_answer,
                "web_search_results": [],
                "timestamp": datetime.utcnow().isoformat(),
                "simple_query": True,
            }

            # Update messages
            state["messages"].append(HumanMessage(content=query))
            state["messages"].append(AIMessage(content=direct_answer))

            # Log
            if session:
                with session.agent_logger("orchestrator") as agent_log:
                    agent_log.log("Simple query - direct response generated")
                    agent_log.log_output({"response": direct_answer[:500]})

            # Pre-set status for the interrupt before await_feedback
            # This ensures the status is correct even if workflow pauses before await_feedback_node runs
            state["status"] = WorkflowStatus.AWAITING_FEEDBACK
            state["current_step"] = "awaiting_feedback"

            logger.info(f"[{state['session_id']}] Simple query answered directly")
            return state

        # Complex query - proceed with full research
        logger.info(f"[{state['session_id']}] Complex query - proceeding with full research")

        # Generate research prompt (context already built above)
        prompt = RESEARCH_PROMPT.format(
            query=query,
            context=context or "No previous context.",
        )

        # Get LLM response
        messages = [
            SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = llm.invoke(messages)
        research_summary = response.content

        # Do a quick web search for context
        web_search = WebSearchTool(session_logger=session)
        search_result = web_search.execute({
            "query": query,
            "num_results": 5,
        })

        # Compile preliminary research
        state["preliminary_research"] = {
            "summary": research_summary,
            "web_search_results": search_result.data if search_result.data else [],
            "timestamp": datetime.utcnow().isoformat(),
            "simple_query": False,
        }

        # Update messages
        state["messages"].append(HumanMessage(content=query))
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
                elif result.error and "is required" in str(result.error):
                    # Tool requires specific inputs we don't have - fall back to LLM
                    logger.info(f"[{state['session_id']}] Tool {tool_name} missing inputs, falling back to LLM")
                    llm_result = _execute_with_llm(state, step, session)
                    # Mark as completed with LLM result
                    state = update_plan_step(state, step_id, "completed", output=llm_result)
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

        # Always generate a report when we have a summary
        num_outputs = len(state["intermediate_outputs"])
        logger.info(f"[{state['session_id']}] Generating PDF report (intermediate_outputs: {num_outputs})")

        try:
            doc_gen = DocumentGeneratorTool(session_logger=session)

            # Collect references from web search results
            references = []
            web_search_findings = []
            analysis_findings = []
            data_tables = []

            for output in state["intermediate_outputs"]:
                tool = output.get("tool", "")
                out_data = output.get("output", {})

                if tool == "web_search":
                    # Extract references from web search
                    if isinstance(out_data, list):
                        for item in out_data:
                            if isinstance(item, dict):
                                ref = {
                                    "title": item.get("title", ""),
                                    "source": item.get("source", item.get("url", "")),
                                    "url": item.get("url", item.get("link", "")),
                                    "date": datetime.utcnow().strftime("%Y-%m-%d"),
                                }
                                if ref["title"] and ref not in references:
                                    references.append(ref)
                                # Collect findings
                                snippet = item.get("snippet", item.get("description", ""))
                                if snippet:
                                    web_search_findings.append(f"**{item.get('title', 'Source')}**: {snippet}")
                    elif isinstance(out_data, dict):
                        results = out_data.get("results", [])
                        for item in results:
                            if isinstance(item, dict):
                                ref = {
                                    "title": item.get("title", ""),
                                    "source": item.get("source", ""),
                                    "url": item.get("url", item.get("link", "")),
                                    "date": datetime.utcnow().strftime("%Y-%m-%d"),
                                }
                                if ref["title"] and ref not in references:
                                    references.append(ref)

                elif tool == "code_agent":
                    # Extract analysis results
                    if isinstance(out_data, dict):
                        result = out_data.get("result", out_data.get("output", ""))
                        if result:
                            analysis_findings.append(str(result)[:1500])

            # Build structured sections
            sections = []

            # Executive Summary - first paragraph of LLM summary
            exec_summary = summary.split("\n\n")[0] if "\n\n" in summary else summary[:800]
            sections.append({
                "title": "Executive Summary",
                "content": exec_summary,
            })

            # Background - from query context
            sections.append({
                "title": "Background and Objectives",
                "content": f"This analysis was conducted in response to the following query:\n\n**Query:** {state['query']}\n\nThe objective was to provide comprehensive analysis using publicly available data sources including USGS, DOE, and other federal resources.",
            })

            # Key Findings - from web search
            if web_search_findings:
                findings_content = "Based on research from authoritative sources:\n\n"
                findings_content += "\n\n".join(web_search_findings[:10])
                sections.append({
                    "title": "Key Findings from Research",
                    "content": findings_content,
                })

            # Analysis section - full LLM summary
            sections.append({
                "title": "Detailed Analysis",
                "content": summary,
            })

            # Data Analysis section - from code agent
            if analysis_findings:
                analysis_content = "Quantitative analysis results:\n\n"
                analysis_content += "\n\n---\n\n".join(analysis_findings[:5])
                sections.append({
                    "title": "Data Analysis Results",
                    "content": analysis_content,
                })

            # Methodology
            tools_used = list(set(o.get("tool", "") for o in state["intermediate_outputs"]))
            method_content = "This analysis employed the following methods and tools:\n\n"
            method_content += "- **Web Search**: Federal document retrieval from USGS, DOE, and other authoritative sources\n"
            if "code_agent" in tools_used:
                method_content += "- **Statistical Analysis**: Python-based data analysis and visualization\n"
            if "pdf_rag" in tools_used:
                method_content += "- **Document Analysis**: Extraction and analysis of PDF reports\n"
            if "excel_handler" in tools_used:
                method_content += "- **Data Processing**: Excel and CSV data analysis\n"
            method_content += "\nAll data sources are cited in the References section."
            sections.append({
                "title": "Methodology",
                "content": method_content,
            })

            # Add references section if we have any
            if references:
                sections.append({
                    "title": "References and Data Sources",
                    "content": "The following sources were consulted during this analysis:",
                    "references": references[:20],  # Limit to 20 references
                })

            report_result = doc_gen.execute({
                "title": state["query"][:80] if len(state["query"]) <= 80 else state["query"][:77] + "...",
                "sections": sections,
                "metadata": {
                    "Session ID": state["session_id"],
                    "Analysis Date": datetime.utcnow().strftime("%B %d, %Y"),
                    "Prepared by": "MICA - Materials Intelligence Co-Analyst",
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
                logger.info(f"[{state['session_id']}] PDF report generated: {report_result.data['path']}")
            else:
                logger.warning(f"[{state['session_id']}] Report generation returned no path: {report_result}")

        except Exception as report_error:
            logger.error(f"[{state['session_id']}] PDF report generation failed: {report_error}")
            import traceback
            logger.error(traceback.format_exc())

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
        logger.info(f"[{state['session_id']}] Processing follow-up query: {follow_up_query[:100]}")

        # Preserve context from previous conversation
        previous_query = state.get("query", "")
        previous_summary = state.get("final_summary", "")

        # Add previous conversation to clarifications for context
        if previous_query:
            state["clarifications"].append(f"Previous query: {previous_query}")
        if previous_summary:
            # Truncate summary to avoid overwhelming the context
            summary_preview = previous_summary[:1000] + "..." if len(previous_summary) > 1000 else previous_summary
            state["clarifications"].append(f"Previous response summary: {summary_preview}")

        # Set the new follow-up as the current query
        state["query"] = follow_up_query

        # Reset only execution-related state, preserve context
        state["plan"] = []
        state["tool_results"] = {}
        state["intermediate_outputs"] = []
        state["approved"] = None
        state["approval_feedback"] = None
        state["errors"] = []

        # Clear the follow_up_query to prevent re-processing
        state["metadata"]["follow_up_query"] = None
        state["metadata"]["feedback"] = None

    return state


def get_plan_summary(state: AgentState) -> str:
    """Get a human-readable summary of the current plan."""
    if not state["plan"]:
        return "No plan generated yet."

    lines = ["## Analysis Plan\n"]

    for i, step in enumerate(state["plan"], 1):
        status_icon = {
            "pending": "[ ]",
            "running": "[~]",
            "completed": "[x]",
            "failed": "[!]",
        }.get(step["status"], "[?]")

        lines.append(f"{i}. {status_icon} [{step['tool']}] {step['description']}")

    return "\n".join(lines)
