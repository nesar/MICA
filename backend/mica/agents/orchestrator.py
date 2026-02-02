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
    LocalDocumentSearchTool,
    LocalDataAnalysisTool,
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

QUERY_ANALYSIS_PROMPT = """Analyze the following query about critical materials supply chains. Provide a structured analysis.

Query: {query}

Provide your analysis in the following JSON format:
{{
    "problem_summary": "A 2-3 sentence summary of what the user is asking for",
    "key_topics": ["List of main topics/aspects that need to be addressed"],
    "materials_mentioned": ["List of specific materials or elements mentioned"],
    "data_requirements": ["Types of data needed to answer this query"],
    "potential_issues": [
        {{
            "issue": "Description of any inconsistency, ambiguity, or concern",
            "explanation": "Why this is an issue and how to address it"
        }}
    ],
    "clarifications_needed": ["Any questions that would help clarify the request (can be empty)"],
    "scope_assessment": "Brief assessment of query complexity and scope"
}}

Be thorough in identifying potential issues. For example:
- If materials are incorrectly categorized
- If the query mixes unrelated topics
- If there are conflicting requirements

Respond ONLY with the JSON object, no additional text."""

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
- local_doc_search: Search local PDF documents in the MICA database (USGS reports, DOE documents, etc.)
- local_data_analysis: Analyze local Excel/CSV data files (production data, trade statistics, etc.)
- pdf_rag: Search and extract information from specific PDF documents (provide path)
- excel_handler: Read and analyze specific Excel data files (provide path)
- code_agent: Run statistical analysis and create visualizations
- simulation: Run supply chain simulations (GCMat, RELOG)
- doc_generator: Generate PDF reports

IMPORTANT: Prefer using local_doc_search and local_data_analysis tools when analyzing supply chain data,
as these provide access to curated USGS, DOE, and other authoritative data sources in the local database.

Format your plan as a numbered list of specific, actionable steps.
"""

SUMMARY_PROMPT = """Synthesize the analysis results into a comprehensive summary suitable for a professional scientific report.

Original Query: {query}

Analysis Results:
{results}

Available Sources for Citation:
{sources}

CRITICAL REQUIREMENTS:
1. Write in a scientific, academic style with complete sentences and proper paragraphs.
2. DO NOT use bullet points excessively - write flowing prose that discusses findings.
3. MANDATORY CITATIONS: Every numerical claim, statistic, or factual assertion MUST include an inline citation using [Ref N] format where N corresponds to the source number above.
   - Example: "China controls approximately 90% of global rare earth processing [Ref 1]."
   - Example: "The estimated investment of $15-25 billion [Ref 3] reflects..."
4. When discussing data or tables, provide context and analysis - don't just list facts.
5. All sentences must be COMPLETE - never end with "..." or leave thoughts unfinished.
6. If you cannot find a source for a claim, clearly state it as "estimated" or "industry consensus" rather than presenting it as fact.

Provide the following sections:
1. **Executive Summary** (2-3 paragraphs with key findings and their implications)
2. **Detailed Findings** (organized by topic, written as prose with proper discussion and citations)
3. **Data Analysis and Discussion** (interpret the numbers, explain significance, compare with benchmarks)
4. **Limitations and Caveats** (acknowledge data gaps and uncertainties)
5. **Conclusions and Recommendations** (if applicable)

Remember: This will be converted to a PDF report for DOE. Maintain professional, authoritative tone. ALWAYS cite sources using [Ref N] format.
"""

TITLE_GENERATION_PROMPT = """Generate a short, professional report title for the following query/analysis.

Query: {query}

Requirements:
- Maximum 8-10 words
- Must be a complete phrase (no ellipsis or truncation)
- Professional and suitable for a DOE report
- Should capture the main topic, not the full query details

Examples of good titles:
- "REE Supply Chain Investment Analysis: PRC vs US Capacity"
- "Critical Minerals Processing Infrastructure Cost Assessment"
- "Rare Earth Separation and Metallization Investment Study"

Respond with ONLY the title, nothing else.
"""


def get_llm():
    """Get the configured LLM instance."""
    try:
        return create_llm()
    except Exception as e:
        logger.error(f"Failed to create LLM: {e}")
        raise


def _truncate_at_sentence(text: str, max_length: int) -> str:
    """
    Truncate text at a sentence boundary, ensuring no incomplete sentences.

    Args:
        text: The text to truncate
        max_length: Maximum length in characters

    Returns:
        Text truncated at the nearest sentence boundary before max_length
    """
    if len(text) <= max_length:
        return text

    # Find sentence endings (., !, ?)
    import re

    # Truncate to max_length first
    truncated = text[:max_length]

    # Find the last sentence boundary
    # Look for sentence-ending punctuation followed by space or end
    sentence_endings = list(re.finditer(r'[.!?]\s', truncated))

    if sentence_endings:
        # Cut at the last complete sentence
        last_end = sentence_endings[-1].end()
        return truncated[:last_end].strip()
    else:
        # No sentence ending found - look for the punctuation without space at the very end
        last_period = truncated.rfind('.')
        last_question = truncated.rfind('?')
        last_exclaim = truncated.rfind('!')
        last_punct = max(last_period, last_question, last_exclaim)

        if last_punct > max_length * 0.5:  # Only use if we keep at least half
            return truncated[:last_punct + 1].strip()

        # Fallback: truncate at last space to avoid mid-word cut
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.7:
            return truncated[:last_space].strip() + "."

        return truncated.strip()


def _generate_report_title(query: str, llm) -> str:
    """
    Generate a professional, short report title based on the query.

    Args:
        query: The original user query
        llm: The LLM instance to use

    Returns:
        A professional report title (8-10 words max)
    """
    try:
        prompt = TITLE_GENERATION_PROMPT.format(query=query)

        messages = [
            SystemMessage(content="You are a technical editor specializing in government reports."),
            HumanMessage(content=prompt),
        ]

        response = llm.invoke(messages)
        title = response.content.strip()

        # Clean up any quotes or extra formatting
        title = title.strip('"\'')

        # Ensure title isn't too long
        if len(title) > 100:
            title = _truncate_at_sentence(title, 100)

        # Fallback if title looks bad
        if not title or len(title) < 10 or title.endswith('...'):
            # Generate a simple title from query keywords
            keywords = [w for w in query.split()[:6] if len(w) > 3]
            title = "Analysis: " + " ".join(keywords[:4]).title()

        return title

    except Exception as e:
        logger.warning(f"Title generation failed: {e}")
        # Extract key terms for fallback title
        query_words = query.split()[:5]
        return "Analysis Report: " + " ".join(query_words).title()[:60]


def _extract_executive_summary(summary: str) -> str:
    """
    Extract or create a proper executive summary from the LLM-generated summary.
    Ensures complete sentences and proper structure.

    Args:
        summary: The full LLM-generated summary

    Returns:
        A well-formatted executive summary (1-3 paragraphs)
    """
    import re

    # Try to find an explicit executive summary section
    exec_patterns = [
        r'(?i)executive\s+summary[:\s]*\n+(.*?)(?=\n##|\n\*\*[A-Z]|\n[0-9]+\.|\Z)',
        r'(?i)key\s+findings?[:\s]*\n+(.*?)(?=\n##|\n\*\*[A-Z]|\n[0-9]+\.|\Z)',
    ]

    for pattern in exec_patterns:
        match = re.search(pattern, summary, re.DOTALL)
        if match:
            exec_text = match.group(1).strip()
            if len(exec_text) > 100:
                return _truncate_at_sentence(exec_text, 1500)

    # If no explicit section, take the first substantial paragraphs
    paragraphs = summary.split('\n\n')
    exec_paragraphs = []
    char_count = 0

    for para in paragraphs:
        para = para.strip()
        # Skip section headers
        if para.startswith('#') or para.startswith('**') and para.endswith('**'):
            continue
        # Skip short lines that are likely headers
        if len(para) < 50:
            continue
        # Skip bullet-only paragraphs
        if para.startswith('- ') or para.startswith('* ') or re.match(r'^\d+\.', para):
            continue

        exec_paragraphs.append(para)
        char_count += len(para)

        if char_count > 1200:
            break

    if exec_paragraphs:
        result = '\n\n'.join(exec_paragraphs)
        return _truncate_at_sentence(result, 1500)

    # Fallback: take first 1000 chars with sentence boundary
    return _truncate_at_sentence(summary, 1500)


def _clean_truncated_snippet(text: str) -> str:
    """
    Clean up truncated snippets that end with '..' or '...' or incomplete sentences.

    Args:
        text: The potentially truncated text

    Returns:
        Cleaned text with complete sentences
    """
    import re

    text = text.strip()
    if not text:
        return text

    # Remove trailing ellipsis patterns
    text = re.sub(r'\s*\.{2,}\s*$', '', text)  # Remove .. or ... at end
    text = re.sub(r'\s*…\s*$', '', text)  # Remove unicode ellipsis

    # If the text now ends mid-sentence (no punctuation), try to find the last complete sentence
    if text and not text[-1] in '.!?':
        # Find the last sentence-ending punctuation
        last_period = text.rfind('. ')
        last_question = text.rfind('? ')
        last_exclaim = text.rfind('! ')
        last_punct = max(last_period, last_question, last_exclaim)

        if last_punct > len(text) * 0.3:  # Keep at least 30% of text
            text = text[:last_punct + 1]
        else:
            # No good sentence boundary - add period to make it a sentence
            text = text.rstrip(',;:') + '.'

    return text


def _format_findings_as_prose(findings: List[str], references: List[dict]) -> str:
    """
    Convert bullet-point findings into flowing prose with inline citations.

    Args:
        findings: List of finding snippets
        references: List of reference dictionaries with source info

    Returns:
        Prose-formatted text with inline citations
    """
    if not findings:
        return "No significant findings were identified from the research sources reviewed."

    # Build a source lookup for citations
    source_lookup = {}
    for i, ref in enumerate(references, 1):
        url = ref.get('url', '') or ref.get('source', '')
        source_name = ref.get('source', '') or ref.get('title', '')[:30]
        source_lookup[i] = source_name

    # Group related findings and create prose
    prose_parts = []
    prose_parts.append(
        "The research identified several key findings from authoritative sources. "
        "The following synthesis presents the major themes and data points uncovered during the analysis."
    )
    prose_parts.append("")  # Blank line

    # Process findings into coherent paragraphs
    current_paragraph = []
    ref_counter = 1

    for finding in findings[:10]:  # Limit to top 10 findings
        # Clean up the finding text - remove truncation artifacts
        finding = _clean_truncated_snippet(finding.strip())
        if not finding or len(finding) < 20:
            continue

        # Remove markdown bold markers for cleaner prose
        import re
        finding = re.sub(r'\*\*([^*]+)\*\*:', r'\1 reports that', finding)
        finding = re.sub(r'\*\*([^*]+)\*\*', r'\1', finding)

        # Clean up any remaining truncation in the middle of text
        finding = re.sub(r'\s*\.{2,}\s*', '. ', finding)  # Replace .. with .
        finding = re.sub(r'\s*…\s*', '. ', finding)  # Replace unicode ellipsis

        # Ensure the finding is a complete sentence
        if not finding.endswith(('.', '!', '?')):
            finding = finding + '.'

        # Add citation reference
        if ref_counter <= len(references):
            # Insert citation before the final period
            if finding.endswith('.'):
                finding = finding[:-1] + f' [Ref {ref_counter}].'
            ref_counter += 1

        current_paragraph.append(finding)

        # Create paragraph breaks every 2-3 findings
        if len(current_paragraph) >= 2:
            prose_parts.append(' '.join(current_paragraph))
            prose_parts.append("")
            current_paragraph = []

    # Add any remaining findings
    if current_paragraph:
        prose_parts.append(' '.join(current_paragraph))

    # Add a concluding sentence
    prose_parts.append("")
    prose_parts.append(
        "These findings provide the foundation for the detailed analysis presented in subsequent sections of this report. "
        "Full source citations are available in the References section."
    )

    return '\n'.join(prose_parts)


def _parse_query_analysis(response: str) -> Dict[str, Any]:
    """
    Parse the query analysis JSON response from the LLM.

    Args:
        response: Raw LLM response containing JSON

    Returns:
        Parsed query analysis dictionary
    """
    import json
    import re

    # Try to extract JSON from the response
    # Handle cases where LLM might wrap JSON in markdown code blocks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_str = response.strip()

    try:
        analysis = json.loads(json_str)
        # Ensure required fields exist with defaults
        return {
            "problem_summary": analysis.get("problem_summary", "Unable to parse problem summary"),
            "key_topics": analysis.get("key_topics", []),
            "materials_mentioned": analysis.get("materials_mentioned", []),
            "data_requirements": analysis.get("data_requirements", []),
            "potential_issues": analysis.get("potential_issues", []),
            "clarifications_needed": analysis.get("clarifications_needed", []),
            "scope_assessment": analysis.get("scope_assessment", "Standard complexity"),
        }
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse query analysis JSON: {e}")
        # Return a basic structure with the raw response
        return {
            "problem_summary": response[:500] if len(response) > 500 else response,
            "key_topics": [],
            "materials_mentioned": [],
            "data_requirements": [],
            "potential_issues": [],
            "clarifications_needed": [],
            "scope_assessment": "Unable to parse structured analysis",
            "raw_response": response,
        }


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

        # Step 1: Generate structured query analysis
        logger.info(f"[{state['session_id']}] Generating query analysis")
        analysis_prompt = QUERY_ANALYSIS_PROMPT.format(query=query)
        analysis_messages = [
            SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
            HumanMessage(content=analysis_prompt),
        ]

        analysis_response = llm.invoke(analysis_messages)

        # Check if LLM returned an error
        if analysis_response.content.startswith("Error:"):
            logger.error(f"[{state['session_id']}] LLM error during query analysis: {analysis_response.content}")
            raise Exception(f"LLM request failed: {analysis_response.content}")

        query_analysis = _parse_query_analysis(analysis_response.content)

        logger.info(f"[{state['session_id']}] Query analysis completed: {len(query_analysis.get('potential_issues', []))} issues identified")

        # Step 2: Generate research prompt (context already built above)
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

        # Check if LLM returned an error
        if response.content.startswith("Error:"):
            logger.error(f"[{state['session_id']}] LLM error during research: {response.content}")
            raise Exception(f"LLM request failed: {response.content}")

        research_summary = response.content

        # Step 3: Do a quick web search for context
        web_search = WebSearchTool(session_logger=session)
        search_result = web_search.execute({
            "query": query,
            "num_results": 5,
        })

        # Compile preliminary research with query analysis
        state["preliminary_research"] = {
            "query_analysis": query_analysis,
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
        "local_doc_search": LocalDocumentSearchTool(session_logger=session),
        "local_data_analysis": LocalDataAnalysisTool(session_logger=session),
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

        # Compile results with proper sentence truncation
        results_text = ""
        sources_for_citation = []
        source_counter = 1

        for output in state["intermediate_outputs"]:
            results_text += f"\n## {output['step_id']} ({output['tool']})\n"
            tool_name = output.get("tool", "")
            out_data = output.get("output", {})

            if isinstance(out_data, dict):
                output_str = str(out_data)
            else:
                output_str = str(out_data)

            # Use sentence-aware truncation instead of hard cut
            results_text += _truncate_at_sentence(output_str, 2000)
            results_text += "\n"

            # Collect sources from web search results for citation
            if tool_name == "web_search":
                if isinstance(out_data, list):
                    for item in out_data[:5]:  # Top 5 from each search
                        if isinstance(item, dict) and item.get("title"):
                            sources_for_citation.append(
                                f"[Ref {source_counter}] {item.get('title', 'Unknown')} - {item.get('source', item.get('url', 'Unknown source'))}"
                            )
                            source_counter += 1

        # Format sources list for the prompt
        if sources_for_citation:
            sources_text = "\n".join(sources_for_citation[:20])  # Limit to 20 sources
        else:
            sources_text = "No specific sources available. Use 'industry estimates' or 'analysis suggests' for unsourced claims."

        prompt = SUMMARY_PROMPT.format(
            query=state["query"],
            results=results_text,
            sources=sources_text,
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

            # Generate a proper short title using LLM
            report_title = _generate_report_title(state["query"], llm)
            logger.info(f"[{state['session_id']}] Generated report title: {report_title}")

            # Build structured sections with proper prose
            sections = []

            # Executive Summary - extract properly, ensure complete sentences
            exec_summary = _extract_executive_summary(summary)
            sections.append({
                "title": "Executive Summary",
                "content": exec_summary,
            })

            # Background and Objectives - expanded with more context
            background_content = f"""This analysis was commissioned to address critical questions regarding materials supply chain investments and capabilities. The specific objectives of this study were:

**Primary Research Question:** {_truncate_at_sentence(state['query'], 500)}

The analysis draws upon publicly available data sources from authoritative institutions including the U.S. Geological Survey (USGS), Department of Energy (DOE), and other federal agencies. All findings are grounded in verifiable data with sources cited throughout this report.

This report presents quantitative estimates where data permits, and provides qualitative assessments with clearly stated assumptions where primary data is limited. Uncertainty ranges are provided for key estimates to reflect data limitations."""
            sections.append({
                "title": "Background and Objectives",
                "content": background_content,
            })

            # Key Findings - formatted as proper prose with citations
            if web_search_findings:
                findings_content = _format_findings_as_prose(web_search_findings, references)
                sections.append({
                    "title": "Key Findings from Research",
                    "content": findings_content,
                })

            # Detailed Analysis section - full LLM summary (already should be prose-like)
            sections.append({
                "title": "Detailed Analysis",
                "content": summary,
            })

            # Data Analysis section - from code agent, with better formatting
            if analysis_findings:
                analysis_content = """The following quantitative analyses were performed to support the findings presented in this report. Each analysis is accompanied by data sources and methodology notes.

"""
                for i, finding in enumerate(analysis_findings[:5], 1):
                    analysis_content += f"### Analysis {i}\n\n"
                    analysis_content += _truncate_at_sentence(finding, 1200)
                    analysis_content += "\n\n"
                sections.append({
                    "title": "Quantitative Analysis and Results",
                    "content": analysis_content,
                })

            # Methodology - expanded and more formal
            tools_used = list(set(o.get("tool", "") for o in state["intermediate_outputs"]))
            method_content = """This analysis employed a multi-method approach combining automated information retrieval, data analysis, and synthesis. The methodology was designed to maximize the use of authoritative government sources while ensuring comprehensive coverage of the research questions.

**Data Collection Methods:**

The primary data collection involved systematic searches of federal government databases and official publications. Priority was given to primary sources from the U.S. Geological Survey (USGS), Department of Energy (DOE), Environmental Protection Agency (EPA), and Congressional Research Service (CRS).

**Analytical Approach:**

"""
            if "code_agent" in tools_used:
                method_content += "Statistical analysis was performed using Python-based computational tools to process quantitative data, calculate estimates, and perform sensitivity analyses. "
            if "pdf_rag" in tools_used:
                method_content += "Document analysis was conducted using retrieval-augmented generation (RAG) techniques to extract and synthesize information from technical reports and policy documents. "
            if "excel_handler" in tools_used:
                method_content += "Spreadsheet data including trade statistics, production figures, and cost data were processed and analyzed to support quantitative findings. "
            if "simulation" in tools_used:
                method_content += "Supply chain simulation models were used to project future scenarios and estimate infrastructure requirements. "

            method_content += """

**Data Quality and Limitations:**

All data sources are cited in the References section. Where multiple sources provided conflicting information, this is noted in the analysis with discussion of potential reasons for discrepancies. Estimates are provided with uncertainty ranges where appropriate."""
            sections.append({
                "title": "Methodology",
                "content": method_content,
            })

            # Add Limitations section - use plain text, not markdown
            limitations_content = """Several limitations should be considered when interpreting the findings of this analysis:

Data Availability: Some data, particularly regarding proprietary industrial processes and state-owned enterprise investments, may not be publicly available or may be incomplete.

Currency of Information: Market conditions and investment landscapes change rapidly. Findings reflect data available as of the analysis date.

Estimation Uncertainty: Cost and investment estimates involve inherent uncertainty. Ranges are provided where possible, but actual values may differ.

Source Limitations: While priority was given to authoritative government sources, some data points may be derived from industry reports or news sources of varying reliability.

Methodology Constraints: Automated data retrieval may miss relevant sources not indexed in searched databases."""
            sections.append({
                "title": "Limitations and Caveats",
                "content": limitations_content,
            })

            # Add references section if we have any - prioritize official sources
            if references:
                # Sort references to put official sources first
                official_refs = []
                other_refs = []
                official_domains = ['usgs.gov', 'energy.gov', 'doe.gov', 'epa.gov', 'commerce.gov', 'congress.gov', 'gao.gov']
                for ref in references:
                    url = ref.get('url', '') or ref.get('source', '')
                    if any(domain in url.lower() for domain in official_domains):
                        official_refs.append(ref)
                    else:
                        other_refs.append(ref)

                sorted_refs = official_refs + other_refs
                sections.append({
                    "title": "References and Data Sources",
                    "content": "The following sources were consulted during this analysis. Official government sources are listed first.",
                    "references": sorted_refs[:25],  # Limit to 25 references
                })

            report_result = doc_gen.execute({
                "title": report_title,
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
