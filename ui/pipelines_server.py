"""
MICA Pipelines Server for Open WebUI

This server implements the Open WebUI Pipelines protocol.
Run this alongside the MICA backend, then add the URL to Open WebUI.

Usage:
    python pipelines_server.py

Then in Open WebUI:
    Admin → Settings → Pipelines → Add: http://localhost:9099
"""

import os
import json
import asyncio
from typing import Optional, List, Dict, Any, AsyncGenerator
from datetime import datetime

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Configuration
MICA_BACKEND_URL = os.getenv("MICA_BACKEND_URL", "http://localhost:8000")
PIPELINES_PORT = int(os.getenv("PIPELINES_PORT", "9099"))

app = FastAPI(title="MICA Pipelines Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session tracking
sessions: Dict[str, str] = {}  # chat_id -> mica_session_id


def format_step_output(tool: str, output: Any) -> str:
    """Format step output for display based on tool type."""
    if not output:
        return ""

    try:
        if tool == "web_search":
            # Show search result summaries
            if isinstance(output, list) and output:
                results = []
                for item in output[:3]:  # Show top 3 results
                    if isinstance(item, dict):
                        title = item.get("title", "")[:60]
                        snippet = item.get("snippet", item.get("description", ""))[:100]
                        if title:
                            results.append(f"  - {title}")
                            if snippet:
                                results.append(f"    {snippet}...")
                return "\n".join(results) if results else ""
            elif isinstance(output, str):
                return f"  {output[:200]}..." if len(output) > 200 else f"  {output}"

        elif tool == "code_agent":
            # Show code execution summary
            if isinstance(output, dict):
                result = output.get("result", output.get("output", ""))
                if isinstance(result, str) and result:
                    return f"  Result: {result[:150]}..." if len(result) > 150 else f"  Result: {result}"
            elif isinstance(output, str):
                return f"  {output[:150]}..." if len(output) > 150 else f"  {output}"

        elif tool == "pdf_rag":
            # Show extracted content preview
            if isinstance(output, dict):
                content = output.get("content", output.get("text", ""))
                if content:
                    return f"  Extracted: {content[:150]}..."
            elif isinstance(output, str):
                return f"  {output[:150]}..."

        elif tool == "doc_generator":
            # Show generated document info
            if isinstance(output, dict):
                path = output.get("path", "")
                if path:
                    return f"  Generated: {path}"
            elif isinstance(output, str):
                return f"  {output[:100]}"

        elif tool == "orchestrator":
            # Show LLM analysis summary
            if isinstance(output, str) and output:
                # Get first meaningful line
                lines = [l.strip() for l in output.split("\n") if l.strip() and not l.startswith("#")]
                if lines:
                    preview = lines[0][:150]
                    return f"  {preview}..." if len(lines[0]) > 150 else f"  {preview}"

        # Generic fallback
        if isinstance(output, str) and output:
            return f"  {output[:100]}..." if len(output) > 100 else f"  {output}"
        elif isinstance(output, dict):
            return f"  {str(output)[:100]}..."

    except Exception:
        pass

    return ""


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = True


# Pipelines protocol endpoints
@app.get("/")
async def root():
    return {"status": "ok", "name": "MICA Pipelines Server"}


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    """Return available models (pipelines)."""
    return {
        "data": [
            {
                "id": "mica-analyst",
                "name": "MICA - Materials Intelligence Co-Analyst",
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "mica",
            }
        ]
    }


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    """Handle chat completions - main entry point."""

    if request.stream:
        return StreamingResponse(
            stream_response(request),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming response
        content = await process_message(request)
        return {
            "id": "mica-response",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }]
        }


async def stream_response(request: ChatRequest) -> AsyncGenerator[str, None]:
    """Stream the response."""

    async for chunk in process_message_stream(request):
        data = {
            "id": "mica-response",
            "object": "chat.completion.chunk",
            "created": int(datetime.now().timestamp()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {"content": chunk},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(data)}\n\n"

    # Send done
    yield "data: [DONE]\n\n"


def extract_session_id_from_messages(messages: List[Message]) -> Optional[str]:
    """Extract MICA session ID from the MOST RECENT assistant message only."""
    import re
    # Only look at the last assistant message to avoid stale sessions
    for msg in reversed(messages):
        if msg.role == "assistant":
            # Look for session ID pattern like "Session: `abc123...`"
            match = re.search(r'Session:\s*`?([a-f0-9-]{8,36})', msg.content)
            if match:
                return match.group(1)
            # If the last assistant message doesn't have a session ID, return None
            # (don't search older messages - they're stale)
            return None
    return None


def is_new_query(message: str, has_prior_session: bool) -> bool:
    """Detect if a message looks like a new query rather than a follow-up.

    Be VERY conservative - only start new session if clearly changing topic.
    Default to treating messages as follow-ups when we have a prior session.
    """
    if not has_prior_session:
        return True  # No prior session = definitely new query

    # If we have a prior session, almost everything is a follow-up
    # Only start new session for explicit topic changes
    msg_lower = message.lower().strip()

    # Explicit new topic patterns - completely different subjects
    new_topic_patterns = [
        "let's talk about ", "new question:", "different topic:",
        "changing subject", "unrelated question",
    ]

    return any(p in msg_lower for p in new_topic_patterns)


def get_first_user_query(messages: List[Message]) -> str:
    """Get the first user message to use as stable chat identifier."""
    for msg in messages:
        if msg.role == "user":
            return msg.content.strip()
    return ""


def is_system_query(message: str) -> bool:
    """Check if this is an Open WebUI system query (title/tag generation) that should be ignored."""
    system_patterns = [
        "### Task:",
        "Generate 1-3 broad tags",
        "Create a concise, 3-5 word title",
        "as a title for the chat history",
        "categorizing the main themes",
    ]
    return any(pattern.lower() in message.lower() for pattern in system_patterns)


async def process_message_stream(request: ChatRequest) -> AsyncGenerator[str, None]:
    """Process the message and yield response chunks."""

    messages = request.messages
    if not messages:
        yield "Please ask a question about critical materials supply chains."
        return

    user_message = messages[-1].content.strip()

    # Skip Open WebUI's automatic title/tag generation queries
    if is_system_query(user_message):
        yield "MICA Analysis System"
        return

    # Generate a stable chat ID from the first user query (not changing each message)
    first_query = get_first_user_query(messages)
    chat_id = str(hash(first_query))[:12] if first_query else "default"

    # Also try to extract session ID from previous assistant messages
    extracted_session_id = extract_session_id_from_messages(messages)

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            # Check if we have an active session (prefer extracted, then stored)
            session_id = extracted_session_id or sessions.get(chat_id)

            # Check if this looks like a completely new query
            has_prior_session = session_id is not None
            if session_id and is_new_query(user_message, has_prior_session):
                # Clear the stale session - user is starting fresh
                if chat_id in sessions:
                    del sessions[chat_id]
                session_id = None

            if session_id:
                # Validate the session still exists in the backend
                try:
                    status_res = await client.get(f"{MICA_BACKEND_URL}/api/v1/session/{session_id}")
                    if status_res.status_code != 200:
                        # Session doesn't exist in backend, clear it
                        if chat_id in sessions:
                            del sessions[chat_id]
                        session_id = None
                except:
                    session_id = None

            if session_id:
                # Ensure session is tracked in our local dict
                sessions[chat_id] = session_id

                # Check session status
                try:
                    status_res = await client.get(f"{MICA_BACKEND_URL}/api/v1/session/{session_id}")
                    if status_res.status_code == 200:
                        status_data = status_res.json()
                        current_status = status_data.get("status", "")

                        # Handle approval
                        if current_status in ["plan_proposed", "awaiting_approval"]:
                            lower_msg = user_message.lower()
                            if any(w in lower_msg for w in ["yes", "approve", "ok", "proceed", "go"]):
                                await client.post(
                                    f"{MICA_BACKEND_URL}/api/v1/session/{session_id}/approve",
                                    json={"session_id": session_id, "approved": True}
                                )
                                yield "**Plan approved.** Executing analysis...\n\n"
                                yield "<details>\n<summary>Execution Progress (click to expand)</summary>\n\n```\n"

                                # Poll for execution results
                                shown_steps = set()

                                for i in range(180):  # 30 minutes max (180 * 10s)
                                    await asyncio.sleep(10)

                                    exec_res = await client.get(f"{MICA_BACKEND_URL}/api/v1/session/{session_id}")
                                    exec_data = exec_res.json()
                                    exec_status = exec_data.get("status", "")
                                    plan = exec_data.get("plan", [])

                                    if exec_status in ["completed", "awaiting_feedback"]:
                                        # Close progress section
                                        yield "```\n</details>\n\n---\n\n"
                                        summary = exec_data.get("final_summary", "Analysis complete.")
                                        yield f"## Results\n\n{summary}"

                                        # Check for plots/artifacts
                                        try:
                                            artifacts_res = await client.get(f"{MICA_BACKEND_URL}/api/v1/session/{session_id}/artifacts")
                                            if artifacts_res.status_code == 200:
                                                artifacts = artifacts_res.json().get("artifacts", {})
                                                plots = artifacts.get("plots", [])
                                                if plots:
                                                    yield "\n\n### Generated Visualizations\n\n"
                                                    for plot in plots:
                                                        plot_url = f"{MICA_BACKEND_URL}/api/v1/session/{session_id}/artifact/plots/{plot}"
                                                        yield f"![{plot}]({plot_url})\n\n"
                                        except Exception:
                                            pass  # Artifacts optional

                                        yield f"\n\n---\n*Session: `{session_id}` - Ask a follow-up question or say 'done' to finish.*"
                                        return

                                    elif exec_status == "failed":
                                        yield "```\n</details>\n\n"
                                        errors = exec_data.get("errors", [])
                                        yield f"\n\nExecution failed: {errors[0].get('error') if errors else 'Unknown error'}"
                                        return

                                    # Show progress in compact format (inside code block)
                                    completed_steps = [s for s in plan if s.get("status") == "completed"]
                                    running_steps = [s for s in plan if s.get("status") == "running"]

                                    for step in completed_steps:
                                        step_id = step.get("step_id", "")
                                        if step_id not in shown_steps:
                                            shown_steps.add(step_id)
                                            tool = step.get("tool", "")
                                            desc = step.get("description", "")[:60]
                                            yield f"[done] {desc} ({tool})\n"

                                    if running_steps and i % 3 == 0:
                                        current = running_steps[0]
                                        desc = current.get("description", "")[:50]
                                        tool = current.get("tool", "")
                                        progress = f"{len(completed_steps)}/{len(plan)}"
                                        yield f"[running] {desc}... ({progress})\n"

                                yield "```\n</details>\n\nExecution is taking longer than expected. Check session status manually."
                                return
                            elif any(w in lower_msg for w in ["no", "reject", "cancel"]):
                                await client.post(
                                    f"{MICA_BACKEND_URL}/api/v1/session/{session_id}/approve",
                                    json={"session_id": session_id, "approved": False}
                                )
                                if chat_id in sessions:
                                    del sessions[chat_id]
                                yield "Plan rejected. Feel free to ask a new question."
                                return

                        # Handle follow-up questions when session is awaiting feedback
                        elif current_status == "awaiting_feedback":
                            lower_msg = user_message.lower()
                            # Check if user wants to end the session
                            if lower_msg in ["done", "finish", "end", "thanks", "thank you"]:
                                await client.post(
                                    f"{MICA_BACKEND_URL}/api/v1/session/{session_id}/feedback",
                                    json={"session_id": session_id, "feedback": user_message}
                                )
                                if chat_id in sessions:
                                    del sessions[chat_id]
                                yield "Session completed. Feel free to start a new query!"
                                return
                            else:
                                # Store the original summary to detect when it changes
                                original_summary = status_data.get("final_summary", "") or ""
                                original_summary_hash = hash(original_summary)

                                # Submit as follow-up query
                                yield f"**Processing follow-up...**\n\n"

                                feedback_res = await client.post(
                                    f"{MICA_BACKEND_URL}/api/v1/session/{session_id}/feedback",
                                    json={
                                        "session_id": session_id,
                                        "feedback": user_message,
                                        "follow_up_query": user_message
                                    }
                                )

                                if feedback_res.status_code != 200:
                                    yield f"Error submitting follow-up: {feedback_res.text}"
                                    return

                                # Poll until the summary changes or we hit a different status
                                found_new_answer = False
                                shown_steps = set()

                                for i in range(180):  # 15 minutes max (180 * 5s)
                                    await asyncio.sleep(5)

                                    status_res = await client.get(f"{MICA_BACKEND_URL}/api/v1/session/{session_id}")
                                    status_data = status_res.json()
                                    status = status_data.get("status", "")
                                    current_summary = status_data.get("final_summary", "") or ""
                                    plan = status_data.get("plan", [])

                                    if status in ["plan_proposed", "awaiting_approval"]:
                                        yield "## Analysis Plan Generated\n\n"
                                        for j, step in enumerate(plan):
                                            yield f"**Step {j+1}:** {step.get('description', 'N/A')}\n"
                                            yield f"- Tool: `{step.get('tool', 'N/A')}`\n\n"
                                        yield "---\n\n"
                                        yield "**Reply 'approve' to execute this plan, or 'reject' to cancel.**"
                                        return

                                    elif status in ["awaiting_feedback", "completed"]:
                                        # Check if summary changed (use hash for reliable comparison)
                                        if current_summary and hash(current_summary) != original_summary_hash:
                                            found_new_answer = True
                                            yield current_summary

                                            # Check for plots/artifacts
                                            try:
                                                artifacts_res = await client.get(f"{MICA_BACKEND_URL}/api/v1/session/{session_id}/artifacts")
                                                if artifacts_res.status_code == 200:
                                                    artifacts = artifacts_res.json().get("artifacts", {})
                                                    plots = artifacts.get("plots", [])
                                                    if plots:
                                                        yield "\n\n### Generated Visualizations\n\n"
                                                        for plot in plots:
                                                            plot_url = f"{MICA_BACKEND_URL}/api/v1/session/{session_id}/artifact/plots/{plot}"
                                                            yield f"![{plot}]({plot_url})\n\n"
                                            except Exception:
                                                pass  # Artifacts optional

                                            yield f"\n\n---\n*Session: `{session_id}` - Ask a follow-up question or say 'done' to finish.*"
                                            return

                                    elif status == "executing":
                                        # Show progress in compact format
                                        completed_steps = [s for s in plan if s.get("status") == "completed"]
                                        running_steps = [s for s in plan if s.get("status") == "running"]

                                        for step in completed_steps:
                                            step_id = step.get("step_id", "")
                                            if step_id not in shown_steps:
                                                shown_steps.add(step_id)
                                                tool = step.get("tool", "")
                                                desc = step.get("description", "")[:60]
                                                yield f"> [done] {desc} ({tool})\n"

                                        if running_steps and i % 4 == 0:
                                            current = running_steps[0]
                                            desc = current.get("description", "")[:50]
                                            progress = f"{len(completed_steps)}/{len(plan)}"
                                            yield f"> [running] {desc}... ({progress})\n"

                                    elif status == "failed":
                                        errors = status_data.get("errors", [])
                                        yield f"Analysis failed: {errors[0].get('error') if errors else 'Unknown error'}"
                                        return

                                    # Show basic progress if no plan updates
                                    elif i > 0 and i % 12 == 0:
                                        yield f"Still analyzing... ({i*5}s, status: {status})\n"

                                # Timeout - show whatever summary we have
                                if not found_new_answer:
                                    status_res = await client.get(f"{MICA_BACKEND_URL}/api/v1/session/{session_id}")
                                    status_data = status_res.json()
                                    final_summary = status_data.get("final_summary", "")
                                    if final_summary:
                                        yield final_summary
                                        yield f"\n\n---\n*Session: `{session_id}` - Ask a follow-up question or say 'done' to finish.*"
                                    else:
                                        yield "Analysis is taking longer than expected. Please check the session status."
                                return

                except Exception as e:
                    # Log the error but continue to submit as new query
                    pass

            # Check credentials
            creds_res = await client.get(f"{MICA_BACKEND_URL}/api/v1/credentials")
            creds_data = creds_res.json()

            if creds_data.get("requires_setup"):
                yield "**Credentials Required**\n\n"
                yield "Before using MICA, please set up your LLM credentials:\n\n"
                yield "1. Open http://localhost:8000/docs\n"
                yield "2. Use `POST /api/v1/credentials` with:\n"
                yield "```json\n{\"provider\": \"argo\", \"credential\": \"YOUR_USERNAME\"}\n```\n\n"
                yield "Then come back and ask your question again."
                return

            # Submit new query
            yield "**Submitting query to MICA...**\n\n"

            query_res = await client.post(
                f"{MICA_BACKEND_URL}/api/v1/query",
                json={"query": user_message, "user_id": "webui_user"}
            )
            query_data = query_res.json()
            session_id = query_data.get("session_id")
            sessions[chat_id] = session_id

            yield f"Session: `{session_id}`\n\n"
            yield "Conducting preliminary research and generating analysis plan...\n\n"

            # Poll for plan generation
            for i in range(60):  # 10 minutes max for complex queries
                await asyncio.sleep(10)

                status_res = await client.get(f"{MICA_BACKEND_URL}/api/v1/session/{session_id}")
                status_data = status_res.json()
                status = status_data.get("status", "")
                current_step = status_data.get("current_step", "")

                if status in ["plan_proposed", "awaiting_approval"]:
                    plan = status_data.get("plan", [])

                    yield "## Analysis Plan Generated\n\n"

                    for j, step in enumerate(plan):
                        yield f"**Step {j+1}:** {step.get('description', 'N/A')}\n"
                        yield f"- Tool: `{step.get('tool', 'N/A')}`\n\n"

                    yield "---\n\n"
                    yield "**Reply 'approve' to execute this plan, or 'reject' to cancel.**"
                    return

                elif status == "failed":
                    errors = status_data.get("errors", [])
                    yield f"Analysis failed: {errors[0].get('error') if errors else 'Unknown error'}"
                    return

                elif status in ["completed", "awaiting_feedback"]:
                    summary = status_data.get("final_summary", "Analysis complete.")
                    yield summary

                    # Check for plots/artifacts
                    try:
                        artifacts_res = await client.get(f"{MICA_BACKEND_URL}/api/v1/session/{session_id}/artifacts")
                        if artifacts_res.status_code == 200:
                            artifacts = artifacts_res.json().get("artifacts", {})
                            plots = artifacts.get("plots", [])
                            if plots:
                                yield "\n\n### Generated Visualizations\n\n"
                                for plot in plots:
                                    plot_url = f"{MICA_BACKEND_URL}/api/v1/session/{session_id}/artifact/plots/{plot}"
                                    yield f"![{plot}]({plot_url})\n\n"
                    except Exception:
                        pass  # Artifacts optional

                    # Keep session for follow-ups (don't delete from sessions dict)
                    yield f"\n\n---\n*Session: `{session_id}` - Ask a follow-up question or say 'done' to finish.*"
                    return

                if i % 3 == 0 and i > 0:
                    yield f"Still working... (status: {status})\n"

            yield "Analysis is taking longer than expected. Check back later."

        except httpx.ConnectError:
            yield f"Cannot connect to MICA backend at {MICA_BACKEND_URL}"
        except Exception as e:
            yield f"Error: {str(e)}"


async def process_message(request: ChatRequest) -> str:
    """Non-streaming version."""
    chunks = []
    async for chunk in process_message_stream(request):
        chunks.append(chunk)
    return "".join(chunks)


if __name__ == "__main__":
    import uvicorn
    print(f"Starting MICA Pipelines Server on port {PIPELINES_PORT}")
    print(f"MICA Backend URL: {MICA_BACKEND_URL}")
    print(f"\nAdd this URL to Open WebUI Pipelines: http://localhost:{PIPELINES_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PIPELINES_PORT)
