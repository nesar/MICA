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
    """Extract MICA session ID from previous assistant messages."""
    import re
    for msg in reversed(messages):
        if msg.role == "assistant":
            # Look for session ID pattern like "Session: `abc123...`"
            match = re.search(r'Session:\s*`?([a-f0-9-]{8,36})', msg.content)
            if match:
                return match.group(1)
    return None


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
                                yield "✅ **Plan approved!** Executing analysis...\n\n"

                                # Poll for execution results
                                for i in range(60):  # 10 minutes max
                                    await asyncio.sleep(10)

                                    exec_res = await client.get(f"{MICA_BACKEND_URL}/api/v1/session/{session_id}")
                                    exec_data = exec_res.json()
                                    exec_status = exec_data.get("status", "")

                                    if exec_status in ["completed", "awaiting_feedback"]:
                                        summary = exec_data.get("final_summary", "Analysis complete.")
                                        if chat_id in sessions:
                                            del sessions[chat_id]
                                        yield f"\n\n## 📊 Results\n\n{summary}"
                                        return

                                    elif exec_status == "failed":
                                        errors = exec_data.get("errors", [])
                                        yield f"\n\n❌ Execution failed: {errors[0].get('error') if errors else 'Unknown error'}"
                                        return

                                    if i % 3 == 0 and i > 0:
                                        current_step = exec_data.get("current_step", "working")
                                        yield f"⏳ Still executing... (step: {current_step})\n"

                                yield "\n⚠️ Execution is taking longer than expected. Check back later."
                                return
                            elif any(w in lower_msg for w in ["no", "reject", "cancel"]):
                                await client.post(
                                    f"{MICA_BACKEND_URL}/api/v1/session/{session_id}/approve",
                                    json={"session_id": session_id, "approved": False}
                                )
                                if chat_id in sessions:
                                    del sessions[chat_id]
                                yield "❌ Plan rejected. Feel free to ask a new question."
                                return
                except:
                    pass

            # Check credentials
            creds_res = await client.get(f"{MICA_BACKEND_URL}/api/v1/credentials")
            creds_data = creds_res.json()

            if creds_data.get("requires_setup"):
                yield "⚠️ **Credentials Required**\n\n"
                yield "Before using MICA, please set up your LLM credentials:\n\n"
                yield "1. Open http://localhost:8000/docs\n"
                yield "2. Use `POST /api/v1/credentials` with:\n"
                yield "```json\n{\"provider\": \"argo\", \"credential\": \"YOUR_USERNAME\"}\n```\n\n"
                yield "Then come back and ask your question again."
                return

            # Submit new query
            yield "🔍 **Submitting query to MICA...**\n\n"

            query_res = await client.post(
                f"{MICA_BACKEND_URL}/api/v1/query",
                json={"query": user_message, "user_id": "webui_user"}
            )
            query_data = query_res.json()
            session_id = query_data.get("session_id")
            sessions[chat_id] = session_id

            yield f"📋 Session: `{session_id}`\n\n"
            yield "⏳ Conducting preliminary research and generating analysis plan...\n\n"

            # Poll for plan
            for i in range(30):  # 5 minutes max
                await asyncio.sleep(10)

                status_res = await client.get(f"{MICA_BACKEND_URL}/api/v1/session/{session_id}")
                status_data = status_res.json()
                status = status_data.get("status", "")

                if status in ["plan_proposed", "awaiting_approval"]:
                    plan = status_data.get("plan", [])

                    yield "## 📊 Analysis Plan Generated\n\n"

                    for j, step in enumerate(plan):
                        yield f"**Step {j+1}:** {step.get('description', 'N/A')}\n"
                        yield f"- Tool: `{step.get('tool', 'N/A')}`\n\n"

                    yield "---\n\n"
                    yield "**Reply 'approve' to execute this plan, or 'reject' to cancel.**"
                    return

                elif status == "failed":
                    errors = status_data.get("errors", [])
                    yield f"❌ Analysis failed: {errors[0].get('error') if errors else 'Unknown error'}"
                    return

                elif status in ["completed", "awaiting_feedback"]:
                    summary = status_data.get("final_summary", "Analysis complete.")
                    if chat_id in sessions:
                        del sessions[chat_id]
                    yield summary
                    return

                if i % 3 == 0 and i > 0:
                    yield f"⏳ Still working... (status: {status})\n"

            yield "⚠️ Analysis is taking longer than expected. Check back later."

        except httpx.ConnectError:
            yield f"❌ Cannot connect to MICA backend at {MICA_BACKEND_URL}"
        except Exception as e:
            yield f"❌ Error: {str(e)}"


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
