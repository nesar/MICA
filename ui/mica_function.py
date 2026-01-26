"""
MICA Function for Open WebUI

Add this as a Function in Open WebUI:
1. Go to Workspace → Functions
2. Click + Create Function
3. Paste this code
4. Save and enable

Then select "MICA" as your model in the chat.
"""

import os
import json
import httpx
from typing import Optional, Callable, Awaitable

class Pipe:
    def __init__(self):
        self.type = "manifold"
        self.id = "mica"
        self.name = "MICA"
        self.MICA_URL = os.getenv("MICA_BACKEND_URL", "http://host.docker.internal:8000")

        # Track sessions per conversation
        self.sessions = {}

    def pipes(self):
        """Return available models/pipes."""
        return [
            {"id": "mica-analyst", "name": "MICA - Materials Intelligence Co-Analyst"}
        ]

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
        """Process messages through MICA."""

        messages = body.get("messages", [])
        if not messages:
            return "Please ask a question about critical materials supply chains."

        user_message = messages[-1].get("content", "").strip()
        user_id = __user__.get("id", "anonymous") if __user__ else "anonymous"

        # Get or create conversation ID
        conv_id = body.get("chat_id", user_id)

        async def emit(content: str, done: bool = False):
            if __event_emitter__:
                await __event_emitter__({
                    "type": "message",
                    "data": {"content": content}
                })
                if done:
                    await __event_emitter__({"type": "message", "data": {"done": True}})

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                # Check if we have an active session awaiting approval
                session_id = self.sessions.get(conv_id)

                if session_id:
                    # Check session status
                    status_res = await client.get(f"{self.MICA_URL}/api/v1/session/{session_id}")
                    if status_res.status_code == 200:
                        status_data = status_res.json()
                        current_status = status_data.get("status", "")

                        # Handle approval responses
                        if current_status in ["plan_proposed", "awaiting_approval"]:
                            lower_msg = user_message.lower()
                            if any(w in lower_msg for w in ["yes", "approve", "ok", "proceed", "go ahead"]):
                                # Approve the plan
                                approve_res = await client.post(
                                    f"{self.MICA_URL}/api/v1/session/{session_id}/approve",
                                    json={"session_id": session_id, "approved": True}
                                )
                                await emit("Plan approved! Executing analysis...\n\n")
                                await emit("This may take several minutes. I'll update you when complete.")
                                return ""
                            elif any(w in lower_msg for w in ["no", "reject", "cancel", "stop"]):
                                # Reject the plan
                                await client.post(
                                    f"{self.MICA_URL}/api/v1/session/{session_id}/approve",
                                    json={"session_id": session_id, "approved": False}
                                )
                                del self.sessions[conv_id]
                                return "Plan rejected. Feel free to ask a new question."

                # Check credentials first
                creds_res = await client.get(f"{self.MICA_URL}/api/v1/credentials")
                creds_data = creds_res.json()

                if creds_data.get("requires_setup"):
                    return """**Credentials Required**

Before using MICA, please set up your LLM credentials:

1. Open http://localhost:8000/docs
2. Use POST /api/v1/credentials with:
   ```json
   {"provider": "argo", "credential": "YOUR_USERNAME"}
   ```

Then come back and ask your question again."""

                # Submit new query
                await emit("Submitting query to MICA...\n\n")

                query_res = await client.post(
                    f"{self.MICA_URL}/api/v1/query",
                    json={"query": user_message, "user_id": user_id}
                )
                query_data = query_res.json()
                session_id = query_data.get("session_id")
                self.sessions[conv_id] = session_id

                await emit(f"Session created: `{session_id[:8]}...`\n\n")
                await emit("Conducting preliminary research and generating analysis plan...\n\n")

                # Poll for plan completion
                max_polls = 30  # 5 minutes max
                for i in range(max_polls):
                    await asyncio.sleep(10)

                    status_res = await client.get(f"{self.MICA_URL}/api/v1/session/{session_id}")
                    status_data = status_res.json()
                    status = status_data.get("status", "")

                    if status in ["plan_proposed", "awaiting_approval"]:
                        # Show the plan
                        plan = status_data.get("plan", [])

                        response = "**Analysis Plan Generated**\n\n"
                        response += "I've created the following analysis plan:\n\n"

                        for j, step in enumerate(plan):
                            response += f"**Step {j+1}:** {step.get('description', 'N/A')}\n"
                            response += f"   Tool: `{step.get('tool', 'N/A')}`\n\n"

                        response += "---\n\n"
                        response += "**Reply 'approve' to execute this plan, or 'reject' to cancel.**"

                        return response

                    elif status == "failed":
                        errors = status_data.get("errors", [])
                        return f"Analysis failed: {errors[0].get('error') if errors else 'Unknown error'}"

                    elif status in ["completed", "awaiting_feedback"]:
                        summary = status_data.get("final_summary", "Analysis complete.")
                        del self.sessions[conv_id]
                        return summary

                    # Still processing
                    if i % 3 == 0:
                        await emit(f"Still working... (status: {status})\n")

                return "Analysis is taking longer than expected. Please check back later."

        except httpx.ConnectError:
            return "**Error:** Cannot connect to MICA backend at " + self.MICA_URL
        except Exception as e:
            return f"**Error:** {str(e)}"


# Required for asyncio in the pipe
import asyncio
