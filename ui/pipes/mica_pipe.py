"""
MICA Pipe for Open WebUI

This pipe connects Open WebUI to the MICA backend API, enabling users
to interact with the MICA analysis system through the Open WebUI interface.

Installation:
1. In Open WebUI, go to Admin Settings > Pipelines
2. Add a new pipeline and paste this code
3. Configure the MICA_BACKEND_URL environment variable

Usage:
Users can interact with MICA through natural language queries about
critical materials supply chains. The pipe handles the workflow:
1. Query submission
2. Plan presentation and approval
3. Results display
"""

import os
import json
import asyncio
from typing import List, Optional, Generator, Iterator, Union
import httpx


# Pipe metadata
class Pipe:
    """
    MICA Integration Pipe for Open WebUI.

    Connects to the MICA backend API for critical materials supply chain analysis.
    """

    class Valves:
        """Configuration valves for the pipe."""

        def __init__(self):
            self.MICA_BACKEND_URL = os.getenv("MICA_BACKEND_URL", "http://localhost:8000")
            self.MICA_API_KEY = os.getenv("MICA_API_KEY", "")
            self.TIMEOUT = int(os.getenv("MICA_TIMEOUT", "120"))

    def __init__(self):
        self.type = "pipe"
        self.id = "mica"
        self.name = "MICA - Materials Intelligence Co-Analyst"
        self.valves = self.Valves()

        # Session tracking
        self._current_session = None
        self._awaiting_approval = False
        self._awaiting_feedback = False

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__=None,
    ) -> Union[str, Generator, Iterator]:
        """
        Main pipe function called by Open WebUI.

        Args:
            body: Request body with messages
            __user__: User information
            __event_emitter__: Event emitter for streaming

        Returns:
            Response string or generator
        """
        messages = body.get("messages", [])
        if not messages:
            return "Please provide a query about critical materials supply chains."

        # Get the latest user message
        user_message = messages[-1].get("content", "")
        user_id = __user__.get("id") if __user__ else None

        # Check if this is an approval/feedback response
        if self._awaiting_approval:
            return await self._handle_approval(user_message, __event_emitter__)
        elif self._awaiting_feedback:
            return await self._handle_feedback(user_message, __event_emitter__)
        else:
            return await self._handle_query(user_message, user_id, __event_emitter__)

    async def _handle_query(
        self,
        query: str,
        user_id: Optional[str],
        event_emitter,
    ) -> str:
        """Handle a new analysis query."""
        try:
            async with httpx.AsyncClient(timeout=self.valves.TIMEOUT) as client:
                # Submit query to MICA backend
                response = await client.post(
                    f"{self.valves.MICA_BACKEND_URL}/api/v1/query",
                    json={
                        "query": query,
                        "user_id": user_id,
                        "session_id": self._current_session,
                    },
                    headers=self._get_headers(),
                )

                if response.status_code != 200:
                    return f"Error submitting query: {response.text}"

                data = response.json()
                self._current_session = data.get("session_id")

                # Emit status update
                if event_emitter:
                    await event_emitter({
                        "type": "status",
                        "data": {"description": "Query submitted. Generating analysis plan..."},
                    })

                # Poll for plan completion
                plan_response = await self._wait_for_plan(client, event_emitter)

                if plan_response:
                    self._awaiting_approval = True
                    return self._format_plan_message(plan_response)
                else:
                    return "Error: Could not generate analysis plan. Please try again."

        except httpx.TimeoutException:
            return "Request timed out. The analysis may take longer than expected. Please check back later."
        except Exception as e:
            return f"Error: {str(e)}"

    async def _wait_for_plan(self, client: httpx.AsyncClient, event_emitter) -> Optional[dict]:
        """Poll the backend until the plan is ready."""
        max_attempts = 60  # 60 seconds max
        attempt = 0

        while attempt < max_attempts:
            response = await client.get(
                f"{self.valves.MICA_BACKEND_URL}/api/v1/session/{self._current_session}",
                headers=self._get_headers(),
            )

            if response.status_code == 200:
                data = response.json()
                status = data.get("status")

                if status == "awaiting_approval":
                    # Plan is ready
                    plan_response = await client.get(
                        f"{self.valves.MICA_BACKEND_URL}/api/v1/session/{self._current_session}/plan",
                        headers=self._get_headers(),
                    )
                    return plan_response.json() if plan_response.status_code == 200 else None

                elif status == "failed":
                    return None

                # Still processing
                if event_emitter:
                    await event_emitter({
                        "type": "status",
                        "data": {
                            "description": f"Processing... ({data.get('current_step', 'working')})",
                        },
                    })

            await asyncio.sleep(1)
            attempt += 1

        return None

    async def _handle_approval(self, response: str, event_emitter) -> str:
        """Handle user approval of the analysis plan."""
        response_lower = response.lower().strip()

        # Determine if approved
        approved = response_lower in ["yes", "approve", "approved", "ok", "okay", "y", "proceed"]
        rejected = response_lower in ["no", "reject", "rejected", "n", "cancel"]

        if not approved and not rejected:
            # Treat as feedback for revision
            feedback = response
            approved = False
        else:
            feedback = None if approved else response

        try:
            async with httpx.AsyncClient(timeout=self.valves.TIMEOUT) as client:
                response = await client.post(
                    f"{self.valves.MICA_BACKEND_URL}/api/v1/session/{self._current_session}/approve",
                    json={
                        "session_id": self._current_session,
                        "approved": approved,
                        "feedback": feedback,
                    },
                    headers=self._get_headers(),
                )

                if response.status_code != 200:
                    return f"Error processing approval: {response.text}"

                self._awaiting_approval = False

                if approved:
                    if event_emitter:
                        await event_emitter({
                            "type": "status",
                            "data": {"description": "Plan approved. Executing analysis..."},
                        })

                    # Wait for execution to complete
                    results = await self._wait_for_results(client, event_emitter)

                    if results:
                        self._awaiting_feedback = True
                        return self._format_results_message(results)
                    else:
                        return "Analysis execution encountered an error. Please check the session status."
                else:
                    # Plan rejected, wait for revised plan
                    plan_response = await self._wait_for_plan(client, event_emitter)

                    if plan_response:
                        self._awaiting_approval = True
                        return "Here's the revised plan:\n\n" + self._format_plan_message(plan_response)
                    else:
                        return "Session cancelled."

        except Exception as e:
            return f"Error: {str(e)}"

    async def _wait_for_results(self, client: httpx.AsyncClient, event_emitter) -> Optional[dict]:
        """Poll the backend until results are ready."""
        max_attempts = 300  # 5 minutes max for execution
        attempt = 0

        while attempt < max_attempts:
            response = await client.get(
                f"{self.valves.MICA_BACKEND_URL}/api/v1/session/{self._current_session}",
                headers=self._get_headers(),
            )

            if response.status_code == 200:
                data = response.json()
                status = data.get("status")

                if status in ["awaiting_feedback", "completed"]:
                    return data

                elif status == "failed":
                    return None

                # Still executing
                if event_emitter:
                    await event_emitter({
                        "type": "status",
                        "data": {
                            "description": f"Executing... ({data.get('current_step', 'working')})",
                        },
                    })

            await asyncio.sleep(2)
            attempt += 1

        return None

    async def _handle_feedback(self, feedback: str, event_emitter) -> str:
        """Handle user feedback on results."""
        feedback_lower = feedback.lower().strip()

        # Check if this is a follow-up query
        is_follow_up = any(word in feedback_lower for word in ["also", "additionally", "what about", "can you", "please"])

        try:
            async with httpx.AsyncClient(timeout=self.valves.TIMEOUT) as client:
                response = await client.post(
                    f"{self.valves.MICA_BACKEND_URL}/api/v1/session/{self._current_session}/feedback",
                    json={
                        "session_id": self._current_session,
                        "feedback": feedback,
                        "follow_up_query": feedback if is_follow_up else None,
                    },
                    headers=self._get_headers(),
                )

                if response.status_code != 200:
                    return f"Error submitting feedback: {response.text}"

                data = response.json()

                if data.get("status") == "processing":
                    # Follow-up is being processed
                    if event_emitter:
                        await event_emitter({
                            "type": "status",
                            "data": {"description": "Processing follow-up query..."},
                        })

                    results = await self._wait_for_results(client, event_emitter)

                    if results:
                        return self._format_results_message(results)
                    else:
                        return "Follow-up analysis encountered an error."
                else:
                    # Session completed
                    self._awaiting_feedback = False
                    self._current_session = None
                    return "Thank you for your feedback! Session completed. Feel free to start a new query."

        except Exception as e:
            return f"Error: {str(e)}"

    def _format_plan_message(self, plan_data: dict) -> str:
        """Format the plan for display."""
        lines = ["## Analysis Plan\n"]

        reasoning = plan_data.get("reasoning", "")
        if reasoning:
            lines.append(f"**Approach:** {reasoning[:500]}...\n" if len(reasoning) > 500 else f"**Approach:** {reasoning}\n")

        lines.append("\n### Steps:\n")

        for i, step in enumerate(plan_data.get("plan", []), 1):
            tool = step.get("tool", "analysis")
            description = step.get("description", "")
            lines.append(f"{i}. **[{tool}]** {description}")

        lines.append("\n---")
        lines.append("\n**Do you approve this plan?**")
        lines.append("Reply with 'yes' to proceed, or provide feedback for revisions.")

        return "\n".join(lines)

    def _format_results_message(self, results_data: dict) -> str:
        """Format the results for display."""
        lines = ["## Analysis Results\n"]

        summary = results_data.get("final_summary", "")
        if summary:
            lines.append(summary)

        report_path = results_data.get("report_path")
        if report_path:
            lines.append(f"\n**Report generated:** {report_path}")

        errors = results_data.get("errors", [])
        if errors:
            lines.append("\n### Notes:")
            for error in errors[:3]:
                lines.append(f"- {error.get('error', 'Unknown error')}")

        lines.append("\n---")
        lines.append("\n**How can I help further?**")
        lines.append("Provide feedback, ask follow-up questions, or say 'done' to complete.")

        return "\n".join(lines)

    def _get_headers(self) -> dict:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.valves.MICA_API_KEY:
            headers["Authorization"] = f"Bearer {self.valves.MICA_API_KEY}"
        return headers

    def pipes(self) -> List[dict]:
        """Return pipe configuration for Open WebUI."""
        return [
            {
                "id": "mica",
                "name": "MICA - Materials Intelligence Co-Analyst",
                "description": "AI-powered critical materials supply chain analysis",
            }
        ]
