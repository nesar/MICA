"""
MICA WebSocket Handler

Provides real-time streaming of workflow progress and results.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from ..agents import WorkflowStatus

logger = logging.getLogger(__name__)

websocket_router = APIRouter(tags=["websocket"])


class ConnectionManager:
    """
    Manages WebSocket connections for real-time updates.

    Supports multiple clients per session for collaborative analysis.
    """

    def __init__(self):
        # session_id -> set of websockets
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()

        if session_id not in self.active_connections:
            self.active_connections[session_id] = set()

        self.active_connections[session_id].add(websocket)
        logger.info(f"WebSocket connected for session {session_id}")

    def disconnect(self, websocket: WebSocket, session_id: str):
        """Handle WebSocket disconnection."""
        if session_id in self.active_connections:
            self.active_connections[session_id].discard(websocket)

            # Clean up empty session entries
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]

        logger.info(f"WebSocket disconnected from session {session_id}")

    async def send_to_session(self, session_id: str, message: dict):
        """Send a message to all connections for a session."""
        if session_id not in self.active_connections:
            return

        disconnected = set()

        for websocket in self.active_connections[session_id]:
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to websocket: {e}")
                disconnected.add(websocket)

        # Clean up disconnected websockets
        for ws in disconnected:
            self.disconnect(ws, session_id)

    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        for session_id in list(self.active_connections.keys()):
            await self.send_to_session(session_id, message)

    def get_connection_count(self, session_id: str) -> int:
        """Get the number of connections for a session."""
        return len(self.active_connections.get(session_id, set()))


# Global connection manager
manager = ConnectionManager()


@websocket_router.websocket("/session/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time session updates.

    Clients can:
    - Receive workflow status updates
    - Receive intermediate results
    - Send approval/feedback messages
    """
    await manager.connect(websocket, session_id)

    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
        })

        while True:
            # Wait for messages from client
            data = await websocket.receive_json()

            # Handle different message types
            message_type = data.get("type")

            if message_type == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat(),
                })

            elif message_type == "approve":
                # Handle approval via WebSocket
                approved = data.get("approved", False)
                feedback = data.get("feedback")

                await send_session_update(
                    session_id,
                    "approval_received",
                    {"approved": approved, "feedback": feedback},
                )

            elif message_type == "feedback":
                # Handle feedback via WebSocket
                feedback = data.get("feedback")
                follow_up = data.get("follow_up_query")

                await send_session_update(
                    session_id,
                    "feedback_received",
                    {"feedback": feedback, "follow_up_query": follow_up},
                )

            elif message_type == "subscribe":
                # Client wants to subscribe to updates
                await websocket.send_json({
                    "type": "subscribed",
                    "session_id": session_id,
                })

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}",
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        manager.disconnect(websocket, session_id)


async def send_session_update(
    session_id: str,
    event_type: str,
    data: dict,
):
    """
    Send an update to all clients connected to a session.

    Called by the workflow execution to notify clients of progress.
    """
    message = {
        "type": event_type,
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data,
    }

    await manager.send_to_session(session_id, message)


async def send_workflow_status(
    session_id: str,
    status: WorkflowStatus,
    current_step: str = None,
    details: dict = None,
):
    """Send workflow status update."""
    await send_session_update(
        session_id,
        "workflow_status",
        {
            "status": status.value if isinstance(status, WorkflowStatus) else status,
            "current_step": current_step,
            "details": details or {},
        },
    )


async def send_plan_update(session_id: str, plan: list, reasoning: str):
    """Send plan update to clients."""
    await send_session_update(
        session_id,
        "plan_ready",
        {
            "plan": plan,
            "reasoning": reasoning,
            "awaiting_approval": True,
        },
    )


async def send_step_progress(
    session_id: str,
    step_id: str,
    status: str,
    output: dict = None,
    error: str = None,
):
    """Send step execution progress."""
    await send_session_update(
        session_id,
        "step_progress",
        {
            "step_id": step_id,
            "status": status,
            "output": output,
            "error": error,
        },
    )


async def send_final_results(
    session_id: str,
    summary: str,
    report_path: str = None,
    artifacts: list = None,
):
    """Send final results to clients."""
    await send_session_update(
        session_id,
        "results_ready",
        {
            "summary": summary,
            "report_path": report_path,
            "artifacts": artifacts or [],
            "awaiting_feedback": True,
        },
    )


@websocket_router.websocket("/stream")
async def stream_endpoint(websocket: WebSocket):
    """
    General streaming endpoint for all updates.

    Clients can subscribe to updates for specific sessions.
    """
    await websocket.accept()

    subscribed_sessions: Set[str] = set()

    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")

            if message_type == "subscribe":
                session_id = data.get("session_id")
                if session_id:
                    subscribed_sessions.add(session_id)
                    await websocket.send_json({
                        "type": "subscribed",
                        "session_id": session_id,
                    })

            elif message_type == "unsubscribe":
                session_id = data.get("session_id")
                subscribed_sessions.discard(session_id)
                await websocket.send_json({
                    "type": "unsubscribed",
                    "session_id": session_id,
                })

            elif message_type == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat(),
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Stream WebSocket error: {e}")


# Export for use in other modules
__all__ = [
    "websocket_router",
    "manager",
    "send_session_update",
    "send_workflow_status",
    "send_plan_update",
    "send_step_progress",
    "send_final_results",
]
