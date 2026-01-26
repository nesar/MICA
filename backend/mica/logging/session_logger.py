"""
Session Logger for MICA

Creates and manages session folders for comprehensive logging of all
agent interactions, queries, artifacts, and outputs.

Every query creates a session folder with the following structure:
/sessions/{uuid}/
├── metadata.json      # timestamp, user, model
├── query.txt          # original query
├── plan.json          # proposed plan
├── approval.json      # user decision
├── agent_logs/        # per-agent logs
│   ├── web_search.log
│   ├── pdf_rag.log
│   └── ...
├── artifacts/         # generated files
│   ├── plots/
│   ├── data/
│   └── code/
└── report.pdf         # final output
"""

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..config import config

# Configure module logger
logger = logging.getLogger(__name__)


class SessionLogger:
    """
    Manages session-based logging for MICA queries.

    Each query creates a unique session folder containing:
    - Query metadata and content
    - Proposed plan and user approval
    - Agent execution logs
    - Generated artifacts (plots, data, code)
    - Final report

    Usage:
        session = SessionLogger.create_session(user_id="analyst1")
        session.log_query("What are the rare earth supply chain risks?")
        session.log_plan({"steps": [...]})

        with session.agent_logger("web_search") as agent_log:
            agent_log.log("Searching for federal documents...")

        session.save_artifact("analysis.csv", csv_data, "data")
        session.finalize()
    """

    def __init__(self, session_id: str, session_dir: Path):
        """
        Initialize a session logger.

        Args:
            session_id: Unique session identifier
            session_dir: Path to the session directory
        """
        self.session_id = session_id
        self.session_dir = session_dir
        self.created_at = datetime.utcnow()

        # Create directory structure
        self._setup_directories()

        # Initialize metadata
        self.metadata = {
            "session_id": session_id,
            "created_at": self.created_at.isoformat(),
            "status": "active",
        }
        self._save_metadata()

    def _setup_directories(self):
        """Create the session directory structure."""
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.session_dir / "agent_logs").mkdir(exist_ok=True)
        (self.session_dir / "artifacts").mkdir(exist_ok=True)
        (self.session_dir / "artifacts" / "plots").mkdir(exist_ok=True)
        (self.session_dir / "artifacts" / "data").mkdir(exist_ok=True)
        (self.session_dir / "artifacts" / "code").mkdir(exist_ok=True)
        (self.session_dir / "search_results").mkdir(exist_ok=True)

    def _save_metadata(self):
        """Save metadata to file."""
        metadata_path = self.session_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def update_metadata(self, **kwargs):
        """Update session metadata with additional fields."""
        self.metadata.update(kwargs)
        self.metadata["updated_at"] = datetime.utcnow().isoformat()
        self._save_metadata()

    def log_query(self, query: str, user_id: Optional[str] = None, model: Optional[str] = None):
        """
        Log the original user query.

        Args:
            query: The user's query text
            user_id: Optional user identifier
            model: LLM model being used
        """
        # Save query text
        query_path = self.session_dir / "query.txt"
        with open(query_path, "w") as f:
            f.write(query)

        # Update metadata
        self.update_metadata(
            query=query,
            user_id=user_id,
            model=model,
        )

        logger.info(f"Session {self.session_id}: Query logged")

    def log_plan(self, plan: dict):
        """
        Log the proposed analysis plan.

        Args:
            plan: Dictionary containing the proposed plan
        """
        plan_path = self.session_dir / "plan.json"
        with open(plan_path, "w") as f:
            json.dump(plan, f, indent=2, default=str)

        self.update_metadata(
            plan_proposed_at=datetime.utcnow().isoformat(),
            status="awaiting_approval",
        )

        logger.info(f"Session {self.session_id}: Plan logged")

    def log_approval(self, approved: bool, feedback: Optional[str] = None):
        """
        Log user's approval decision.

        Args:
            approved: Whether the plan was approved
            feedback: Optional user feedback
        """
        approval_data = {
            "approved": approved,
            "feedback": feedback,
            "timestamp": datetime.utcnow().isoformat(),
        }

        approval_path = self.session_dir / "approval.json"
        with open(approval_path, "w") as f:
            json.dump(approval_data, f, indent=2)

        status = "executing" if approved else "rejected"
        self.update_metadata(
            approved=approved,
            approval_feedback=feedback,
            status=status,
        )

        logger.info(f"Session {self.session_id}: Approval logged (approved={approved})")

    def agent_logger(self, agent_name: str) -> "AgentLogger":
        """
        Get a logger for a specific agent.

        Args:
            agent_name: Name of the agent (e.g., 'web_search', 'pdf_rag')

        Returns:
            AgentLogger instance for the specified agent
        """
        return AgentLogger(self, agent_name)

    def save_artifact(
        self,
        filename: str,
        content: Any,
        artifact_type: str = "data",
        binary: bool = False,
    ) -> Path:
        """
        Save an artifact to the session.

        Args:
            filename: Name of the file
            content: Content to save (string, bytes, or dict/list for JSON)
            artifact_type: Type of artifact ('plots', 'data', 'code')
            binary: Whether to write as binary

        Returns:
            Path to the saved artifact
        """
        artifact_dir = self.session_dir / "artifacts" / artifact_type
        artifact_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = artifact_dir / filename

        if binary:
            with open(artifact_path, "wb") as f:
                f.write(content)
        elif isinstance(content, (dict, list)):
            with open(artifact_path, "w") as f:
                json.dump(content, f, indent=2, default=str)
        else:
            with open(artifact_path, "w") as f:
                f.write(str(content))

        logger.debug(f"Session {self.session_id}: Artifact saved - {artifact_path}")
        return artifact_path

    def save_search_result(self, search_id: str, results: dict) -> Path:
        """
        Save web search results.

        Args:
            search_id: Identifier for this search
            results: Search results data

        Returns:
            Path to the saved results
        """
        results_dir = self.session_dir / "search_results"
        results_path = results_dir / f"{search_id}.json"

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.debug(f"Session {self.session_id}: Search results saved - {search_id}")
        return results_path

    def save_report(self, report_content: bytes, filename: str = "report.pdf") -> Path:
        """
        Save the final report.

        Args:
            report_content: PDF content as bytes
            filename: Report filename

        Returns:
            Path to the saved report
        """
        report_path = self.session_dir / filename
        with open(report_path, "wb") as f:
            f.write(report_content)

        self.update_metadata(
            report_generated_at=datetime.utcnow().isoformat(),
            report_path=str(report_path),
        )

        logger.info(f"Session {self.session_id}: Report saved - {filename}")
        return report_path

    def log_error(self, error: Exception, context: Optional[str] = None):
        """
        Log an error that occurred during the session.

        Args:
            error: The exception that occurred
            context: Optional context about where the error occurred
        """
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Append to errors log
        errors_path = self.session_dir / "errors.json"
        errors = []
        if errors_path.exists():
            with open(errors_path) as f:
                errors = json.load(f)

        errors.append(error_data)

        with open(errors_path, "w") as f:
            json.dump(errors, f, indent=2)

        logger.error(f"Session {self.session_id}: Error - {error_type}: {error_message}")

    def finalize(self, status: str = "completed", summary: Optional[str] = None):
        """
        Finalize the session.

        Args:
            status: Final status ('completed', 'failed', 'cancelled')
            summary: Optional summary of the session
        """
        self.update_metadata(
            status=status,
            completed_at=datetime.utcnow().isoformat(),
            summary=summary,
        )

        logger.info(f"Session {self.session_id}: Finalized with status={status}")

    def get_session_path(self) -> Path:
        """Get the session directory path."""
        return self.session_dir

    @classmethod
    def create_session(
        cls,
        session_id: Optional[str] = None,
        base_dir: Optional[Path] = None,
    ) -> "SessionLogger":
        """
        Create a new session.

        Args:
            session_id: Optional custom session ID (auto-generated if not provided)
            base_dir: Base directory for sessions (defaults to config)

        Returns:
            New SessionLogger instance
        """
        session_id = session_id or str(uuid.uuid4())
        base_dir = base_dir or config.server.session_dir

        # Ensure base directory exists
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        session_dir = base_dir / session_id

        return cls(session_id, session_dir)

    @classmethod
    def load_session(cls, session_id: str, base_dir: Optional[Path] = None) -> "SessionLogger":
        """
        Load an existing session.

        Args:
            session_id: Session ID to load
            base_dir: Base directory for sessions (defaults to config)

        Returns:
            SessionLogger instance for the existing session

        Raises:
            FileNotFoundError: If session doesn't exist
        """
        base_dir = base_dir or config.server.session_dir
        session_dir = Path(base_dir) / session_id

        if not session_dir.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        instance = cls.__new__(cls)
        instance.session_id = session_id
        instance.session_dir = session_dir

        # Load metadata
        metadata_path = session_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                instance.metadata = json.load(f)
            instance.created_at = datetime.fromisoformat(instance.metadata["created_at"])
        else:
            instance.metadata = {"session_id": session_id}
            instance.created_at = datetime.utcnow()

        return instance


class AgentLogger:
    """
    Logger for individual agent execution within a session.

    Usage:
        with session.agent_logger("web_search") as agent_log:
            agent_log.log("Starting search...")
            agent_log.log_input({"query": "rare earth elements"})
            results = search(...)
            agent_log.log_output(results)
    """

    def __init__(self, session: SessionLogger, agent_name: str):
        """
        Initialize an agent logger.

        Args:
            session: Parent session logger
            agent_name: Name of the agent
        """
        self.session = session
        self.agent_name = agent_name
        self.log_path = session.session_dir / "agent_logs" / f"{agent_name}.log"
        self.started_at = None
        self.entries = []

    def __enter__(self) -> "AgentLogger":
        """Start agent logging context."""
        self.started_at = datetime.utcnow()
        self.log(f"Agent {self.agent_name} started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End agent logging context."""
        duration = (datetime.utcnow() - self.started_at).total_seconds()
        if exc_type:
            self.log(f"Agent {self.agent_name} failed: {exc_val}")
            self.log(f"Duration: {duration:.2f}s")
        else:
            self.log(f"Agent {self.agent_name} completed")
            self.log(f"Duration: {duration:.2f}s")

        self._save_log()

    def log(self, message: str, level: str = "INFO"):
        """
        Log a message.

        Args:
            message: Log message
            level: Log level (INFO, DEBUG, WARNING, ERROR)
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
        }
        self.entries.append(entry)

        # Also log to module logger
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(f"[{self.session.session_id}:{self.agent_name}] {message}")

    def log_input(self, input_data: Any):
        """Log agent input."""
        self.log(f"Input: {json.dumps(input_data, default=str)[:500]}", "DEBUG")

    def log_output(self, output_data: Any):
        """Log agent output."""
        self.log(f"Output: {json.dumps(output_data, default=str)[:500]}", "DEBUG")

    def _save_log(self):
        """Save log entries to file."""
        with open(self.log_path, "w") as f:
            for entry in self.entries:
                f.write(f"[{entry['timestamp']}] {entry['level']}: {entry['message']}\n")


# Global session registry for easy access
_sessions: dict[str, SessionLogger] = {}


def get_session_logger(session_id: str) -> Optional[SessionLogger]:
    """
    Get a session logger by ID.

    Args:
        session_id: Session ID

    Returns:
        SessionLogger if found, None otherwise
    """
    return _sessions.get(session_id)


def register_session(session: SessionLogger):
    """Register a session in the global registry."""
    _sessions[session.session_id] = session


def unregister_session(session_id: str):
    """Remove a session from the global registry."""
    _sessions.pop(session_id, None)
