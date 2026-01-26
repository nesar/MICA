"""
MCP Tool: Base Tool Class
Description: Base class for all MCP-compatible tools in MICA
Inputs: Varies by tool implementation
Outputs: Standardized ToolResult

This module provides the foundation for building MCP (Model Context Protocol)
compatible tools that can be used by the LangGraph orchestration system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class ToolStatus(str, Enum):
    """Status of tool execution."""

    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    PENDING = "pending"


@dataclass
class ToolResult:
    """
    Standardized result from tool execution.

    Attributes:
        status: Execution status
        data: Result data (varies by tool)
        message: Human-readable message
        metadata: Additional metadata
        error: Error message if status is ERROR
        execution_time: Time taken in seconds
    """

    status: ToolStatus
    data: Any = None
    message: str = ""
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "data": self.data,
            "message": self.message,
            "metadata": self.metadata,
            "error": self.error,
            "execution_time": self.execution_time,
        }

    @classmethod
    def success(cls, data: Any, message: str = "", **metadata) -> "ToolResult":
        """Create a success result."""
        return cls(
            status=ToolStatus.SUCCESS,
            data=data,
            message=message,
            metadata=metadata,
        )

    @classmethod
    def error(cls, error: str, message: str = "", **metadata) -> "ToolResult":
        """Create an error result."""
        return cls(
            status=ToolStatus.ERROR,
            error=error,
            message=message or error,
            metadata=metadata,
        )

    @classmethod
    def partial(cls, data: Any, message: str = "", **metadata) -> "ToolResult":
        """Create a partial success result."""
        return cls(
            status=ToolStatus.PARTIAL,
            data=data,
            message=message,
            metadata=metadata,
        )


class MCPToolInput(BaseModel):
    """Base input model for MCP tools."""

    class Config:
        extra = "allow"


class MCPTool(ABC):
    """
    Abstract base class for MCP-compatible tools.

    All MICA tools inherit from this class and implement the execute method.
    Tools are designed to:
    - Have clear, documented inputs and outputs
    - Return standardized ToolResult objects
    - Log their execution through the session logger
    - Be usable by the LangGraph orchestration system

    Usage:
        class MyTool(MCPTool):
            name = "my_tool"
            description = "Does something useful"

            def execute(self, input_data: dict) -> ToolResult:
                # Tool implementation
                return ToolResult.success(result_data)
    """

    # Tool metadata - override in subclasses
    name: str = "base_tool"
    description: str = "Base MCP tool"
    version: str = "1.0.0"

    # Agent instructions - override in subclasses
    AGENT_INSTRUCTIONS: str = ""

    def __init__(self, session_logger=None):
        """
        Initialize the tool.

        Args:
            session_logger: Optional SessionLogger for logging execution
        """
        self.session_logger = session_logger

    @abstractmethod
    def execute(self, input_data: dict) -> ToolResult:
        """
        Execute the tool.

        Args:
            input_data: Tool-specific input dictionary

        Returns:
            ToolResult with execution results
        """
        pass

    async def aexecute(self, input_data: dict) -> ToolResult:
        """
        Async version of execute. Override for async implementations.

        By default, wraps the sync execute method.
        """
        return self.execute(input_data)

    def validate_input(self, input_data: dict) -> tuple[bool, Optional[str]]:
        """
        Validate input data before execution.

        Args:
            input_data: Input data to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        return True, None

    def __call__(self, **kwargs) -> ToolResult:
        """Allow calling the tool directly."""
        return self.execute(kwargs)

    def to_langchain_tool(self) -> BaseTool:
        """Convert to a LangChain-compatible tool."""
        return LangChainToolWrapper(self)

    def get_schema(self) -> dict:
        """Get the tool's input schema."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "agent_instructions": self.AGENT_INSTRUCTIONS,
        }


class LangChainToolWrapper(BaseTool):
    """Wrapper to make MCPTool compatible with LangChain."""

    mcp_tool: MCPTool

    def __init__(self, mcp_tool: MCPTool):
        """
        Initialize the wrapper.

        Args:
            mcp_tool: The MCP tool to wrap
        """
        super().__init__(
            name=mcp_tool.name,
            description=mcp_tool.description,
        )
        self.mcp_tool = mcp_tool

    def _run(self, **kwargs) -> str:
        """Execute the tool and return result as string."""
        result = self.mcp_tool.execute(kwargs)
        if result.status == ToolStatus.SUCCESS:
            return str(result.data)
        else:
            return f"Error: {result.error or result.message}"

    async def _arun(self, **kwargs) -> str:
        """Async execution."""
        result = await self.mcp_tool.aexecute(kwargs)
        if result.status == ToolStatus.SUCCESS:
            return str(result.data)
        else:
            return f"Error: {result.error or result.message}"


class ToolRegistry:
    """
    Registry for available MCP tools.

    Provides a central place to register and retrieve tools.
    """

    def __init__(self):
        self._tools: dict[str, type[MCPTool]] = {}

    def register(self, tool_class: type[MCPTool]):
        """Register a tool class."""
        self._tools[tool_class.name] = tool_class

    def get(self, name: str) -> Optional[type[MCPTool]]:
        """Get a tool class by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_all_schemas(self) -> list[dict]:
        """Get schemas for all registered tools."""
        schemas = []
        for tool_class in self._tools.values():
            tool = tool_class()
            schemas.append(tool.get_schema())
        return schemas


# Global tool registry
tool_registry = ToolRegistry()


def register_tool(tool_class: type[MCPTool]) -> type[MCPTool]:
    """Decorator to register a tool."""
    tool_registry.register(tool_class)
    return tool_class
