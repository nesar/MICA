"""MICA MCP Tools module."""

from .base import MCPTool, ToolResult, ToolStatus, tool_registry, register_tool
from .web_search import WebSearchTool, FederalDocumentSearchTool
from .pdf_rag import PDFRAGTool, PDFParserTool
from .excel_handler import ExcelHandlerTool, ExcelAnalysisTool
from .code_agent import CodeAgentTool, StatisticsAgentTool
from .simulation import SimulationTool, ScenarioComparisonTool
from .doc_generator import DocumentGeneratorTool, ReportTemplateTool

__all__ = [
    # Base
    "MCPTool",
    "ToolResult",
    "ToolStatus",
    "tool_registry",
    "register_tool",
    # Web Search
    "WebSearchTool",
    "FederalDocumentSearchTool",
    # PDF/RAG
    "PDFRAGTool",
    "PDFParserTool",
    # Excel
    "ExcelHandlerTool",
    "ExcelAnalysisTool",
    # Code
    "CodeAgentTool",
    "StatisticsAgentTool",
    # Simulation
    "SimulationTool",
    "ScenarioComparisonTool",
    # Documentation
    "DocumentGeneratorTool",
    "ReportTemplateTool",
]
