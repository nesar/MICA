"""
MCP Tool: Code Agent
Description: Executes Python code for statistical analysis and visualization
Inputs: code (str), context (dict with data), output_type (str)
Outputs: Execution results, generated plots, computed values

AGENT_INSTRUCTIONS:
You are a data analysis and visualization agent specialized in Python programming.
Your task is to:

1. Write and execute Python code for statistical analysis
2. Create visualizations (charts, plots, maps) from data
3. Perform calculations and transformations on datasets
4. Generate summary statistics and reports
5. Handle data cleaning and preprocessing

When writing code:
- Use pandas for data manipulation
- Use matplotlib/plotly for visualization
- Use scipy/numpy for statistical analysis
- Write clean, readable, well-commented code
- Handle errors gracefully and provide informative messages

For supply chain analysis, focus on:
- Time series analysis and forecasting
- Trade flow visualizations (Sankey diagrams)
- Geographic mapping of supply sources
- Cost and price trend analysis
- Comparative analysis across materials/regions

Security requirements:
- Do not access external networks (use provided data only)
- Do not write to system directories
- Do not execute shell commands
- Keep execution time reasonable (< 60 seconds)

Always validate inputs and provide clear output with interpretation.
"""

import io
import logging
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..logging import SessionLogger
from .base import MCPTool, ToolResult, register_tool

logger = logging.getLogger(__name__)

# Agent instructions exposed at module level
AGENT_INSTRUCTIONS = __doc__.split("AGENT_INSTRUCTIONS:")[-1].strip()

# Allowed imports for code execution (security)
ALLOWED_IMPORTS = {
    "pandas",
    "numpy",
    "matplotlib",
    "matplotlib.pyplot",
    "plotly",
    "plotly.express",
    "plotly.graph_objects",
    "scipy",
    "scipy.stats",
    "sklearn",
    "statistics",
    "math",
    "json",
    "datetime",
    "collections",
    "itertools",
    "functools",
    "re",
}

# Maximum execution time in seconds
MAX_EXECUTION_TIME = 60


@register_tool
class CodeAgentTool(MCPTool):
    """
    Code execution agent for statistical analysis and visualization.

    Executes Python code in a sandboxed environment with access to
    data analysis libraries.
    """

    name = "code_agent"
    description = "Execute Python code for statistical analysis and visualization"
    version = "1.0.0"
    AGENT_INSTRUCTIONS = AGENT_INSTRUCTIONS

    def __init__(
        self,
        session_logger: Optional[SessionLogger] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the code agent tool.

        Args:
            session_logger: Optional session logger
            output_dir: Directory for saving outputs (plots, data)
        """
        super().__init__(session_logger)
        self.output_dir = output_dir

    def execute(self, input_data: dict) -> ToolResult:
        """
        Execute Python code.

        Args:
            input_data: Dictionary with:
                - code (str): Python code to execute
                - context (dict, optional): Variables to inject into execution context
                - save_plots (bool, optional): Whether to save generated plots
                - timeout (int, optional): Execution timeout in seconds

        Returns:
            ToolResult with execution results
        """
        start_time = datetime.now()

        code = input_data.get("code")
        if not code:
            return ToolResult.error("code is required")

        context = input_data.get("context", {})
        save_plots = input_data.get("save_plots", True)
        timeout = min(input_data.get("timeout", MAX_EXECUTION_TIME), MAX_EXECUTION_TIME)

        try:
            result = self._execute_code(code, context, save_plots, timeout)
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time

            # Log code execution
            if self.session_logger:
                self.session_logger.save_artifact(
                    f"code_{datetime.now().strftime('%H%M%S')}.py",
                    code,
                    "code",
                )

            return result

        except Exception as e:
            logger.error(f"Code execution error: {e}")
            return ToolResult.error(str(e))

    def _execute_code(
        self,
        code: str,
        context: dict,
        save_plots: bool,
        timeout: int,
    ) -> ToolResult:
        """Execute code in a sandboxed environment."""
        # Prepare execution namespace
        namespace = self._prepare_namespace(context)

        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        plots = []
        result_value = None

        try:
            # Configure matplotlib for non-interactive backend
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # Execute code
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, namespace)

            # Get result if defined
            result_value = namespace.get("result", namespace.get("output"))

            # Capture any plots
            if save_plots:
                figs = [plt.figure(num) for num in plt.get_fignums()]
                for i, fig in enumerate(figs):
                    plot_data = self._save_figure(fig, f"plot_{i}")
                    if plot_data:
                        plots.append(plot_data)

                plt.close('all')

            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()

            return ToolResult.success(
                data={
                    "result": self._serialize_result(result_value),
                    "stdout": stdout_output,
                    "stderr": stderr_output,
                    "plots": plots,
                },
                message="Code executed successfully",
                has_plots=len(plots) > 0,
            )

        except SyntaxError as e:
            return ToolResult.error(
                f"Syntax error in code: {e}",
                line=e.lineno,
                text=e.text,
            )
        except Exception as e:
            tb = traceback.format_exc()
            return ToolResult.error(
                f"Execution error: {type(e).__name__}: {e}",
                traceback=tb,
            )

    def _prepare_namespace(self, context: dict) -> dict:
        """Prepare the execution namespace with safe imports and context."""
        namespace = {"__builtins__": self._get_safe_builtins()}

        # Import allowed modules
        try:
            import numpy as np
            import pandas as pd
            namespace["np"] = np
            namespace["numpy"] = np
            namespace["pd"] = pd
            namespace["pandas"] = pd
        except ImportError:
            pass

        try:
            import matplotlib.pyplot as plt
            namespace["plt"] = plt
            namespace["matplotlib"] = __import__("matplotlib")
        except ImportError:
            pass

        try:
            import plotly.express as px
            import plotly.graph_objects as go
            namespace["px"] = px
            namespace["go"] = go
            namespace["plotly"] = __import__("plotly")
        except ImportError:
            pass

        try:
            from scipy import stats
            namespace["stats"] = stats
            namespace["scipy"] = __import__("scipy")
        except ImportError:
            pass

        # Add standard library modules
        import json
        import math
        import statistics
        from collections import Counter, defaultdict
        from datetime import datetime, timedelta

        namespace.update({
            "json": json,
            "math": math,
            "statistics": statistics,
            "Counter": Counter,
            "defaultdict": defaultdict,
            "datetime": datetime,
            "timedelta": timedelta,
        })

        # Inject user context
        namespace.update(context)

        return namespace

    def _get_safe_builtins(self) -> dict:
        """Get a restricted set of built-in functions."""
        safe_builtins = {}
        allowed_builtins = [
            "abs", "all", "any", "bool", "bytes", "callable", "chr",
            "dict", "dir", "divmod", "enumerate", "filter", "float",
            "format", "frozenset", "getattr", "hasattr", "hash", "hex",
            "int", "isinstance", "issubclass", "iter", "len", "list",
            "map", "max", "min", "next", "oct", "ord", "pow", "print",
            "range", "repr", "reversed", "round", "set", "slice",
            "sorted", "str", "sum", "tuple", "type", "zip",
        ]

        import builtins
        for name in allowed_builtins:
            if hasattr(builtins, name):
                safe_builtins[name] = getattr(builtins, name)

        # Add True, False, None
        safe_builtins["True"] = True
        safe_builtins["False"] = False
        safe_builtins["None"] = None

        return safe_builtins

    def _serialize_result(self, value: Any) -> Any:
        """Serialize execution result for JSON output."""
        if value is None:
            return None

        # Handle pandas objects
        try:
            import pandas as pd
            if isinstance(value, pd.DataFrame):
                return {
                    "type": "DataFrame",
                    "shape": list(value.shape),
                    "columns": list(value.columns),
                    "data": value.head(100).to_dict(orient="records"),
                }
            if isinstance(value, pd.Series):
                return {
                    "type": "Series",
                    "name": value.name,
                    "data": value.head(100).to_dict(),
                }
        except ImportError:
            pass

        # Handle numpy arrays
        try:
            import numpy as np
            if isinstance(value, np.ndarray):
                return {
                    "type": "ndarray",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "data": value.tolist() if value.size <= 1000 else value[:100].tolist(),
                }
        except ImportError:
            pass

        # Handle basic types
        if isinstance(value, (int, float, str, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [self._serialize_result(v) for v in value[:100]]
        if isinstance(value, dict):
            return {k: self._serialize_result(v) for k, v in list(value.items())[:100]}

        # Default: convert to string
        return str(value)

    def _save_figure(self, fig, name: str) -> Optional[dict]:
        """Save a matplotlib figure and return metadata."""
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)

            # Save to session if available
            if self.session_logger:
                path = self.session_logger.save_artifact(
                    f"{name}.png",
                    buf.getvalue(),
                    "plots",
                    binary=True,
                )
                return {
                    "name": name,
                    "path": str(path),
                    "format": "png",
                }

            return {
                "name": name,
                "format": "png",
                "data_available": True,
            }

        except Exception as e:
            logger.warning(f"Failed to save figure: {e}")
            return None


@register_tool
class StatisticsAgentTool(MCPTool):
    """
    Simplified statistics agent for common calculations.

    Provides pre-built statistical analysis without code execution.
    """

    name = "statistics_agent"
    description = "Compute common statistics on data"
    version = "1.0.0"

    AGENT_INSTRUCTIONS = """
    You compute basic statistics on provided data including:
    mean, median, std, min, max, percentiles, correlations, and trend analysis.
    """

    def execute(self, input_data: dict) -> ToolResult:
        """Calculate statistics on data."""
        data = input_data.get("data")
        if not data:
            return ToolResult.error("data is required")

        operation = input_data.get("operation", "summary")

        try:
            import numpy as np

            # Convert to numpy array if list
            if isinstance(data, list):
                data = np.array(data, dtype=float)

            if operation == "summary":
                result = {
                    "count": len(data),
                    "mean": float(np.mean(data)),
                    "median": float(np.median(data)),
                    "std": float(np.std(data)),
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "percentile_25": float(np.percentile(data, 25)),
                    "percentile_75": float(np.percentile(data, 75)),
                }
            elif operation == "histogram":
                hist, bin_edges = np.histogram(data, bins=input_data.get("bins", 10))
                result = {
                    "counts": hist.tolist(),
                    "bin_edges": bin_edges.tolist(),
                }
            else:
                return ToolResult.error(f"Unknown operation: {operation}")

            return ToolResult.success(data=result, message="Statistics computed")

        except Exception as e:
            return ToolResult.error(f"Statistics error: {e}")
