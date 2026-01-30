"""
MCP Tool: Local Data Analysis
Description: Analyzes Excel and CSV data files from the local database
Inputs: operation (str), file_name (str), query (str), analysis_type (str)
Outputs: Data summaries, statistics, and visualizations

AGENT_INSTRUCTIONS:
You are a data analysis agent specialized in analyzing Excel and CSV files from
the local database. Your task is to:

1. List available data files in the local database
2. Read and summarize data files (columns, data types, row counts)
3. Perform statistical analysis (descriptive stats, correlations, trends)
4. Create visualizations (charts, plots, graphs)
5. Filter and query data based on user requirements

When analyzing data:
- Identify numeric vs categorical columns automatically
- Handle missing values appropriately
- Detect and report data quality issues
- Provide context about the data structure

For supply chain analysis, focus on:
- Time series data (production, imports, exports)
- Geographic data (countries, regions)
- Material categories and classifications
- Price and cost data trends
- Supply and demand metrics

Visualization capabilities:
- Bar charts for categorical comparisons
- Line plots for time series
- Scatter plots for correlations
- Pie charts for composition
- Heatmaps for matrices

Always provide clear interpretations of statistical results and visualizations.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import config
from ..logging import SessionLogger
from .base import MCPTool, ToolResult, register_tool

logger = logging.getLogger(__name__)

# Agent instructions exposed at module level
AGENT_INSTRUCTIONS = __doc__.split("AGENT_INSTRUCTIONS:")[-1].strip()


@register_tool
class LocalDataAnalysisTool(MCPTool):
    """
    Local data analysis tool for Excel and CSV files.

    Provides data exploration, statistical analysis, and visualization
    capabilities for files in the local database.
    """

    name = "local_data_analysis"
    description = "Analyze Excel and CSV data files from the local database"
    version = "1.0.0"
    AGENT_INSTRUCTIONS = AGENT_INSTRUCTIONS

    def __init__(
        self,
        session_logger: Optional[SessionLogger] = None,
        database_dir: Optional[Path] = None,
    ):
        """
        Initialize the local data analysis tool.

        Args:
            session_logger: Optional session logger
            database_dir: Override for the database directory
        """
        super().__init__(session_logger)
        self.database_dir = database_dir or config.database.database_dir
        self.data_dir = self.database_dir / config.database.data_subdir

    def execute(self, input_data: dict) -> ToolResult:
        """
        Execute local data analysis operations.

        Args:
            input_data: Dictionary with:
                - operation (str): 'list', 'summary', 'read', 'analyze', 'visualize'
                - file_name (str, optional): Specific file to analyze
                - sheet_name (str, optional): Sheet name for Excel files
                - columns (list, optional): Specific columns to analyze
                - query (str, optional): Filter query
                - analysis_type (str, optional): Type of analysis (describe, correlation, etc.)
                - chart_type (str, optional): Type of chart for visualization

        Returns:
            ToolResult with data or analysis results
        """
        start_time = datetime.now()

        try:
            operation = input_data.get("operation", "list")

            if operation == "list":
                result = self._list_files()
            elif operation == "summary":
                result = self._get_summary(input_data)
            elif operation == "read":
                result = self._read_data(input_data)
            elif operation == "analyze":
                result = self._analyze_data(input_data)
            elif operation == "visualize":
                result = self._visualize_data(input_data)
            elif operation == "query":
                result = self._query_data(input_data)
            else:
                return ToolResult.error(f"Unknown operation: {operation}")

            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time

            return result

        except Exception as e:
            logger.error(f"Local data analysis error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ToolResult.error(str(e))

    def _list_files(self) -> ToolResult:
        """List all available data files in the database."""
        if not self.data_dir.exists():
            return ToolResult.success(
                data={"files": [], "count": 0},
                message=f"Data directory not found: {self.data_dir}",
            )

        data_files = config.database.get_data_files()

        files = []
        for file_path in data_files:
            file_info = {
                "filename": file_path.name,
                "path": str(file_path),
                "type": file_path.suffix.lower().lstrip("."),
                "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            }

            # Try to get additional info
            try:
                if file_path.suffix.lower() in [".xlsx", ".xls"]:
                    import openpyxl
                    wb = openpyxl.load_workbook(file_path, read_only=True)
                    file_info["sheets"] = wb.sheetnames
                    wb.close()
                elif file_path.suffix.lower() == ".csv":
                    import csv
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        reader = csv.reader(f)
                        header = next(reader, [])
                        file_info["columns"] = len(header)
            except Exception as e:
                logger.debug(f"Could not get extra info for {file_path.name}: {e}")

            files.append(file_info)

        # Sort by name
        files.sort(key=lambda x: x["filename"])

        return ToolResult.success(
            data={
                "files": files,
                "count": len(files),
                "data_directory": str(self.data_dir),
            },
            message=f"Found {len(files)} data files in local database",
        )

    def _find_file(self, file_name: str) -> Optional[Path]:
        """Find a file by name in the data directory."""
        if not file_name:
            return None

        # Check exact path first
        exact_path = Path(file_name)
        if exact_path.exists():
            return exact_path

        # Check in data directory
        data_path = self.data_dir / file_name
        if data_path.exists():
            return data_path

        # Search for partial match
        data_files = config.database.get_data_files()
        for f in data_files:
            if file_name.lower() in f.name.lower():
                return f

        return None

    def _get_summary(self, input_data: dict) -> ToolResult:
        """Get summary information about a data file."""
        file_name = input_data.get("file_name")
        if not file_name:
            return ToolResult.error("file_name is required for summary operation")

        file_path = self._find_file(file_name)
        if not file_path:
            return ToolResult.error(f"File not found: {file_name}")

        try:
            import pandas as pd

            # Read the file
            if file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path, nrows=1000)  # Sample for large files
            else:
                sheet_name = input_data.get("sheet_name", 0)
                df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=1000)

            summary = {
                "filename": file_path.name,
                "rows": len(df),
                "columns": len(df.columns),
                "column_info": [],
                "sample_data": df.head(5).to_dict(orient="records"),
            }

            for col in df.columns:
                col_info = {
                    "name": str(col),
                    "dtype": str(df[col].dtype),
                    "non_null": int(df[col].count()),
                    "null_count": int(df[col].isna().sum()),
                }

                if pd.api.types.is_numeric_dtype(df[col]):
                    col_info["min"] = float(df[col].min()) if not pd.isna(df[col].min()) else None
                    col_info["max"] = float(df[col].max()) if not pd.isna(df[col].max()) else None
                    col_info["mean"] = float(df[col].mean()) if not pd.isna(df[col].mean()) else None
                else:
                    col_info["unique_values"] = int(df[col].nunique())
                    if df[col].nunique() <= 10:
                        col_info["value_counts"] = df[col].value_counts().head(10).to_dict()

                summary["column_info"].append(col_info)

            return ToolResult.success(
                data=summary,
                message=f"Summary for {file_path.name}: {summary['rows']} rows, {summary['columns']} columns",
            )

        except ImportError:
            return ToolResult.error("pandas is required. Install with: pip install pandas openpyxl")
        except Exception as e:
            return ToolResult.error(f"Failed to read file: {e}")

    def _read_data(self, input_data: dict) -> ToolResult:
        """Read data from a file."""
        file_name = input_data.get("file_name")
        if not file_name:
            return ToolResult.error("file_name is required for read operation")

        file_path = self._find_file(file_name)
        if not file_path:
            return ToolResult.error(f"File not found: {file_name}")

        try:
            import pandas as pd

            # Read options
            columns = input_data.get("columns")
            limit = input_data.get("limit", 100)
            sheet_name = input_data.get("sheet_name", 0)

            # Read the file
            if file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path, sheet_name=sheet_name)

            # Filter columns if specified
            if columns:
                available_cols = [c for c in columns if c in df.columns]
                if available_cols:
                    df = df[available_cols]

            # Limit rows
            if limit and limit < len(df):
                df = df.head(limit)

            # Convert to records
            data = df.to_dict(orient="records")

            return ToolResult.success(
                data={
                    "filename": file_path.name,
                    "rows": len(data),
                    "columns": list(df.columns),
                    "data": data,
                },
                message=f"Read {len(data)} rows from {file_path.name}",
            )

        except Exception as e:
            return ToolResult.error(f"Failed to read file: {e}")

    def _analyze_data(self, input_data: dict) -> ToolResult:
        """Perform statistical analysis on data."""
        file_name = input_data.get("file_name")
        if not file_name:
            return ToolResult.error("file_name is required for analyze operation")

        file_path = self._find_file(file_name)
        if not file_path:
            return ToolResult.error(f"File not found: {file_name}")

        analysis_type = input_data.get("analysis_type", "describe")
        columns = input_data.get("columns")

        try:
            import pandas as pd
            import numpy as np

            # Read the file
            if file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)
            else:
                sheet_name = input_data.get("sheet_name", 0)
                df = pd.read_excel(file_path, sheet_name=sheet_name)

            # Filter columns if specified
            if columns:
                available_cols = [c for c in columns if c in df.columns]
                if available_cols:
                    df = df[available_cols]

            result = {"filename": file_path.name, "analysis_type": analysis_type}

            if analysis_type == "describe":
                # Descriptive statistics
                desc = df.describe(include='all').to_dict()
                result["statistics"] = desc

            elif analysis_type == "correlation":
                # Correlation matrix for numeric columns
                numeric_df = df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) > 1:
                    corr = numeric_df.corr().to_dict()
                    result["correlation_matrix"] = corr
                else:
                    result["message"] = "Need at least 2 numeric columns for correlation"

            elif analysis_type == "value_counts":
                # Value counts for categorical columns
                column = input_data.get("column")
                if column and column in df.columns:
                    counts = df[column].value_counts().head(20).to_dict()
                    result["value_counts"] = counts
                    result["column"] = column
                else:
                    result["error"] = "column parameter required for value_counts"

            elif analysis_type == "trend":
                # Time series trend (requires date column)
                date_col = input_data.get("date_column")
                value_col = input_data.get("value_column")
                if date_col and value_col:
                    if date_col in df.columns and value_col in df.columns:
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                        df = df.sort_values(date_col)
                        result["trend_data"] = df[[date_col, value_col]].dropna().to_dict(orient='records')
                    else:
                        result["error"] = "Specified columns not found"
                else:
                    result["error"] = "date_column and value_column required for trend analysis"

            elif analysis_type == "group_by":
                # Group by analysis
                group_col = input_data.get("group_column")
                agg_col = input_data.get("agg_column")
                agg_func = input_data.get("agg_func", "sum")
                if group_col and agg_col:
                    if group_col in df.columns and agg_col in df.columns:
                        grouped = df.groupby(group_col)[agg_col].agg(agg_func)
                        result["grouped_data"] = grouped.to_dict()
                    else:
                        result["error"] = "Specified columns not found"
                else:
                    result["error"] = "group_column and agg_column required"

            else:
                result["error"] = f"Unknown analysis type: {analysis_type}"

            return ToolResult.success(
                data=result,
                message=f"Completed {analysis_type} analysis on {file_path.name}",
            )

        except Exception as e:
            return ToolResult.error(f"Analysis failed: {e}")

    def _visualize_data(self, input_data: dict) -> ToolResult:
        """Create visualizations from data."""
        file_name = input_data.get("file_name")
        if not file_name:
            return ToolResult.error("file_name is required for visualize operation")

        file_path = self._find_file(file_name)
        if not file_path:
            return ToolResult.error(f"File not found: {file_name}")

        chart_type = input_data.get("chart_type", "bar")
        x_column = input_data.get("x_column")
        y_column = input_data.get("y_column")
        title = input_data.get("title", f"Chart from {file_path.name}")

        try:
            import pandas as pd
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import io

            # Read the file
            if file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)
            else:
                sheet_name = input_data.get("sheet_name", 0)
                df = pd.read_excel(file_path, sheet_name=sheet_name)

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))

            if chart_type == "bar":
                if x_column and y_column:
                    df_plot = df[[x_column, y_column]].dropna().head(20)
                    ax.bar(df_plot[x_column].astype(str), df_plot[y_column])
                    ax.set_xlabel(x_column)
                    ax.set_ylabel(y_column)
                else:
                    # Default: first categorical vs first numeric
                    cat_cols = df.select_dtypes(include=['object', 'category']).columns
                    num_cols = df.select_dtypes(include=['number']).columns
                    if len(cat_cols) > 0 and len(num_cols) > 0:
                        df_plot = df.groupby(cat_cols[0])[num_cols[0]].sum().head(10)
                        df_plot.plot(kind='bar', ax=ax)
                    else:
                        return ToolResult.error("Need columns specified or data with categorical and numeric columns")

            elif chart_type == "line":
                if x_column and y_column:
                    df_plot = df[[x_column, y_column]].dropna()
                    ax.plot(df_plot[x_column], df_plot[y_column])
                    ax.set_xlabel(x_column)
                    ax.set_ylabel(y_column)
                else:
                    return ToolResult.error("x_column and y_column required for line chart")

            elif chart_type == "scatter":
                if x_column and y_column:
                    df_plot = df[[x_column, y_column]].dropna()
                    ax.scatter(df_plot[x_column], df_plot[y_column], alpha=0.6)
                    ax.set_xlabel(x_column)
                    ax.set_ylabel(y_column)
                else:
                    return ToolResult.error("x_column and y_column required for scatter plot")

            elif chart_type == "pie":
                if x_column and y_column:
                    df_plot = df[[x_column, y_column]].dropna().head(10)
                    ax.pie(df_plot[y_column], labels=df_plot[x_column], autopct='%1.1f%%')
                else:
                    return ToolResult.error("x_column (labels) and y_column (values) required for pie chart")

            elif chart_type == "histogram":
                column = input_data.get("column", y_column or x_column)
                if column:
                    df[column].dropna().hist(ax=ax, bins=20)
                    ax.set_xlabel(column)
                    ax.set_ylabel("Frequency")
                else:
                    return ToolResult.error("column required for histogram")

            else:
                return ToolResult.error(f"Unknown chart type: {chart_type}")

            ax.set_title(title)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            # Save figure
            plot_data = None
            if self.session_logger:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)

                plot_path = self.session_logger.save_artifact(
                    f"chart_{datetime.now().strftime('%H%M%S')}.png",
                    buf.getvalue(),
                    "plots",
                    binary=True,
                )
                plot_data = {"path": str(plot_path), "format": "png"}

            plt.close(fig)

            return ToolResult.success(
                data={
                    "chart_type": chart_type,
                    "title": title,
                    "plot": plot_data,
                },
                message=f"Created {chart_type} chart from {file_path.name}",
            )

        except ImportError as e:
            return ToolResult.error(f"Required library missing: {e}")
        except Exception as e:
            return ToolResult.error(f"Visualization failed: {e}")

    def _query_data(self, input_data: dict) -> ToolResult:
        """Query/filter data based on conditions."""
        file_name = input_data.get("file_name")
        if not file_name:
            return ToolResult.error("file_name is required for query operation")

        file_path = self._find_file(file_name)
        if not file_path:
            return ToolResult.error(f"File not found: {file_name}")

        query_str = input_data.get("query")
        if not query_str:
            return ToolResult.error("query parameter required (pandas query string)")

        try:
            import pandas as pd

            # Read the file
            if file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)
            else:
                sheet_name = input_data.get("sheet_name", 0)
                df = pd.read_excel(file_path, sheet_name=sheet_name)

            # Execute query
            result_df = df.query(query_str)

            limit = input_data.get("limit", 100)
            if limit and limit < len(result_df):
                result_df = result_df.head(limit)

            return ToolResult.success(
                data={
                    "filename": file_path.name,
                    "query": query_str,
                    "rows": len(result_df),
                    "columns": list(result_df.columns),
                    "data": result_df.to_dict(orient="records"),
                },
                message=f"Query returned {len(result_df)} rows from {file_path.name}",
            )

        except Exception as e:
            return ToolResult.error(f"Query failed: {e}")
