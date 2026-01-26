"""
MCP Tool: Document Generator
Description: Generates PDF reports from analysis results
Inputs: content (dict), template (str), output_path (str)
Outputs: Generated PDF report

AGENT_INSTRUCTIONS:
You are a document generation agent specialized in creating professional reports.
Your task is to:

1. Compile analysis results into structured reports
2. Format content with appropriate headers, sections, and styling
3. Include tables, charts, and visualizations
4. Generate executive summaries and key findings
5. Ensure proper citation and source attribution

When creating reports:
- Use clear, professional language
- Structure content logically (executive summary, findings, details, appendix)
- Include relevant visualizations and data tables
- Highlight key insights and recommendations
- Maintain consistent formatting throughout

For supply chain analysis reports:
- Include supply risk assessments
- Present cost and capacity projections
- Show geographic distribution of sources
- Summarize policy implications
- Provide actionable recommendations

Report sections typically include:
1. Executive Summary
2. Background and Context
3. Methodology
4. Key Findings
5. Detailed Analysis
6. Recommendations
7. Appendices (data tables, methodology details)
"""

import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..logging import SessionLogger
from .base import MCPTool, ToolResult, register_tool

logger = logging.getLogger(__name__)

# Agent instructions exposed at module level
AGENT_INSTRUCTIONS = __doc__.split("AGENT_INSTRUCTIONS:")[-1].strip()


@register_tool
class DocumentGeneratorTool(MCPTool):
    """
    Document generation tool for creating PDF reports.

    Uses ReportLab for PDF generation.
    """

    name = "doc_generator"
    description = "Generate PDF reports from analysis results"
    version = "1.0.0"
    AGENT_INSTRUCTIONS = AGENT_INSTRUCTIONS

    def __init__(self, session_logger: Optional[SessionLogger] = None):
        """Initialize the document generator tool."""
        super().__init__(session_logger)

    def execute(self, input_data: dict) -> ToolResult:
        """
        Generate a PDF report.

        Args:
            input_data: Dictionary with:
                - title (str): Report title
                - sections (list[dict]): Report sections
                - metadata (dict, optional): Report metadata
                - output_path (str, optional): Output file path

        Returns:
            ToolResult with report path or bytes
        """
        start_time = datetime.now()

        title = input_data.get("title", "MICA Analysis Report")
        sections = input_data.get("sections", [])
        metadata = input_data.get("metadata", {})
        output_path = input_data.get("output_path")

        if not sections:
            return ToolResult.error("sections are required")

        try:
            pdf_bytes = self._generate_pdf(title, sections, metadata)

            # Save or return PDF
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(pdf_bytes)
                result_path = str(output_path)
            elif self.session_logger:
                filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                result_path = str(self.session_logger.save_report(pdf_bytes, filename))
            else:
                result_path = None

            execution_time = (datetime.now() - start_time).total_seconds()

            return ToolResult(
                status=ToolResult.success(None).status,
                data={
                    "path": result_path,
                    "title": title,
                    "sections": len(sections),
                    "size_bytes": len(pdf_bytes),
                },
                message=f"Report generated: {title}",
                execution_time=execution_time,
            )

        except Exception as e:
            logger.error(f"Document generation error: {e}")
            return ToolResult.error(str(e))

    def _generate_pdf(
        self,
        title: str,
        sections: list[dict],
        metadata: dict,
    ) -> bytes:
        """Generate PDF document."""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                Paragraph,
                SimpleDocTemplate,
                Spacer,
                Table,
                TableStyle,
                PageBreak,
            )
        except ImportError:
            raise ImportError("reportlab is required. Install with: pip install reportlab")

        buffer = io.BytesIO()

        # Create document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        # Get styles
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Title"],
            fontSize=24,
            spaceAfter=30,
        )

        heading_style = ParagraphStyle(
            "CustomHeading",
            parent=styles["Heading1"],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
        )

        subheading_style = ParagraphStyle(
            "CustomSubheading",
            parent=styles["Heading2"],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=8,
        )

        body_style = ParagraphStyle(
            "CustomBody",
            parent=styles["Normal"],
            fontSize=11,
            spaceBefore=6,
            spaceAfter=6,
        )

        # Build content
        story = []

        # Title page
        story.append(Spacer(1, 2 * inch))
        story.append(Paragraph(title, title_style))

        # Metadata
        if metadata:
            story.append(Spacer(1, 0.5 * inch))
            meta_text = "<br/>".join(
                f"<b>{k}:</b> {v}" for k, v in metadata.items()
            )
            story.append(Paragraph(meta_text, body_style))

        # Generated timestamp
        story.append(Spacer(1, 0.5 * inch))
        story.append(
            Paragraph(
                f"<i>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>",
                body_style,
            )
        )
        story.append(Paragraph("<i>Generated by MICA</i>", body_style))

        story.append(PageBreak())

        # Table of Contents placeholder
        story.append(Paragraph("Table of Contents", heading_style))
        for i, section in enumerate(sections, 1):
            toc_entry = f"{i}. {section.get('title', f'Section {i}')}"
            story.append(Paragraph(toc_entry, body_style))
        story.append(PageBreak())

        # Sections
        for i, section in enumerate(sections, 1):
            section_title = section.get("title", f"Section {i}")
            story.append(Paragraph(f"{i}. {section_title}", heading_style))

            # Section content
            content = section.get("content", "")
            if isinstance(content, str):
                # Plain text content
                paragraphs = content.split("\n\n")
                for para in paragraphs:
                    if para.strip():
                        story.append(Paragraph(para.strip(), body_style))
            elif isinstance(content, list):
                # List of content items
                for item in content:
                    if isinstance(item, str):
                        story.append(Paragraph(item, body_style))
                    elif isinstance(item, dict):
                        self._add_content_item(story, item, styles, body_style, subheading_style)

            # Section tables
            tables = section.get("tables", [])
            for table_data in tables:
                self._add_table(story, table_data)

            story.append(Spacer(1, 0.25 * inch))

        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    def _add_content_item(
        self,
        story: list,
        item: dict,
        styles,
        body_style,
        subheading_style,
    ):
        """Add a content item to the story."""
        item_type = item.get("type", "paragraph")

        if item_type == "paragraph":
            text = item.get("text", "")
            story.append(Paragraph(text, body_style))

        elif item_type == "heading":
            text = item.get("text", "")
            story.append(Paragraph(text, subheading_style))

        elif item_type == "bullet_list":
            items = item.get("items", [])
            for bullet in items:
                story.append(Paragraph(f"• {bullet}", body_style))

        elif item_type == "numbered_list":
            items = item.get("items", [])
            for idx, numbered in enumerate(items, 1):
                story.append(Paragraph(f"{idx}. {numbered}", body_style))

        elif item_type == "key_value":
            key = item.get("key", "")
            value = item.get("value", "")
            story.append(Paragraph(f"<b>{key}:</b> {value}", body_style))

    def _add_table(self, story: list, table_data: dict):
        """Add a table to the story."""
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.platypus import Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet

        styles = getSampleStyleSheet()

        title = table_data.get("title", "")
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])

        if not rows:
            return

        # Add table title
        if title:
            story.append(Spacer(1, 0.15 * inch))
            story.append(
                Paragraph(f"<b>{title}</b>", styles["Normal"])
            )
            story.append(Spacer(1, 0.1 * inch))

        # Build table data
        if headers:
            data = [headers] + rows
        else:
            data = rows

        # Create table
        table = Table(data)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]))

        story.append(table)
        story.append(Spacer(1, 0.2 * inch))


@register_tool
class ReportTemplateTool(MCPTool):
    """
    Tool for generating reports from templates.
    """

    name = "report_template"
    description = "Generate reports using predefined templates"
    version = "1.0.0"

    # Predefined templates
    TEMPLATES = {
        "supply_chain_analysis": {
            "sections": [
                {"title": "Executive Summary", "type": "summary"},
                {"title": "Supply Chain Overview", "type": "content"},
                {"title": "Risk Assessment", "type": "content"},
                {"title": "Cost Analysis", "type": "content"},
                {"title": "Recommendations", "type": "content"},
                {"title": "Appendix: Data Sources", "type": "appendix"},
            ]
        },
        "market_assessment": {
            "sections": [
                {"title": "Executive Summary", "type": "summary"},
                {"title": "Market Overview", "type": "content"},
                {"title": "Supply Analysis", "type": "content"},
                {"title": "Demand Analysis", "type": "content"},
                {"title": "Price Trends", "type": "content"},
                {"title": "Outlook", "type": "content"},
            ]
        },
        "policy_brief": {
            "sections": [
                {"title": "Summary", "type": "summary"},
                {"title": "Issue Background", "type": "content"},
                {"title": "Analysis", "type": "content"},
                {"title": "Policy Options", "type": "content"},
                {"title": "Recommendations", "type": "content"},
            ]
        },
    }

    AGENT_INSTRUCTIONS = """
    You generate reports using predefined templates. Available templates:
    - supply_chain_analysis: Comprehensive supply chain assessment
    - market_assessment: Market overview and analysis
    - policy_brief: Short policy-focused document
    """

    def __init__(self, session_logger: Optional[SessionLogger] = None):
        super().__init__(session_logger)
        self._doc_generator = DocumentGeneratorTool(session_logger)

    def execute(self, input_data: dict) -> ToolResult:
        """Generate report from template."""
        template_name = input_data.get("template")
        if template_name not in self.TEMPLATES:
            return ToolResult.error(
                f"Unknown template: {template_name}. "
                f"Available: {list(self.TEMPLATES.keys())}"
            )

        title = input_data.get("title", f"MICA {template_name.replace('_', ' ').title()}")
        content = input_data.get("content", {})

        template = self.TEMPLATES[template_name]
        sections = []

        for section_template in template["sections"]:
            section_title = section_template["title"]
            section_content = content.get(section_title, content.get(section_title.lower(), ""))

            sections.append({
                "title": section_title,
                "content": section_content or f"[Content for {section_title} to be added]",
            })

        return self._doc_generator.execute({
            "title": title,
            "sections": sections,
            "metadata": input_data.get("metadata", {}),
            "output_path": input_data.get("output_path"),
        })

    def list_templates(self) -> dict:
        """List available templates."""
        return {
            name: {
                "sections": [s["title"] for s in template["sections"]]
            }
            for name, template in self.TEMPLATES.items()
        }
