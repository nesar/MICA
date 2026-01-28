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
        """Generate professional PDF document."""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import inch
            from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
            from reportlab.platypus import (
                Paragraph,
                SimpleDocTemplate,
                Spacer,
                Table,
                TableStyle,
                PageBreak,
                HRFlowable,
                ListFlowable,
                ListItem,
            )
            from reportlab.platypus.frames import Frame
            from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate
        except ImportError:
            raise ImportError("reportlab is required. Install with: pip install reportlab")

        buffer = io.BytesIO()

        # Professional color scheme
        NAVY = colors.HexColor("#1a365d")
        DARK_BLUE = colors.HexColor("#2c5282")
        LIGHT_BLUE = colors.HexColor("#ebf8ff")
        GRAY = colors.HexColor("#4a5568")
        LIGHT_GRAY = colors.HexColor("#f7fafc")
        ACCENT = colors.HexColor("#3182ce")

        # Create document with headers/footers
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=1 * inch,
            bottomMargin=0.75 * inch,
        )

        # Get base styles
        styles = getSampleStyleSheet()

        # Professional custom styles
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=28,
            textColor=NAVY,
            spaceAfter=20,
            alignment=TA_CENTER,
        )

        subtitle_style = ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=14,
            textColor=GRAY,
            spaceAfter=30,
            alignment=TA_CENTER,
        )

        heading1_style = ParagraphStyle(
            "Heading1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            textColor=NAVY,
            spaceBefore=24,
            spaceAfter=12,
            borderColor=NAVY,
            borderWidth=0,
            borderPadding=0,
        )

        heading2_style = ParagraphStyle(
            "Heading2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=DARK_BLUE,
            spaceBefore=18,
            spaceAfter=8,
        )

        heading3_style = ParagraphStyle(
            "Heading3",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=12,
            textColor=GRAY,
            spaceBefore=12,
            spaceAfter=6,
        )

        body_style = ParagraphStyle(
            "BodyText",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            textColor=colors.black,
            spaceBefore=4,
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            leading=14,
        )

        bullet_style = ParagraphStyle(
            "BulletText",
            parent=body_style,
            leftIndent=20,
            bulletIndent=10,
        )

        caption_style = ParagraphStyle(
            "Caption",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=9,
            textColor=GRAY,
            spaceBefore=4,
            spaceAfter=12,
            alignment=TA_CENTER,
        )

        link_style = ParagraphStyle(
            "Link",
            parent=body_style,
            textColor=ACCENT,
            fontSize=9,
        )

        footer_style = ParagraphStyle(
            "Footer",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            textColor=GRAY,
            alignment=TA_CENTER,
        )

        # Build content
        story = []

        # === COVER PAGE ===
        story.append(Spacer(1, 1.5 * inch))

        # Organization header
        story.append(Paragraph("MATERIALS INTELLIGENCE CO-ANALYST", subtitle_style))
        story.append(Spacer(1, 0.3 * inch))

        # Horizontal line
        story.append(HRFlowable(width="80%", thickness=2, color=NAVY, spaceBefore=10, spaceAfter=20))

        # Title
        story.append(Paragraph(title, title_style))

        # Subtitle with date
        report_date = datetime.now().strftime("%B %d, %Y")
        story.append(Paragraph(f"Analysis Report | {report_date}", subtitle_style))

        story.append(Spacer(1, 1 * inch))

        # Metadata box
        if metadata:
            meta_items = []
            for k, v in metadata.items():
                meta_items.append(f"<b>{k}:</b> {v}")
            meta_text = " | ".join(meta_items)
            story.append(Paragraph(meta_text, ParagraphStyle(
                "MetaText",
                parent=body_style,
                alignment=TA_CENTER,
                fontSize=10,
                textColor=GRAY,
            )))

        story.append(Spacer(1, 2 * inch))

        # Footer on cover
        story.append(HRFlowable(width="60%", thickness=1, color=GRAY, spaceBefore=20, spaceAfter=10))
        story.append(Paragraph(
            "Generated by MICA (Materials Intelligence Co-Analyst)<br/>"
            "Department of Energy Critical Materials Analysis System",
            footer_style
        ))

        story.append(PageBreak())

        # === TABLE OF CONTENTS ===
        story.append(Paragraph("Table of Contents", heading1_style))
        story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GRAY, spaceBefore=5, spaceAfter=15))

        toc_style = ParagraphStyle(
            "TOC",
            parent=body_style,
            fontSize=11,
            spaceBefore=8,
            spaceAfter=8,
        )

        for i, section in enumerate(sections, 1):
            section_title = section.get("title", f"Section {i}")
            # Add dots for visual alignment
            story.append(Paragraph(
                f"<b>{i}.</b>  {section_title}",
                toc_style
            ))

        story.append(PageBreak())

        # === MAIN CONTENT ===
        references_collected = []

        for i, section in enumerate(sections, 1):
            section_title = section.get("title", f"Section {i}")

            # Section header with line
            story.append(Paragraph(f"{i}. {section_title}", heading1_style))
            story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_BLUE, spaceBefore=2, spaceAfter=15))

            # Section content
            content = section.get("content", "")
            if isinstance(content, str):
                # Process markdown-like content
                self._process_text_content(story, content, body_style, heading2_style, heading3_style, bullet_style, link_style, references_collected)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        self._process_text_content(story, item, body_style, heading2_style, heading3_style, bullet_style, link_style, references_collected)
                    elif isinstance(item, dict):
                        self._add_content_item(story, item, styles, body_style, heading2_style)

            # Section tables
            tables = section.get("tables", [])
            for table_data in tables:
                self._add_table(story, table_data, caption_style)

            # References for this section
            refs = section.get("references", [])
            if refs:
                references_collected.extend(refs)

            story.append(Spacer(1, 0.3 * inch))

        # === REFERENCES SECTION ===
        if references_collected:
            story.append(PageBreak())
            story.append(Paragraph("References", heading1_style))
            story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_BLUE, spaceBefore=2, spaceAfter=15))

            ref_style = ParagraphStyle(
                "Reference",
                parent=body_style,
                fontSize=9,
                leftIndent=20,
                firstLineIndent=-20,
                spaceBefore=6,
                spaceAfter=6,
            )

            for idx, ref in enumerate(references_collected, 1):
                if isinstance(ref, dict):
                    ref_text = f"[{idx}] "
                    if ref.get("title"):
                        ref_text += f"<b>{ref['title']}</b>. "
                    if ref.get("source"):
                        ref_text += f"{ref['source']}. "
                    if ref.get("url"):
                        ref_text += f'<link href="{ref["url"]}">{ref["url"]}</link>'
                    if ref.get("date"):
                        ref_text += f" (Accessed: {ref['date']})"
                else:
                    ref_text = f"[{idx}] {ref}"
                story.append(Paragraph(ref_text, ref_style))

        # === FOOTER/DISCLAIMER ===
        story.append(Spacer(1, 0.5 * inch))
        story.append(HRFlowable(width="100%", thickness=1, color=GRAY, spaceBefore=20, spaceAfter=10))

        disclaimer_style = ParagraphStyle(
            "Disclaimer",
            parent=body_style,
            fontSize=8,
            textColor=GRAY,
            alignment=TA_CENTER,
            leading=10,
        )

        story.append(Paragraph(
            "This report was generated by MICA (Materials Intelligence Co-Analyst), "
            "an AI-powered analysis system developed for the Department of Energy. "
            "The information contained herein is compiled from publicly available sources "
            "and should be verified before use in decision-making.",
            disclaimer_style
        ))

        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph(
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            disclaimer_style
        ))

        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    def _process_text_content(
        self,
        story: list,
        content: str,
        body_style,
        heading2_style,
        heading3_style,
        bullet_style,
        link_style,
        references_collected: list,
    ):
        """Process text content with markdown-like formatting."""
        import re
        from reportlab.platypus import Paragraph

        lines = content.split("\n")
        current_list = []
        in_list = False

        for line in lines:
            line = line.strip()
            if not line:
                if in_list and current_list:
                    # End current list
                    for item in current_list:
                        story.append(Paragraph(f"* {item}", bullet_style))
                    current_list = []
                    in_list = False
                continue

            # Check for headings (## or ###)
            if line.startswith("### "):
                if in_list and current_list:
                    for item in current_list:
                        story.append(Paragraph(f"* {item}", bullet_style))
                    current_list = []
                    in_list = False
                story.append(Paragraph(line[4:], heading3_style))
            elif line.startswith("## "):
                if in_list and current_list:
                    for item in current_list:
                        story.append(Paragraph(f"* {item}", bullet_style))
                    current_list = []
                    in_list = False
                story.append(Paragraph(line[3:], heading2_style))
            # Check for bullet points
            elif line.startswith("- ") or line.startswith("* "):
                in_list = True
                current_list.append(line[2:])
            # Check for numbered lists
            elif re.match(r"^\d+\.\s", line):
                if in_list and current_list:
                    for item in current_list:
                        story.append(Paragraph(f"* {item}", bullet_style))
                    current_list = []
                story.append(Paragraph(line, bullet_style))
            # Regular paragraph
            else:
                if in_list and current_list:
                    for item in current_list:
                        story.append(Paragraph(f"* {item}", bullet_style))
                    current_list = []
                    in_list = False

                # Process inline formatting
                # Bold: **text** or __text__
                line = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', line)
                line = re.sub(r'__(.+?)__', r'<b>\1</b>', line)
                # Italic: *text* or _text_
                line = re.sub(r'\*(.+?)\*', r'<i>\1</i>', line)
                line = re.sub(r'_(.+?)_', r'<i>\1</i>', line)
                # Links: [text](url)
                line = re.sub(r'\[(.+?)\]\((.+?)\)', r'<link href="\2">\1</link>', line)

                story.append(Paragraph(line, body_style))

        # Flush remaining list items
        if current_list:
            for item in current_list:
                story.append(Paragraph(f"* {item}", bullet_style))

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

    def _add_table(self, story: list, table_data: dict, caption_style=None):
        """Add a professionally styled table to the story."""
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.platypus import Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER

        styles = getSampleStyleSheet()

        # Professional colors
        NAVY = colors.HexColor("#1a365d")
        LIGHT_BLUE = colors.HexColor("#ebf8ff")
        GRAY = colors.HexColor("#4a5568")

        if caption_style is None:
            caption_style = ParagraphStyle(
                "TableCaption",
                parent=styles["Normal"],
                fontName="Helvetica-Oblique",
                fontSize=9,
                textColor=GRAY,
                alignment=TA_CENTER,
                spaceBefore=4,
                spaceAfter=12,
            )

        title = table_data.get("title", "")
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])
        source = table_data.get("source", "")

        if not rows:
            return

        # Add table title
        if title:
            story.append(Spacer(1, 0.2 * inch))
            story.append(Paragraph(f"<b>Table: {title}</b>", ParagraphStyle(
                "TableTitle",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=10,
                textColor=NAVY,
                spaceBefore=10,
                spaceAfter=8,
            )))

        # Build table data
        if headers:
            data = [headers] + rows
        else:
            data = rows

        # Create table with professional styling
        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([
            # Header row
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
            ("TOPPADDING", (0, 0), (-1, 0), 10),
            # Data rows - alternating colors
            ("BACKGROUND", (0, 1), (-1, -1), colors.white),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
            ("TOPPADDING", (0, 1), (-1, -1), 6),
            # Alternating row colors
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BLUE]),
            # Alignment
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            # Grid - subtle
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("LINEBELOW", (0, 0), (-1, 0), 1.5, NAVY),
        ]))

        story.append(table)

        # Add source citation if provided
        if source:
            story.append(Paragraph(f"Source: {source}", caption_style))

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
