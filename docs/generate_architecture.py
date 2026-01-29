"""
MICA Architecture Diagram Generator

Generates professional architecture diagrams for MICA using graphviz.

Usage:
    pip install graphviz
    python generate_architecture.py
"""

from graphviz import Digraph


def mica_overview():
    """
    Simplified high-level overview of MICA.
    Shows the big picture with abstract components.
    """
    dot = Digraph('MICA_Overview')
    dot.attr(rankdir='TB',
             fontsize='24',
             fontname='Helvetica-Bold',
             labelloc='t',
             label='MICA: Materials Intelligence Co-Analyst\nHigh-Level Overview',
             bgcolor='white',
             pad='0.5',
             nodesep='0.8',
             ranksep='1.0',
             dpi='300')

    dot.node_attr.update(fontname='Helvetica', fontsize='14')
    dot.edge_attr.update(fontname='Helvetica', fontsize='11')

    # User
    dot.node('user', 'User',
             shape='ellipse',
             style='filled',
             fillcolor='#E3F2FD',
             color='#1565C0',
             fontcolor='#1565C0',
             fontsize='16',
             penwidth='3')

    # Chat Interface
    dot.node('chat', 'Chat Interface\n(Open WebUI)',
             shape='box',
             style='rounded,filled',
             fillcolor='#42A5F5',
             fontcolor='white',
             fontsize='14',
             penwidth='0',
             width='2.5',
             height='0.8')

    # MICA Core - the main box
    with dot.subgraph(name='cluster_core') as core:
        core.attr(label='MICA Core',
                  style='rounded,filled',
                  fillcolor='#E8F5E9',
                  color='#2E7D32',
                  penwidth='3',
                  fontsize='16',
                  fontname='Helvetica-Bold',
                  margin='20')

        core.node('orchestrator', 'LangGraph\nOrchestrator',
                  shape='box',
                  style='rounded,filled',
                  fillcolor='#FFCA28',
                  fontcolor='#424242',
                  fontsize='14',
                  penwidth='0',
                  width='2',
                  height='0.8')

        core.node('hitl', 'Human-in-the-Loop\nApproval',
                  shape='diamond',
                  style='filled',
                  fillcolor='#FF7043',
                  fontcolor='white',
                  fontsize='12',
                  penwidth='0',
                  width='2',
                  height='1.2')

    # External Services row
    dot.node('llm', 'LLM Providers\n(Claude, GPT, Gemini)',
             shape='box',
             style='rounded,filled',
             fillcolor='#EF5350',
             fontcolor='white',
             fontsize='12',
             penwidth='0',
             width='2.5',
             height='0.8')

    dot.node('tools', 'MCP Tools\n(Search, PDF, Code, Reports)',
             shape='box',
             style='rounded,filled',
             fillcolor='#AB47BC',
             fontcolor='white',
             fontsize='12',
             penwidth='0',
             width='3',
             height='0.8')

    dot.node('storage', 'Storage\n(Sessions, ChromaDB)',
             shape='cylinder',
             style='filled',
             fillcolor='#78909C',
             fontcolor='white',
             fontsize='12',
             penwidth='0')

    # Outputs
    dot.node('report', 'Analysis Report\n(PDF)',
             shape='note',
             style='filled',
             fillcolor='#FFF9C4',
             fontcolor='#F57F17',
             fontsize='12',
             penwidth='2',
             color='#F57F17')

    # Main flow - clean vertical layout
    dot.edge('user', 'chat', label='Query', color='#1565C0', penwidth='2', arrowhead='vee')
    dot.edge('chat', 'orchestrator', label='Streaming', color='#2E7D32', penwidth='2', arrowhead='vee')
    dot.edge('orchestrator', 'hitl', color='#F9A825', penwidth='2', arrowhead='vee')
    dot.edge('hitl', 'orchestrator', label='Feedback', color='#FF7043', penwidth='1.5', style='dashed', arrowhead='vee')

    # Side connections - use constraint=false for horizontal
    dot.edge('orchestrator', 'llm', color='#C62828', penwidth='1.5', style='dashed', constraint='false')
    dot.edge('orchestrator', 'tools', color='#7B1FA2', penwidth='2', arrowhead='vee')
    dot.edge('orchestrator', 'storage', color='#546E7A', penwidth='1.5', style='dashed')

    # Output
    dot.edge('tools', 'report', color='#F57F17', penwidth='2', arrowhead='vee')
    dot.edge('report', 'user', label='Results', color='#F57F17', penwidth='2', arrowhead='vee', constraint='false')

    # Force layout
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('llm')
        s.node('tools')
        s.node('storage')

    return dot


def mica_architecture():
    """
    MICA detailed architecture diagram.
    Clean layout with vertical MCP tools and structured arrows.
    """
    dot = Digraph('MICA_Architecture')
    dot.attr(rankdir='TB',
             fontsize='20',
             fontname='Helvetica-Bold',
             labelloc='t',
             label='MICA: Detailed Architecture',
             bgcolor='#FAFAFA',
             pad='0.4',
             nodesep='0.5',
             ranksep='0.6',
             dpi='300',
             splines='ortho')  # Orthogonal edges for cleaner lines

    dot.node_attr.update(fontname='Helvetica', fontsize='11')
    dot.edge_attr.update(fontname='Helvetica', fontsize='9')

    # User Interface Layer
    with dot.subgraph(name='cluster_ui') as ui:
        ui.attr(label='User Interface',
                style='rounded,filled',
                fillcolor='#E3F2FD',
                color='#1565C0',
                penwidth='2',
                fontsize='13',
                fontname='Helvetica-Bold')
        ui.node('webui', 'Open WebUI',
                shape='box',
                style='rounded,filled',
                fillcolor='#42A5F5',
                fontcolor='white',
                penwidth='0')
        ui.node('pipelines', 'Pipelines Server',
                shape='box',
                style='rounded,filled',
                fillcolor='#64B5F6',
                fontcolor='white',
                penwidth='0')

    # API Layer
    dot.node('api', 'FastAPI Backend',
             shape='box',
             style='rounded,filled',
             fillcolor='#66BB6A',
             fontcolor='white',
             penwidth='0')

    # LangGraph Orchestrator
    with dot.subgraph(name='cluster_orchestrator') as orch:
        orch.attr(label='LangGraph Orchestrator',
                  style='rounded,filled',
                  fillcolor='#FFF8E1',
                  color='#F9A825',
                  penwidth='2',
                  fontsize='13',
                  fontname='Helvetica-Bold')

        # Workflow nodes in a row
        orch.node('research', 'Research',
                  shape='box', style='rounded,filled', fillcolor='#FFCA28',
                  fontcolor='#424242', penwidth='0')
        orch.node('plan', 'Plan',
                  shape='box', style='rounded,filled', fillcolor='#FFCA28',
                  fontcolor='#424242', penwidth='0')
        orch.node('approval', 'Approval',
                  shape='diamond', style='filled', fillcolor='#FF7043',
                  fontcolor='white', penwidth='0')
        orch.node('execute', 'Execute',
                  shape='box', style='rounded,filled', fillcolor='#FFCA28',
                  fontcolor='#424242', penwidth='0')
        orch.node('summary', 'Summary',
                  shape='box', style='rounded,filled', fillcolor='#FFCA28',
                  fontcolor='#424242', penwidth='0')

    # MCP Tools - Vertical layout in its own cluster
    with dot.subgraph(name='cluster_tools') as tools:
        tools.attr(label='MCP Tools',
                   style='rounded,filled',
                   fillcolor='#F3E5F5',
                   color='#7B1FA2',
                   penwidth='2',
                   fontsize='13',
                   fontname='Helvetica-Bold',
                   rankdir='TB')

        # Single column of tools
        tools.node('tool_hub', 'Tool\nDispatcher',
                   shape='box',
                   style='rounded,filled',
                   fillcolor='#9C27B0',
                   fontcolor='white',
                   penwidth='0')

        tools.node('web_search', 'Web Search',
                   shape='box', style='rounded,filled', fillcolor='#AB47BC',
                   fontcolor='white', penwidth='0')
        tools.node('pdf_rag', 'PDF RAG',
                   shape='box', style='rounded,filled', fillcolor='#AB47BC',
                   fontcolor='white', penwidth='0')
        tools.node('code_agent', 'Code Agent',
                   shape='box', style='rounded,filled', fillcolor='#AB47BC',
                   fontcolor='white', penwidth='0')
        tools.node('doc_gen', 'Doc Generator',
                   shape='box', style='rounded,filled', fillcolor='#AB47BC',
                   fontcolor='white', penwidth='0')
        tools.node('excel', 'Excel Handler',
                   shape='box', style='rounded,filled', fillcolor='#CE93D8',
                   fontcolor='#4A148C', penwidth='0')

    # LLM Providers
    with dot.subgraph(name='cluster_llm') as llm:
        llm.attr(label='LLM Providers',
                 style='rounded,filled',
                 fillcolor='#FFEBEE',
                 color='#C62828',
                 penwidth='2',
                 fontsize='13',
                 fontname='Helvetica-Bold')
        llm.node('argo', 'Argo API\n(Claude, GPT)',
                 shape='ellipse', style='filled', fillcolor='#EF5350',
                 fontcolor='white', penwidth='0')
        llm.node('gemini', 'Gemini API',
                 shape='ellipse', style='filled', fillcolor='#EF5350',
                 fontcolor='white', penwidth='0')

    # Storage
    with dot.subgraph(name='cluster_storage') as storage:
        storage.attr(label='Storage',
                     style='rounded,filled',
                     fillcolor='#ECEFF1',
                     color='#546E7A',
                     penwidth='2',
                     fontsize='13',
                     fontname='Helvetica-Bold')
        storage.node('sessions', 'Sessions',
                     shape='cylinder', style='filled', fillcolor='#78909C',
                     fontcolor='white', penwidth='0')
        storage.node('chroma', 'ChromaDB',
                     shape='cylinder', style='filled', fillcolor='#78909C',
                     fontcolor='white', penwidth='0')

    # Main vertical flow
    dot.edge('webui', 'pipelines', color='#1565C0', penwidth='2')
    dot.edge('pipelines', 'api', color='#1565C0', penwidth='2')
    dot.edge('api', 'research', color='#2E7D32', penwidth='2')

    # Orchestrator internal flow
    dot.edge('research', 'plan', color='#F9A825', penwidth='1.5')
    dot.edge('plan', 'approval', color='#F9A825', penwidth='1.5')
    dot.edge('approval', 'execute', color='#4CAF50', penwidth='1.5')
    dot.edge('execute', 'summary', color='#F9A825', penwidth='1.5')

    # Single connection from execute to tool hub
    dot.edge('execute', 'tool_hub', color='#7B1FA2', penwidth='2')

    # Tool hub to individual tools
    dot.edge('tool_hub', 'web_search', color='#7B1FA2', penwidth='1')
    dot.edge('tool_hub', 'pdf_rag', color='#7B1FA2', penwidth='1')
    dot.edge('tool_hub', 'code_agent', color='#7B1FA2', penwidth='1')
    dot.edge('tool_hub', 'doc_gen', color='#7B1FA2', penwidth='1')
    dot.edge('tool_hub', 'excel', color='#7B1FA2', penwidth='1')

    # LLM connection - single arrow from orchestrator cluster
    dot.edge('research', 'argo', style='dashed', color='#C62828', penwidth='1.5', constraint='false')

    # Storage connections
    dot.edge('api', 'sessions', style='dashed', color='#546E7A', penwidth='1')
    dot.edge('pdf_rag', 'chroma', style='dashed', color='#546E7A', penwidth='1')

    return dot


def mica_workflow():
    """
    MICA workflow state machine diagram.
    Shows the LangGraph state transitions.
    """
    dot = Digraph('MICA_Workflow')
    dot.attr(rankdir='LR',
             fontsize='18',
             fontname='Helvetica-Bold',
             labelloc='t',
             label='MICA Workflow State Machine',
             bgcolor='#FAFAFA',
             pad='0.3',
             nodesep='0.6',
             ranksep='0.8',
             dpi='300')

    dot.node_attr.update(fontname='Helvetica', fontsize='11')
    dot.edge_attr.update(fontname='Helvetica', fontsize='10')

    # States
    dot.node('initial', 'INITIAL',
             shape='circle',
             style='filled',
             fillcolor='#90CAF9',
             fontcolor='#1565C0',
             penwidth='2',
             color='#1565C0')

    dot.node('researching', 'RESEARCHING',
             shape='box',
             style='rounded,filled',
             fillcolor='#A5D6A7',
             fontcolor='#2E7D32',
             penwidth='0')

    dot.node('plan_proposed', 'PLAN\nPROPOSED',
             shape='box',
             style='rounded,filled',
             fillcolor='#FFE082',
             fontcolor='#F57F17',
             penwidth='0')

    dot.node('awaiting_approval', 'AWAITING\nAPPROVAL',
             shape='diamond',
             style='filled',
             fillcolor='#FFAB91',
             fontcolor='#BF360C',
             penwidth='0')

    dot.node('executing', 'EXECUTING',
             shape='box',
             style='rounded,filled',
             fillcolor='#CE93D8',
             fontcolor='#6A1B9A',
             penwidth='0')

    dot.node('completed', 'COMPLETED',
             shape='box',
             style='rounded,filled',
             fillcolor='#80CBC4',
             fontcolor='#00695C',
             penwidth='0')

    dot.node('awaiting_feedback', 'AWAITING\nFEEDBACK',
             shape='diamond',
             style='filled',
             fillcolor='#FFAB91',
             fontcolor='#BF360C',
             penwidth='0')

    dot.node('failed', 'FAILED',
             shape='box',
             style='rounded,filled',
             fillcolor='#EF9A9A',
             fontcolor='#B71C1C',
             penwidth='0')

    dot.node('end', 'END',
             shape='doublecircle',
             style='filled',
             fillcolor='#78909C',
             fontcolor='white',
             penwidth='2',
             color='#455A64')

    # Transitions
    dot.edge('initial', 'researching', label='query', color='#2E7D32', penwidth='2')
    dot.edge('researching', 'plan_proposed', label='complex', color='#F57F17', penwidth='2')
    dot.edge('researching', 'awaiting_feedback', label='simple', color='#00695C', penwidth='2')
    dot.edge('plan_proposed', 'awaiting_approval', color='#BF360C', penwidth='2')
    dot.edge('awaiting_approval', 'executing', label='approved', color='#4CAF50', penwidth='2')
    dot.edge('awaiting_approval', 'plan_proposed', label='rejected', color='#F44336', penwidth='1.5', style='dashed')
    dot.edge('executing', 'completed', color='#00695C', penwidth='2')
    dot.edge('completed', 'awaiting_feedback', color='#BF360C', penwidth='2')
    dot.edge('awaiting_feedback', 'researching', label='follow-up', color='#2196F3', penwidth='1.5', style='dashed')
    dot.edge('awaiting_feedback', 'end', label='done', color='#455A64', penwidth='2')

    # Error paths
    dot.edge('researching', 'failed', style='dashed', color='#B71C1C', penwidth='1')
    dot.edge('executing', 'failed', style='dashed', color='#B71C1C', penwidth='1')
    dot.edge('failed', 'end', color='#B71C1C', penwidth='1.5')

    return dot


def mica_tools_detail():
    """
    Detailed view of MICA MCP tools and their capabilities.
    """
    dot = Digraph('MICA_Tools')
    dot.attr(rankdir='TB',
             fontsize='18',
             fontname='Helvetica-Bold',
             labelloc='t',
             label='MICA MCP Tools',
             bgcolor='#FAFAFA',
             pad='0.3',
             nodesep='0.3',
             ranksep='0.4',
             dpi='300')

    dot.node_attr.update(fontname='Helvetica', fontsize='10')
    dot.edge_attr.update(fontname='Helvetica', fontsize='9')

    # Web Search Tool
    with dot.subgraph(name='cluster_websearch') as ws:
        ws.attr(label='web_search',
                style='rounded,filled',
                fillcolor='#E3F2FD',
                color='#1565C0',
                penwidth='2',
                fontsize='12',
                fontname='Helvetica-Bold')
        ws.node('ws_main', 'Federal Document Search',
                shape='box', style='rounded,filled', fillcolor='#42A5F5', fontcolor='white', penwidth='0')
        ws.node('ws_duck', 'DuckDuckGo', shape='ellipse', style='filled', fillcolor='#90CAF9', penwidth='0')
        ws.node('ws_tavily', 'Tavily', shape='ellipse', style='filled', fillcolor='#90CAF9', penwidth='0')
        ws.node('ws_serp', 'SerpAPI', shape='ellipse', style='filled', fillcolor='#90CAF9', penwidth='0')

    # PDF RAG Tool
    with dot.subgraph(name='cluster_pdfrag') as pr:
        pr.attr(label='pdf_rag',
                style='rounded,filled',
                fillcolor='#E8F5E9',
                color='#2E7D32',
                penwidth='2',
                fontsize='12',
                fontname='Helvetica-Bold')
        pr.node('pr_main', 'Document Analysis',
                shape='box', style='rounded,filled', fillcolor='#66BB6A', fontcolor='white', penwidth='0')
        pr.node('pr_parse', 'PDF Parsing', shape='ellipse', style='filled', fillcolor='#A5D6A7', penwidth='0')
        pr.node('pr_chunk', 'Chunking', shape='ellipse', style='filled', fillcolor='#A5D6A7', penwidth='0')
        pr.node('pr_embed', 'Embeddings', shape='ellipse', style='filled', fillcolor='#A5D6A7', penwidth='0')
        pr.node('pr_search', 'Semantic Search', shape='ellipse', style='filled', fillcolor='#A5D6A7', penwidth='0')

    # Code Agent Tool
    with dot.subgraph(name='cluster_code') as ca:
        ca.attr(label='code_agent',
                style='rounded,filled',
                fillcolor='#FFF3E0',
                color='#E65100',
                penwidth='2',
                fontsize='12',
                fontname='Helvetica-Bold')
        ca.node('ca_main', 'Python Execution',
                shape='box', style='rounded,filled', fillcolor='#FF9800', fontcolor='white', penwidth='0')
        ca.node('ca_analysis', 'Data Analysis', shape='ellipse', style='filled', fillcolor='#FFCC80', penwidth='0')
        ca.node('ca_viz', 'Visualization', shape='ellipse', style='filled', fillcolor='#FFCC80', penwidth='0')
        ca.node('ca_stats', 'Statistics', shape='ellipse', style='filled', fillcolor='#FFCC80', penwidth='0')

    # Document Generator Tool
    with dot.subgraph(name='cluster_docgen') as dg:
        dg.attr(label='doc_generator',
                style='rounded,filled',
                fillcolor='#F3E5F5',
                color='#7B1FA2',
                penwidth='2',
                fontsize='12',
                fontname='Helvetica-Bold')
        dg.node('dg_main', 'PDF Report Generation',
                shape='box', style='rounded,filled', fillcolor='#AB47BC', fontcolor='white', penwidth='0')
        dg.node('dg_template', 'Templates', shape='ellipse', style='filled', fillcolor='#CE93D8', penwidth='0')
        dg.node('dg_style', 'Professional Styling', shape='ellipse', style='filled', fillcolor='#CE93D8', penwidth='0')
        dg.node('dg_refs', 'References', shape='ellipse', style='filled', fillcolor='#CE93D8', penwidth='0')

    # Excel Handler Tool
    with dot.subgraph(name='cluster_excel') as ex:
        ex.attr(label='excel_handler',
                style='rounded,filled',
                fillcolor='#ECEFF1',
                color='#455A64',
                penwidth='2',
                fontsize='12',
                fontname='Helvetica-Bold')
        ex.node('ex_main', 'Spreadsheet Analysis',
                shape='box', style='rounded,filled', fillcolor='#607D8B', fontcolor='white', penwidth='0')
        ex.node('ex_read', 'Read Excel/CSV', shape='ellipse', style='filled', fillcolor='#B0BEC5', penwidth='0')
        ex.node('ex_filter', 'Filter/Query', shape='ellipse', style='filled', fillcolor='#B0BEC5', penwidth='0')
        ex.node('ex_transform', 'Transform', shape='ellipse', style='filled', fillcolor='#B0BEC5', penwidth='0')

    # Internal connections
    dot.edge('ws_main', 'ws_duck', penwidth='1', color='#1565C0')
    dot.edge('ws_main', 'ws_tavily', penwidth='1', color='#1565C0')
    dot.edge('ws_main', 'ws_serp', penwidth='1', color='#1565C0')

    dot.edge('pr_main', 'pr_parse', penwidth='1', color='#2E7D32')
    dot.edge('pr_parse', 'pr_chunk', penwidth='1', color='#2E7D32')
    dot.edge('pr_chunk', 'pr_embed', penwidth='1', color='#2E7D32')
    dot.edge('pr_embed', 'pr_search', penwidth='1', color='#2E7D32')

    dot.edge('ca_main', 'ca_analysis', penwidth='1', color='#E65100')
    dot.edge('ca_main', 'ca_viz', penwidth='1', color='#E65100')
    dot.edge('ca_main', 'ca_stats', penwidth='1', color='#E65100')

    dot.edge('dg_main', 'dg_template', penwidth='1', color='#7B1FA2')
    dot.edge('dg_main', 'dg_style', penwidth='1', color='#7B1FA2')
    dot.edge('dg_main', 'dg_refs', penwidth='1', color='#7B1FA2')

    dot.edge('ex_main', 'ex_read', penwidth='1', color='#455A64')
    dot.edge('ex_main', 'ex_filter', penwidth='1', color='#455A64')
    dot.edge('ex_main', 'ex_transform', penwidth='1', color='#455A64')

    return dot


if __name__ == '__main__':
    print("Generating MICA architecture diagrams...")

    # High-level overview (new)
    d0 = mica_overview()
    d0.render('mica_overview', format='png', cleanup=True)
    print("  Generated mica_overview.png")

    # Detailed architecture (cleaned up)
    d1 = mica_architecture()
    d1.render('mica_architecture', format='png', cleanup=True)
    print("  Generated mica_architecture.png")

    # Workflow state machine
    d2 = mica_workflow()
    d2.render('mica_workflow', format='png', cleanup=True)
    print("  Generated mica_workflow.png")

    # Tools detail
    d3 = mica_tools_detail()
    d3.render('mica_tools', format='png', cleanup=True)
    print("  Generated mica_tools.png")

    print("\nDone! Generated 4 architecture diagrams.")
