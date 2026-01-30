"""
MICA Architecture Diagram Generator

Generates professional architecture diagrams for MICA using graphviz.

Usage:
    pip install graphviz
    python generate_architecture.py
"""

from graphviz import Digraph

# Common style settings
COMMON_GRAPH_ATTRS = {
    'fontname': 'Helvetica-Bold',
    'bgcolor': '#FAFAFA',
    'dpi': '300',
    'splines': 'ortho',  # Orthogonal edges everywhere
}

COMMON_NODE_ATTRS = {
    'fontname': 'Helvetica',
    'fontsize': '10',
}

COMMON_EDGE_ATTRS = {
    'fontname': 'Helvetica',
    'fontsize': '9',
}

# Color palette
COLORS = {
    'ui': {'fill': '#E3F2FD', 'border': '#1565C0', 'node': '#42A5F5'},
    'backend': {'fill': '#E8F5E9', 'border': '#2E7D32', 'node': '#66BB6A'},
    'orchestrator': {'fill': '#FFF8E1', 'border': '#F9A825', 'node': '#FFCA28'},
    'hitl': {'fill': '#FF7043', 'border': '#E64A19'},
    'tools': {'fill': '#F3E5F5', 'border': '#7B1FA2', 'node': '#AB47BC'},
    'llm': {'fill': '#FFEBEE', 'border': '#C62828', 'node': '#EF5350'},
    'storage': {'fill': '#ECEFF1', 'border': '#546E7A', 'node': '#78909C'},
    'output': {'fill': '#FFF9C4', 'border': '#F57F17'},
}


def mica_overview():
    """
    Simplified high-level overview of MICA.
    Shows the big picture with abstract components.
    """
    dot = Digraph('MICA_Overview')
    dot.attr(rankdir='TB',
             fontsize='16',
             labelloc='t',
             label='MICA: High-Level Overview',
             pad='0.2',
             nodesep='0.3',
             ranksep='0.4',
             margin='0.1',
             **COMMON_GRAPH_ATTRS)

    dot.node_attr.update(**COMMON_NODE_ATTRS)
    dot.edge_attr.update(**COMMON_EDGE_ATTRS)

    # User
    dot.node('user', 'User',
             shape='ellipse', style='filled',
             fillcolor=COLORS['ui']['fill'],
             color=COLORS['ui']['border'],
             fontcolor=COLORS['ui']['border'],
             fontsize='12', penwidth='2')

    # Chat Interface
    dot.node('chat', 'Chat Interface',
             shape='box', style='rounded,filled',
             fillcolor=COLORS['ui']['node'],
             fontcolor='white', penwidth='0')

    # MICA Core cluster
    with dot.subgraph(name='cluster_core') as core:
        core.attr(label='MICA Core',
                  style='rounded,filled',
                  fillcolor=COLORS['backend']['fill'],
                  color=COLORS['backend']['border'],
                  penwidth='2', fontsize='11',
                  fontname='Helvetica-Bold',
                  margin='8')

        core.node('orchestrator', 'LangGraph\nOrchestrator',
                  shape='box', style='rounded,filled',
                  fillcolor=COLORS['orchestrator']['node'],
                  fontcolor='#424242', penwidth='0')

        core.node('hitl', 'Human-in-Loop',
                  shape='diamond', style='filled',
                  fillcolor=COLORS['hitl']['fill'],
                  fontcolor='white', penwidth='0',
                  width='1.2', height='0.8')

    # External services - vertical stack
    with dot.subgraph(name='cluster_services') as svc:
        svc.attr(label='External Services',
                 style='rounded,filled',
                 fillcolor='#FAFAFA',
                 color='#9E9E9E',
                 penwidth='1', fontsize='10',
                 fontname='Helvetica-Bold',
                 margin='8')

        svc.node('llm', 'LLM Providers',
                 shape='box', style='rounded,filled',
                 fillcolor=COLORS['llm']['node'],
                 fontcolor='white', penwidth='0')

        svc.node('tools', 'MCP Tools',
                 shape='box', style='rounded,filled',
                 fillcolor=COLORS['tools']['node'],
                 fontcolor='white', penwidth='0')

        svc.node('local_db', 'Local Database',
                 shape='folder', style='filled',
                 fillcolor='#7E57C2',
                 fontcolor='white', penwidth='0')

        svc.node('storage', 'Storage',
                 shape='cylinder', style='filled',
                 fillcolor=COLORS['storage']['node'],
                 fontcolor='white', penwidth='0')

    # Output
    dot.node('report', 'PDF Report',
             shape='note', style='filled',
             fillcolor=COLORS['output']['fill'],
             fontcolor=COLORS['output']['border'],
             penwidth='1', color=COLORS['output']['border'])

    # Edges - main flow
    dot.edge('user', 'chat', color=COLORS['ui']['border'], penwidth='1.5')
    dot.edge('chat', 'orchestrator', color=COLORS['backend']['border'], penwidth='1.5')
    dot.edge('orchestrator', 'hitl', color=COLORS['orchestrator']['border'], penwidth='1.5')

    # Core to services
    dot.edge('orchestrator', 'llm', color=COLORS['llm']['border'], penwidth='1', style='dashed')
    dot.edge('orchestrator', 'tools', color=COLORS['tools']['border'], penwidth='1.5')
    dot.edge('orchestrator', 'storage', color=COLORS['storage']['border'], penwidth='1', style='dashed')

    # Vertical chain in services
    dot.edge('llm', 'tools', style='invis')
    dot.edge('tools', 'local_db', style='invis')
    dot.edge('local_db', 'storage', style='invis')

    # Local database connection
    dot.edge('tools', 'local_db', color='#512DA8', penwidth='1', style='dashed')

    # Output
    dot.edge('tools', 'report', color=COLORS['output']['border'], penwidth='1.5')
    dot.edge('report', 'user', color=COLORS['output']['border'], penwidth='1.5', style='dashed')

    return dot


def mica_architecture():
    """
    MICA detailed architecture diagram.
    Clean layout with vertical MCP tools and structured arrows.
    """
    dot = Digraph('MICA_Architecture')
    dot.attr(rankdir='TB',
             fontsize='14',
             labelloc='t',
             label='MICA: Detailed Architecture',
             pad='0.2',
             nodesep='0.25',
             ranksep='0.35',
             margin='0.1',
             **COMMON_GRAPH_ATTRS)

    dot.node_attr.update(**COMMON_NODE_ATTRS)
    dot.edge_attr.update(**COMMON_EDGE_ATTRS)

    # User Interface Layer
    with dot.subgraph(name='cluster_ui') as ui:
        ui.attr(label='User Interface',
                style='rounded,filled',
                fillcolor=COLORS['ui']['fill'],
                color=COLORS['ui']['border'],
                penwidth='2', fontsize='10',
                fontname='Helvetica-Bold',
                margin='8')
        ui.node('webui', 'Open WebUI',
                shape='box', style='rounded,filled',
                fillcolor=COLORS['ui']['node'],
                fontcolor='white', penwidth='0')
        ui.node('pipelines', 'Pipelines',
                shape='box', style='rounded,filled',
                fillcolor='#64B5F6',
                fontcolor='white', penwidth='0')

    # API Layer
    dot.node('api', 'FastAPI',
             shape='box', style='rounded,filled',
             fillcolor=COLORS['backend']['node'],
             fontcolor='white', penwidth='0')

    # LangGraph Orchestrator
    with dot.subgraph(name='cluster_orchestrator') as orch:
        orch.attr(label='LangGraph Orchestrator',
                  style='rounded,filled',
                  fillcolor=COLORS['orchestrator']['fill'],
                  color=COLORS['orchestrator']['border'],
                  penwidth='2', fontsize='10',
                  fontname='Helvetica-Bold',
                  margin='8')

        orch.node('research', 'Research',
                  shape='box', style='rounded,filled',
                  fillcolor=COLORS['orchestrator']['node'],
                  fontcolor='#424242', penwidth='0')
        orch.node('plan', 'Plan',
                  shape='box', style='rounded,filled',
                  fillcolor=COLORS['orchestrator']['node'],
                  fontcolor='#424242', penwidth='0')
        orch.node('approval', 'Approval',
                  shape='diamond', style='filled',
                  fillcolor=COLORS['hitl']['fill'],
                  fontcolor='white', penwidth='0')
        orch.node('execute', 'Execute',
                  shape='box', style='rounded,filled',
                  fillcolor=COLORS['orchestrator']['node'],
                  fontcolor='#424242', penwidth='0')
        orch.node('summary', 'Summary',
                  shape='box', style='rounded,filled',
                  fillcolor=COLORS['orchestrator']['node'],
                  fontcolor='#424242', penwidth='0')

    # MCP Tools - Vertical
    with dot.subgraph(name='cluster_tools') as tools:
        tools.attr(label='MCP Tools',
                   style='rounded,filled',
                   fillcolor=COLORS['tools']['fill'],
                   color=COLORS['tools']['border'],
                   penwidth='2', fontsize='10',
                   fontname='Helvetica-Bold',
                   margin='8')

        tools.node('tool_hub', 'Dispatcher',
                   shape='box', style='rounded,filled',
                   fillcolor='#9C27B0',
                   fontcolor='white', penwidth='0')
        tools.node('web_search', 'Web Search',
                   shape='box', style='rounded,filled',
                   fillcolor=COLORS['tools']['node'],
                   fontcolor='white', penwidth='0')
        tools.node('local_doc', 'Local Docs',
                   shape='box', style='rounded,filled',
                   fillcolor='#7E57C2',
                   fontcolor='white', penwidth='0')
        tools.node('local_data', 'Local Data',
                   shape='box', style='rounded,filled',
                   fillcolor='#7E57C2',
                   fontcolor='white', penwidth='0')
        tools.node('pdf_rag', 'PDF RAG',
                   shape='box', style='rounded,filled',
                   fillcolor=COLORS['tools']['node'],
                   fontcolor='white', penwidth='0')
        tools.node('code_agent', 'Code Agent',
                   shape='box', style='rounded,filled',
                   fillcolor=COLORS['tools']['node'],
                   fontcolor='white', penwidth='0')
        tools.node('doc_gen', 'Doc Gen',
                   shape='box', style='rounded,filled',
                   fillcolor=COLORS['tools']['node'],
                   fontcolor='white', penwidth='0')

    # LLM Providers - Vertical
    with dot.subgraph(name='cluster_llm') as llm:
        llm.attr(label='LLM',
                 style='rounded,filled',
                 fillcolor=COLORS['llm']['fill'],
                 color=COLORS['llm']['border'],
                 penwidth='2', fontsize='10',
                 fontname='Helvetica-Bold',
                 margin='8')
        llm.node('argo', 'Argo',
                 shape='ellipse', style='filled',
                 fillcolor=COLORS['llm']['node'],
                 fontcolor='white', penwidth='0')
        llm.node('gemini', 'Gemini',
                 shape='ellipse', style='filled',
                 fillcolor=COLORS['llm']['node'],
                 fontcolor='white', penwidth='0')

    # Storage - Vertical
    with dot.subgraph(name='cluster_storage') as storage:
        storage.attr(label='Storage',
                     style='rounded,filled',
                     fillcolor=COLORS['storage']['fill'],
                     color=COLORS['storage']['border'],
                     penwidth='2', fontsize='10',
                     fontname='Helvetica-Bold',
                     margin='8')
        storage.node('sessions', 'Sessions',
                     shape='cylinder', style='filled',
                     fillcolor=COLORS['storage']['node'],
                     fontcolor='white', penwidth='0')
        storage.node('chroma', 'ChromaDB',
                     shape='cylinder', style='filled',
                     fillcolor=COLORS['storage']['node'],
                     fontcolor='white', penwidth='0')
        storage.node('local_database', 'Local DB',
                     shape='folder', style='filled',
                     fillcolor='#7E57C2',
                     fontcolor='white', penwidth='0')

    # Main vertical flow
    dot.edge('webui', 'pipelines', color=COLORS['ui']['border'], penwidth='1.5')
    dot.edge('pipelines', 'api', color=COLORS['ui']['border'], penwidth='1.5')
    dot.edge('api', 'research', color=COLORS['backend']['border'], penwidth='1.5')

    # Orchestrator flow
    dot.edge('research', 'plan', color=COLORS['orchestrator']['border'], penwidth='1')
    dot.edge('plan', 'approval', color=COLORS['orchestrator']['border'], penwidth='1')
    dot.edge('approval', 'execute', color='#4CAF50', penwidth='1')
    dot.edge('execute', 'summary', color=COLORS['orchestrator']['border'], penwidth='1')

    # Execute to tools
    dot.edge('execute', 'tool_hub', color=COLORS['tools']['border'], penwidth='1.5')
    dot.edge('tool_hub', 'web_search', color=COLORS['tools']['border'], penwidth='1')
    dot.edge('tool_hub', 'local_doc', color=COLORS['tools']['border'], penwidth='1')
    dot.edge('tool_hub', 'local_data', color=COLORS['tools']['border'], penwidth='1')
    dot.edge('tool_hub', 'pdf_rag', color=COLORS['tools']['border'], penwidth='1')
    dot.edge('tool_hub', 'code_agent', color=COLORS['tools']['border'], penwidth='1')
    dot.edge('tool_hub', 'doc_gen', color=COLORS['tools']['border'], penwidth='1')

    # LLM connection
    dot.edge('research', 'argo', style='dashed', color=COLORS['llm']['border'], penwidth='1')

    # Storage connections
    dot.edge('api', 'sessions', style='dashed', color=COLORS['storage']['border'], penwidth='1')
    dot.edge('pdf_rag', 'chroma', style='dashed', color=COLORS['storage']['border'], penwidth='1')

    # Local database connections
    dot.edge('local_doc', 'local_database', style='dashed', color='#512DA8', penwidth='1')
    dot.edge('local_data', 'local_database', style='dashed', color='#512DA8', penwidth='1')
    dot.edge('local_doc', 'chroma', style='dashed', color=COLORS['storage']['border'], penwidth='1')

    # Invisible edges for layout
    dot.edge('argo', 'gemini', style='invis')
    dot.edge('sessions', 'chroma', style='invis')
    dot.edge('chroma', 'local_database', style='invis')

    return dot


def mica_workflow():
    """
    MICA workflow state machine diagram.
    """
    dot = Digraph('MICA_Workflow')
    dot.attr(rankdir='LR',
             fontsize='14',
             labelloc='t',
             label='MICA Workflow State Machine',
             pad='0.2',
             nodesep='0.4',
             ranksep='0.5',
             margin='0.1',
             **COMMON_GRAPH_ATTRS)

    dot.node_attr.update(**COMMON_NODE_ATTRS)
    dot.edge_attr.update(**COMMON_EDGE_ATTRS)

    # States
    dot.node('initial', 'INITIAL',
             shape='circle', style='filled',
             fillcolor='#90CAF9', fontcolor=COLORS['ui']['border'],
             penwidth='2', color=COLORS['ui']['border'])

    dot.node('researching', 'RESEARCH',
             shape='box', style='rounded,filled',
             fillcolor='#A5D6A7', fontcolor='#2E7D32', penwidth='0')

    dot.node('plan_proposed', 'PLAN',
             shape='box', style='rounded,filled',
             fillcolor='#FFE082', fontcolor='#F57F17', penwidth='0')

    dot.node('awaiting_approval', 'APPROVAL',
             shape='diamond', style='filled',
             fillcolor='#FFAB91', fontcolor='#BF360C', penwidth='0')

    dot.node('executing', 'EXECUTE',
             shape='box', style='rounded,filled',
             fillcolor='#CE93D8', fontcolor='#6A1B9A', penwidth='0')

    dot.node('completed', 'COMPLETE',
             shape='box', style='rounded,filled',
             fillcolor='#80CBC4', fontcolor='#00695C', penwidth='0')

    dot.node('feedback', 'FEEDBACK',
             shape='diamond', style='filled',
             fillcolor='#FFAB91', fontcolor='#BF360C', penwidth='0')

    dot.node('end', 'END',
             shape='doublecircle', style='filled',
             fillcolor='#78909C', fontcolor='white',
             penwidth='2', color='#455A64')

    # Transitions (use xlabel for ortho compatibility)
    dot.edge('initial', 'researching', xlabel='query', color='#2E7D32', penwidth='1.5')
    dot.edge('researching', 'plan_proposed', color='#F57F17', penwidth='1.5')
    dot.edge('plan_proposed', 'awaiting_approval', color='#BF360C', penwidth='1.5')
    dot.edge('awaiting_approval', 'executing', xlabel='ok', color='#4CAF50', penwidth='1.5')
    dot.edge('awaiting_approval', 'plan_proposed', xlabel='reject', color='#F44336', penwidth='1', style='dashed')
    dot.edge('executing', 'completed', color='#00695C', penwidth='1.5')
    dot.edge('completed', 'feedback', color='#BF360C', penwidth='1.5')
    dot.edge('feedback', 'researching', xlabel='more', color='#2196F3', penwidth='1', style='dashed')
    dot.edge('feedback', 'end', xlabel='done', color='#455A64', penwidth='1.5')

    return dot


def mica_tools_detail():
    """
    Detailed view of MICA MCP tools.
    """
    dot = Digraph('MICA_Tools')
    dot.attr(rankdir='TB',
             fontsize='14',
             labelloc='t',
             label='MICA MCP Tools',
             pad='0.2',
             nodesep='0.2',
             ranksep='0.3',
             margin='0.1',
             **COMMON_GRAPH_ATTRS)

    dot.node_attr.update(**COMMON_NODE_ATTRS)
    dot.edge_attr.update(**COMMON_EDGE_ATTRS)

    # Web Search
    with dot.subgraph(name='cluster_ws') as ws:
        ws.attr(label='web_search',
                style='rounded,filled',
                fillcolor=COLORS['ui']['fill'],
                color=COLORS['ui']['border'],
                penwidth='2', fontsize='10',
                fontname='Helvetica-Bold', margin='6')
        ws.node('ws_main', 'Federal Search',
                shape='box', style='rounded,filled',
                fillcolor=COLORS['ui']['node'], fontcolor='white', penwidth='0')
        ws.node('ws_duck', 'DuckDuckGo',
                shape='ellipse', style='filled', fillcolor='#90CAF9', penwidth='0')
        ws.node('ws_tavily', 'Tavily',
                shape='ellipse', style='filled', fillcolor='#90CAF9', penwidth='0')

    # PDF RAG
    with dot.subgraph(name='cluster_pdf') as pr:
        pr.attr(label='pdf_rag',
                style='rounded,filled',
                fillcolor=COLORS['backend']['fill'],
                color=COLORS['backend']['border'],
                penwidth='2', fontsize='10',
                fontname='Helvetica-Bold', margin='6')
        pr.node('pr_main', 'Doc Analysis',
                shape='box', style='rounded,filled',
                fillcolor=COLORS['backend']['node'], fontcolor='white', penwidth='0')
        pr.node('pr_parse', 'Parse',
                shape='ellipse', style='filled', fillcolor='#A5D6A7', penwidth='0')
        pr.node('pr_embed', 'Embed',
                shape='ellipse', style='filled', fillcolor='#A5D6A7', penwidth='0')
        pr.node('pr_search', 'Search',
                shape='ellipse', style='filled', fillcolor='#A5D6A7', penwidth='0')

    # Code Agent
    with dot.subgraph(name='cluster_code') as ca:
        ca.attr(label='code_agent',
                style='rounded,filled',
                fillcolor='#FFF3E0',
                color='#E65100',
                penwidth='2', fontsize='10',
                fontname='Helvetica-Bold', margin='6')
        ca.node('ca_main', 'Python Exec',
                shape='box', style='rounded,filled',
                fillcolor='#FF9800', fontcolor='white', penwidth='0')
        ca.node('ca_analysis', 'Analysis',
                shape='ellipse', style='filled', fillcolor='#FFCC80', penwidth='0')
        ca.node('ca_viz', 'Plots',
                shape='ellipse', style='filled', fillcolor='#FFCC80', penwidth='0')

    # Doc Generator
    with dot.subgraph(name='cluster_doc') as dg:
        dg.attr(label='doc_generator',
                style='rounded,filled',
                fillcolor=COLORS['tools']['fill'],
                color=COLORS['tools']['border'],
                penwidth='2', fontsize='10',
                fontname='Helvetica-Bold', margin='6')
        dg.node('dg_main', 'PDF Reports',
                shape='box', style='rounded,filled',
                fillcolor=COLORS['tools']['node'], fontcolor='white', penwidth='0')
        dg.node('dg_style', 'Styling',
                shape='ellipse', style='filled', fillcolor='#CE93D8', penwidth='0')
        dg.node('dg_refs', 'References',
                shape='ellipse', style='filled', fillcolor='#CE93D8', penwidth='0')

    # Local Document Search
    with dot.subgraph(name='cluster_local_doc') as ld:
        ld.attr(label='local_doc_search',
                style='rounded,filled',
                fillcolor='#EDE7F6',
                color='#512DA8',
                penwidth='2', fontsize='10',
                fontname='Helvetica-Bold', margin='6')
        ld.node('ld_main', 'Local PDFs',
                shape='box', style='rounded,filled',
                fillcolor='#7E57C2', fontcolor='white', penwidth='0')
        ld.node('ld_index', 'Index',
                shape='ellipse', style='filled', fillcolor='#B39DDB', penwidth='0')
        ld.node('ld_search', 'Search',
                shape='ellipse', style='filled', fillcolor='#B39DDB', penwidth='0')

    # Local Data Analysis
    with dot.subgraph(name='cluster_local_data') as lda:
        lda.attr(label='local_data_analysis',
                 style='rounded,filled',
                 fillcolor='#E8EAF6',
                 color='#303F9F',
                 penwidth='2', fontsize='10',
                 fontname='Helvetica-Bold', margin='6')
        lda.node('lda_main', 'Excel/CSV',
                 shape='box', style='rounded,filled',
                 fillcolor='#5C6BC0', fontcolor='white', penwidth='0')
        lda.node('lda_stats', 'Statistics',
                 shape='ellipse', style='filled', fillcolor='#9FA8DA', penwidth='0')
        lda.node('lda_viz', 'Visualize',
                 shape='ellipse', style='filled', fillcolor='#9FA8DA', penwidth='0')

    # Connections
    dot.edge('ws_main', 'ws_duck', color=COLORS['ui']['border'], penwidth='1')
    dot.edge('ws_main', 'ws_tavily', color=COLORS['ui']['border'], penwidth='1')

    dot.edge('pr_main', 'pr_parse', color=COLORS['backend']['border'], penwidth='1')
    dot.edge('pr_parse', 'pr_embed', color=COLORS['backend']['border'], penwidth='1')
    dot.edge('pr_embed', 'pr_search', color=COLORS['backend']['border'], penwidth='1')

    dot.edge('ca_main', 'ca_analysis', color='#E65100', penwidth='1')
    dot.edge('ca_main', 'ca_viz', color='#E65100', penwidth='1')

    dot.edge('dg_main', 'dg_style', color=COLORS['tools']['border'], penwidth='1')
    dot.edge('dg_main', 'dg_refs', color=COLORS['tools']['border'], penwidth='1')

    dot.edge('ld_main', 'ld_index', color='#512DA8', penwidth='1')
    dot.edge('ld_index', 'ld_search', color='#512DA8', penwidth='1')

    dot.edge('lda_main', 'lda_stats', color='#303F9F', penwidth='1')
    dot.edge('lda_main', 'lda_viz', color='#303F9F', penwidth='1')

    return dot


if __name__ == '__main__':
    print("Generating MICA architecture diagrams...")

    d0 = mica_overview()
    d0.render('mica_overview', format='png', cleanup=True)
    print("  Generated mica_overview.png")

    d1 = mica_architecture()
    d1.render('mica_architecture', format='png', cleanup=True)
    print("  Generated mica_architecture.png")

    d2 = mica_workflow()
    d2.render('mica_workflow', format='png', cleanup=True)
    print("  Generated mica_workflow.png")

    d3 = mica_tools_detail()
    d3.render('mica_tools', format='png', cleanup=True)
    print("  Generated mica_tools.png")

    print("\nDone! Generated 4 architecture diagrams.")
