"""
MICA Architecture Diagram Generator

Generates professional architecture diagrams for MICA using graphviz.

Usage:
    pip install graphviz
    python generate_architecture.py
"""

from graphviz import Digraph


def mica_architecture():
    """
    MICA (Materials Intelligence Co-Analyst) architecture diagram.
    Shows the multi-agent workflow with human-in-the-loop pattern.
    """
    dot = Digraph('MICA_Architecture')
    dot.attr(rankdir='TB',
             fontsize='20',
             fontname='Helvetica-Bold',
             labelloc='t',
             label='MICA: Materials Intelligence Co-Analyst',
             bgcolor='#FAFAFA',
             pad='0.3',
             nodesep='0.4',
             ranksep='0.5',
             dpi='300')

    # Set default node attributes
    dot.node_attr.update(fontname='Helvetica', fontsize='12')
    dot.edge_attr.update(fontname='Helvetica', fontsize='10')

    # User Interface Layer
    with dot.subgraph(name='cluster_ui') as ui:
        ui.attr(label='User Interface',
                style='rounded,filled',
                fillcolor='#E3F2FD',
                color='#1565C0',
                penwidth='2',
                fontsize='14',
                fontname='Helvetica-Bold')
        ui.node('webui', 'Open WebUI\n\nChat Interface',
                shape='box',
                style='rounded,filled',
                fillcolor='#42A5F5',
                fontcolor='white',
                fontsize='12',
                penwidth='0')
        ui.node('pipelines', 'Pipelines Server\n\nStreaming Protocol',
                shape='box',
                style='rounded,filled',
                fillcolor='#64B5F6',
                fontcolor='white',
                fontsize='12',
                penwidth='0')

    # MICA Backend
    with dot.subgraph(name='cluster_backend') as backend:
        backend.attr(label='MICA Backend',
                     style='rounded,filled',
                     fillcolor='#E8F5E9',
                     color='#2E7D32',
                     penwidth='2',
                     fontsize='14',
                     fontname='Helvetica-Bold')

        backend.node('api', 'FastAPI\n\nREST + WebSocket',
                     shape='box',
                     style='rounded,filled',
                     fillcolor='#66BB6A',
                     fontcolor='white',
                     fontsize='12',
                     penwidth='0')

        # LangGraph Orchestrator
        with backend.subgraph(name='cluster_orchestrator') as orch:
            orch.attr(label='LangGraph Orchestrator',
                      style='rounded,filled',
                      fillcolor='#FFF8E1',
                      color='#F9A825',
                      penwidth='1.5',
                      fontsize='12',
                      fontname='Helvetica-Bold')

            orch.node('research', 'Preliminary\nResearch',
                      shape='box',
                      style='rounded,filled',
                      fillcolor='#FFCA28',
                      fontcolor='#424242',
                      fontsize='11',
                      penwidth='0')
            orch.node('plan', 'Plan\nGeneration',
                      shape='box',
                      style='rounded,filled',
                      fillcolor='#FFCA28',
                      fontcolor='#424242',
                      fontsize='11',
                      penwidth='0')
            orch.node('approval', 'Human\nApproval',
                      shape='diamond',
                      style='filled',
                      fillcolor='#FF7043',
                      fontcolor='white',
                      fontsize='11',
                      penwidth='0')
            orch.node('execute', 'Plan\nExecution',
                      shape='box',
                      style='rounded,filled',
                      fillcolor='#FFCA28',
                      fontcolor='#424242',
                      fontsize='11',
                      penwidth='0')
            orch.node('summary', 'Summary\nGeneration',
                      shape='box',
                      style='rounded,filled',
                      fillcolor='#FFCA28',
                      fontcolor='#424242',
                      fontsize='11',
                      penwidth='0')
            orch.node('feedback', 'User\nFeedback',
                      shape='diamond',
                      style='filled',
                      fillcolor='#FF7043',
                      fontcolor='white',
                      fontsize='11',
                      penwidth='0')

    # MCP Tools
    with dot.subgraph(name='cluster_tools') as tools:
        tools.attr(label='MCP Tools',
                   style='rounded,filled',
                   fillcolor='#F3E5F5',
                   color='#7B1FA2',
                   penwidth='2',
                   fontsize='14',
                   fontname='Helvetica-Bold')

        tools.node('web_search', 'Web Search\n\nFederal Docs',
                   shape='box',
                   style='rounded,filled',
                   fillcolor='#AB47BC',
                   fontcolor='white',
                   fontsize='11',
                   penwidth='0')
        tools.node('pdf_rag', 'PDF RAG\n\nChromaDB',
                   shape='box',
                   style='rounded,filled',
                   fillcolor='#AB47BC',
                   fontcolor='white',
                   fontsize='11',
                   penwidth='0')
        tools.node('excel', 'Excel Handler\n\nData Analysis',
                   shape='box',
                   style='rounded,filled',
                   fillcolor='#AB47BC',
                   fontcolor='white',
                   fontsize='11',
                   penwidth='0')
        tools.node('code_agent', 'Code Agent\n\nPython Exec',
                   shape='box',
                   style='rounded,filled',
                   fillcolor='#AB47BC',
                   fontcolor='white',
                   fontsize='11',
                   penwidth='0')
        tools.node('doc_gen', 'Doc Generator\n\nPDF Reports',
                   shape='box',
                   style='rounded,filled',
                   fillcolor='#AB47BC',
                   fontcolor='white',
                   fontsize='11',
                   penwidth='0')
        tools.node('simulation', 'Simulation\n\nSupply Chain',
                   shape='box',
                   style='rounded,filled',
                   fillcolor='#CE93D8',
                   fontcolor='#4A148C',
                   fontsize='11',
                   penwidth='0')

    # LLM Providers
    with dot.subgraph(name='cluster_llm') as llm:
        llm.attr(label='LLM Providers',
                 style='rounded,filled',
                 fillcolor='#FFEBEE',
                 color='#C62828',
                 penwidth='2',
                 fontsize='14',
                 fontname='Helvetica-Bold')

        llm.node('argo', 'Argo API\n\nClaude, GPT',
                 shape='ellipse',
                 style='filled',
                 fillcolor='#EF5350',
                 fontcolor='white',
                 fontsize='11',
                 penwidth='0')
        llm.node('gemini', 'Google API\n\nGemini',
                 shape='ellipse',
                 style='filled',
                 fillcolor='#EF5350',
                 fontcolor='white',
                 fontsize='11',
                 penwidth='0')

    # Storage
    with dot.subgraph(name='cluster_storage') as storage:
        storage.attr(label='Storage',
                     style='rounded,filled',
                     fillcolor='#ECEFF1',
                     color='#546E7A',
                     penwidth='2',
                     fontsize='14',
                     fontname='Helvetica-Bold')

        storage.node('sessions', 'Session Logs\n\nJSON + Artifacts',
                     shape='cylinder',
                     style='filled',
                     fillcolor='#78909C',
                     fontcolor='white',
                     fontsize='11',
                     penwidth='0')
        storage.node('chroma', 'ChromaDB\n\nVector Store',
                     shape='cylinder',
                     style='filled',
                     fillcolor='#78909C',
                     fontcolor='white',
                     fontsize='11',
                     penwidth='0')

    # Main flow connections
    dot.edge('webui', 'pipelines', color='#1565C0', penwidth='2', arrowhead='vee')
    dot.edge('pipelines', 'api', label='HTTP\nStreaming', color='#1565C0', penwidth='2', arrowhead='vee')

    # Orchestrator internal flow
    dot.edge('api', 'research', color='#2E7D32', penwidth='2', arrowhead='vee')
    dot.edge('research', 'plan', color='#F9A825', penwidth='1.5', arrowhead='vee')
    dot.edge('plan', 'approval', color='#F9A825', penwidth='1.5', arrowhead='vee')
    dot.edge('approval', 'execute', label='approved', color='#4CAF50', penwidth='1.5', arrowhead='vee')
    dot.edge('approval', 'plan', label='rejected', color='#F44336', penwidth='1', style='dashed', arrowhead='vee')
    dot.edge('execute', 'summary', color='#F9A825', penwidth='1.5', arrowhead='vee')
    dot.edge('summary', 'feedback', color='#F9A825', penwidth='1.5', arrowhead='vee')
    dot.edge('feedback', 'research', label='follow-up', color='#2196F3', penwidth='1', style='dashed', arrowhead='vee')

    # Execute to tools
    dot.edge('execute', 'web_search', color='#7B1FA2', penwidth='1.5', arrowhead='vee')
    dot.edge('execute', 'pdf_rag', color='#7B1FA2', penwidth='1.5', arrowhead='vee')
    dot.edge('execute', 'excel', color='#7B1FA2', penwidth='1.5', arrowhead='vee')
    dot.edge('execute', 'code_agent', color='#7B1FA2', penwidth='1.5', arrowhead='vee')
    dot.edge('execute', 'doc_gen', color='#7B1FA2', penwidth='1.5', arrowhead='vee')
    dot.edge('execute', 'simulation', color='#7B1FA2', penwidth='1.5', style='dashed', arrowhead='vee')

    # LLM connections
    dot.edge('research', 'argo', style='dashed', color='#C62828', penwidth='1', arrowhead='open')
    dot.edge('plan', 'argo', style='dashed', color='#C62828', penwidth='1', arrowhead='open')
    dot.edge('summary', 'argo', style='dashed', color='#C62828', penwidth='1', arrowhead='open')
    dot.edge('code_agent', 'argo', style='dashed', color='#C62828', penwidth='1', arrowhead='open')

    # Storage connections
    dot.edge('api', 'sessions', style='dashed', color='#546E7A', penwidth='1', arrowhead='open')
    dot.edge('pdf_rag', 'chroma', style='dashed', color='#546E7A', penwidth='1', arrowhead='open')

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

    # Main architecture
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

    print("\nDone! Generated 3 architecture diagrams.")
