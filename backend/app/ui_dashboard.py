import streamlit as st
import requests
import datetime
from streamlit_agraph import agraph, Node, Edge, Config

# ============================================================================
# COMPONENT CONFIGURATION & SURFACE ELEVATION
# ============================================================================
st.set_page_config(
    page_title="Cognitive Data Nexus Console",
    layout="wide",
    initial_sidebar_state="collapsed"
)

API_BASE_URL = "http://127.0.0.1:8000/api"

# Initialize global UI session state wrappers
if "graph_data" not in st.session_state:
    st.session_state.graph_data = {"nodes": [], "edges": []}
if "system_logs" not in st.session_state:
    st.session_state.system_logs = [
        "[SYSTEM] Cognitive Data Nexus core subsystem initialized.",
        "[PERSISTENCE] Local loopback bound to ASGI gateway port 8000.",
        "[ORCHESTRATION] Awaiting multi-format data ingestion stream..."
    ]

def emit_ui_log(event_type: str, details: str):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.system_logs.insert(0, f"[{timestamp}] [{event_type.upper()}] {details}")

# ============================================================================
# VERCEL/GITHUB DARK ELEVATION DESIGN SYSTEM (CSS INJECTION)
# ============================================================================
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background-color: #0d1117; }
    .main { background-color: #0d1117; color: #c9d1d9; }
    .block-container { max-width: 1600px; padding-top: 1.5rem; padding-bottom: 1.5rem; }
    
    /* Typography Overrides */
    h1, h2, h3, h4 { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif !important; font-weight: 600 !important; }
    h1 { color: #f0f6fc !important; font-size: 28px !important; margin-bottom: 4px !important; }
    .subtitle { color: #8b949e; font-size: 14px; margin-bottom: 25px; }
    
    /* Card Component Layouts */
    .enterprise-card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 24px; margin-bottom: 20px; }
    .card-header { color: #58a6ff; font-size: 16px; font-weight: 600; margin-bottom: 16px; border-bottom: 1px solid #21262d; padding-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
    
    /* Horizontal Status Bar Matrix */
    .metric-bar { display: flex; gap: 16px; margin-bottom: 20px; }
    .metric-tile { flex: 1; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; text-align: left; }
    .metric-value { color: #f0f6fc; font-size: 28px; font-weight: 600; line-height: 1.2; }
    .metric-label { color: #8b949e; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
    
    /* Real-Time Terminal Viewport */
    .terminal-window { background: #010409; border: 1px solid #30363d; border-radius: 8px; height: 220px; overflow-y: auto; padding: 16px; box-shadow: inset 0 0 10px rgba(0,0,0,0.5); }
    .terminal-stream { color: #3fb950; font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 12px; line-height: 1.6; white-space: pre-wrap; margin: 0; }
    
    /* Form Element Trims */
    .stButton>button { background-color: #238636; color: #ffffff; border: 1px solid rgba(240,246,242,0.1); border-radius: 6px; padding: 6px 16px; font-weight: 500; font-size: 14px; width: 100%; transition: background 0.2s; }
    .stButton>button:hover { background-color: #2ea043; color: #ffffff; border: 1px solid rgba(240,246,242,0.1); }
    div.stTextArea textarea { background-color: #010409; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; font-size: 14px; }
    div.stTextArea textarea:focus { border-color: #58a6ff; }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LAYOUT HEADER TERMINAL
# ============================================================================
st.markdown("<h1>COGNITIVE DATA NEXUS</h1>", unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multi-Dimensional GraphRAG Local Intelligence & Observability Console</div>', unsafe_allow_html=True)

# Split view allocation matrix
left_col, right_col = st.columns([1, 1.3])

# ============================================================================
# LEFT VIEWPORT: INGESTION PIPELINE & RETRIEVAL WORKBENCH
# ============================================================================
with left_col:
    # Card 1: Document Processing Engine
    st.markdown('<div class="enterprise-card"><div class="card-header">Document Ingestion Subsystem</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload target operational node asset (.txt, .md)", type=["txt", "md"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        if st.button("Execute Structural Pipeline Pipeline"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/plain")}
            with st.spinner("Executing non-blocking text extraction and vector chunk mapping..."):
                try:
                    response = requests.post(f"{API_BASE_URL}/ingest", files=files)
                    if response.status_code == 200:
                        res_data = response.json()
                        emit_ui_log("ingest", f"Successfully processed asset '{uploaded_file.name}'. Generated {res_data['chunks_processed']} partitions.")
                        st.success(f"Ingestion verified. Document ID: {res_data['document_id']}")
                        
                        # Cache the metrics changes directly into current state wrapper
                        st.session_state.graph_data = requests.get(f"{API_BASE_URL}/graph").json()
                    else:
                        st.error(f"Ingestion System Warning: {response.text}")
                except Exception as e:
                    st.error(f"API Gateway Intercept Error: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Card 2: Hybrid Query Workbench
    st.markdown('<div class="enterprise-card"><div class="card-header">Hybrid Fusion Query Matrix</div>', unsafe_allow_html=True)
    user_query = st.text_area("Input cross-document reasoning lookup prompt:", height=120, label_visibility="collapsed", placeholder="Enter cross-space analytical query...")
    
    if st.button("Synthesize Grounded Response"):
        if user_query.strip():
            payload = {"prompt": user_query, "limit": 2}
            with st.spinner("Calculating concurrent RRF matrix paths and routing local Llama3 weights..."):
                try:
                    response = requests.post(f"{API_BASE_URL}/query", json=payload)
                    if response.status_code == 200:
                        res_data = response.json()
                        emit_ui_log("query", f"Synthesized inference for prompt: '{user_query[:30]}...'")
                        
                        st.markdown("<h4 style='font-size:14px; color:#8b949e; margin-bottom:8px;'>Grounded Local Generation:</h4>", unsafe_allow_html=True)
                        st.info(res_data["answer"])
                        
                        with st.expander("Show Fused Context Evidence Layers"):
                            st.markdown("**Retrieved Vector Context (Geometric Space):**")
                            st.code(res_data["vector_context"], language="json")
                            st.markdown("**Retrieved Knowledge Network Components (Topological Space):**")
                            st.code(res_data["graph_context"], language="json")
                    else:
                        st.error(f"Inference Subsystem Warning: {response.text}")
                except Exception as e:
                    st.error(f"Inference Route Missing: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# RIGHT VIEWPORT: TOPOLOGICAL RUNTIME HEALTH & FORCE GRAPH VISUALIZATION
# ============================================================================
with right_col:
    st.markdown('<div class="enterprise-card"><div class="card-header">Topological Knowledge Network Topology</div>', unsafe_allow_html=True)
    
    if st.button("Synchronize Network Schema State"):
        with st.spinner("Extracting in-memory NetworkX relationship records..."):
            try:
                response = requests.get(f"{API_BASE_URL}/graph")
                if response.status_code == 200:
                    st.session_state.graph_data = response.json()
                    emit_ui_log("graph", f"Synchronized relational state. Cache memory updated cleanly.")
                else:
                    st.error("Could not sync data nodes from graph store.")
            except Exception as e:
                st.error(f"Graph Database Connectivity Warning: {str(e)}")

    # Extract dynamic length attributes from saved state records
    current_nodes = st.session_state.graph_data.get("nodes", [])
    current_edges = st.session_state.graph_data.get("edges", [])
    
    # Datadog Style Telemetry Header
    st.markdown(f"""
        <div class="metric-bar">
            <div class="metric-tile">
                <div class="metric-value">{len(current_nodes)}</div>
                <div class="metric-label">Active Nodes</div>
            </div>
            <div class="metric-tile">
                <div class="metric-value">{len(current_edges)}</div>
                <div class="metric-label">Relation Edges</div>
            </div>
            <div class="metric-tile">
                <div class="metric-value" style="color:#3fb950;">Stable</div>
                <div class="metric-label">Runtime Integrity</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # If active items have been indexed into our graph, mount the dynamic force-directed web
    if current_nodes:
        nodes_list = []
        edges_list = []
        
        for node in current_nodes:
            nodes_list.append(Node(
                id=str(node["id"]), 
                label=str(node["id"]), 
                size=18, 
                color="#58a6ff"
            ))
            
        for edge in current_edges:
            edges_list.append(Edge(
                source=str(edge["source"]), 
                target=str(edge["target"]), 
                label=str(edge.get("type", "CONNECTS"))
            ))
            
        network_config = Config(
            width="100%",
            height=400,
            directed=True,
            physics=True,
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#58a6ff",
            collapsible=False
        )
        
        # Render the raw interactive canvas widget on screen
        agraph(nodes=nodes_list, edges=edges_list, config=network_config)
    else:
        st.markdown(
            "<div style='background-color:#010409; border:1px dashed #30363d; border-radius:6px; padding:40px; text-align:center; color:#8b949e; font-size:13px;'>"
            "Awaiting pipeline initialization topology. Process a structural document or run a schema sync step above to load network node visuals."
            "</div>", 
            unsafe_allow_html=True
        )
        
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# PERSISTENT FOOTER MATRIX: TELEMETRY SYSTEM LOG WINDOW
# ============================================================================
st.markdown("---")
compiled_logs = "\n".join(st.session_state.system_logs)
st.markdown(f"""
    <div class="enterprise-card" style="margin-bottom:0px;">
        <div class="card-header" style="margin-bottom:12px;">System Operational Telemetry Stream</div>
        <div class="terminal-window">
            <pre class="terminal-stream">{compiled_logs}</pre>
        </div>
    </div>
""", unsafe_allow_html=True)