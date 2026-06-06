import streamlit as st
import requests
import json

st.set_page_config(
    page_title="Cognitive Data Nexus Console",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Cyberpunk-themed custom CSS styles
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #c9d1d9; }
    h1, h2, h3 { color: #58a6ff !important; font-family: 'Courier New', Courier, monospace; }
    .stButton>button { background-color: #238636; color: white; border-radius: 6px; border: none; }
    .stButton>button:hover { background-color: #2ea043; }
    div.stTextArea textarea { background-color: #161b22; color: #c9d1d9; border: 1px solid #30363d; }
    </style>
""", unsafe_allow_html=True)

API_BASE_URL = "http://127.0.0.1:8000/api"

st.title("🪐 Cognitive Data Nexus")
st.subheader("Enterprise-Grade Multi-Dimensional GraphRAG Control Console")
st.write("---")

# Layout division splits
left_col, right_col = st.columns([1, 1.2])

with left_col:
    st.header("📥 Document Ingestion Pipeline")
    uploaded_file = st.file_uploader("Upload architecture logs or insights (.txt, .md)", type=["txt", "md"])
    
    if uploaded_file is not None:
        if st.button("Execute Core Data Ingestion"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/plain")}
            with st.spinner("Processing structural parsing layers..."):
                try:
                    response = requests.post(f"{API_BASE_URL}/ingest", files=files)
                    if response.status_code == 200:
                        res_data = response.json()
                        st.success(f"Successfully processed {uploaded_file.name}!")
                        st.metric("Context Chunks Generated", res_data["chunks_processed"])
                        st.json(res_data["graph_metrics"])
                    else:
                        st.error(f"Ingestion Server Fault: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to API Gateway: {str(e)}")

    st.write("---")
    st.header("🔎 Hybrid Fusion Query Matrix")
    user_query = st.text_area("Input complex multi-document reasoning prompt:")
    
    if st.button("Synthesize Query Response"):
        if user_query.strip():
            payload = {"prompt": user_query, "limit": 2}
            with st.spinner("Executing concurrent RRF vector lookup and topological path traversals..."):
                try:
                    response = requests.post(f"{API_BASE_URL}/query", json=payload)
                    if response.status_code == 200:
                        res_data = response.json()
                        st.subheader("🎯 Grounded Local Generation Answer:")
                        st.info(res_data["answer"])
                        
                        with st.expander("Show Fused Context Evidence Layers"):
                            st.write("**Retrieved Vector Contexts (Geometric):**")
                            st.write(res_data["vector_context"])
                            st.write("**Retrieved Graph Structural Connections (Topological):**")
                            st.write(res_data["graph_context"])
                    else:
                        st.error(f"Inference Routing Fault: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to Inference Gateway: {str(e)}")

with right_col:
    st.header("🌐 Topological Knowledge Graph State")
    if st.button("Refresh Network State"):
        try:
            response = requests.get(f"{API_BASE_URL}/graph")
            if response.status_code == 200:
                graph_data = response.json()
                st.metric("Total Extracted Entity Nodes", len(graph_data["nodes"]))
                st.metric("Total Directed Relational Edges", len(graph_data["edges"]))
                
                st.write("**Active In-Memory Relationship Schema Matrix:**")
                st.dataframe(graph_data["edges"], use_container_width=True)
            else:
                st.error("Could not fetch active graph structural nodes.")
        except Exception as e:
            st.error(f"Graph Database Sync Offline: {str(e)}")