import os
import json
import httpx
import networkx as nx
from typing import List, Dict, Any
from app.models.entity_model import EntityNode, EntityRelationship

class KnowledgeGraphService:
    """
    Localized Graph Orchestration Service.
    Leverages NetworkX to manage directed structural relationship matrices
    completely in-memory with local JSON persistence blocks and an integrated local SLM parsing engine.
    """
    def __init__(self, storage_path: str = "./graph_db/knowledge_graph.json", ollama_url: str = "http://127.0.0.1:11434"):
        self.storage_path = os.path.abspath(storage_path)
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        self.graph = nx.DiGraph()
        self.load_graph()
        
        # Local inference connection configuration
        self.client = httpx.Client(base_url=ollama_url, timeout=60.0)
        self.model_name = "phi3"

    def extract_graph_via_slm(self, document_id: str, document_name: str, raw_text: str) -> Dict[str, int]:
        """
        Analyzes unstructured text using a local SLM, extracts entities and relationships,
        and automatically updates the NetworkX state matrices.
        """
        prompt = f"""
        Analyze the following technical text fragment and extract core system components, frameworks, services, or data stores as Entities, alongside their explicit directional relationships.
        Return your answer strictly as a valid JSON object with 'nodes' and 'relationships' keys. Do not include any conversational markdown block wrappers, explanations, or leading/trailing text.

        Required JSON Schema Specification:
        {{
          "nodes": [
            {{"name": "FastAPI", "type": "Framework", "properties": {{"description": "Core backend gateway server"}}}}
          ],
          "relationships": [
            {{"source": "FastAPI", "target": "ChromaDB", "type": "DEPENDS_ON", "properties": {{"purpose": "Vector persistence search"}}}}
          ]
        }}

        Text to analyze:
        "{raw_text[:4000]}"
        """

        try:
            # 1. Dispatch prompt context payload to local Ollama worker
            response = self.client.post("/api/generate", json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0  # Eradicate creative variance
                }
            })
            
            raw_output = response.json().get("response", "").strip()
            
            # 2. Intercept and scrub markdown code fence artifacts
            if raw_output.startswith("```json"):
                raw_output = raw_output.split("```json")[1].split("```")[0].strip()
            elif raw_output.startswith("```"):
                raw_output = raw_output.split("```")[1].split("```")[0].strip()
                
            extracted_data = json.loads(raw_output)
            
            initial_nodes = self.graph.number_of_nodes()
            initial_edges = self.graph.number_of_edges()

            # 3. Process and seed extracted structural nodes
            for node_data in extracted_data.get("nodes", []):
                properties = node_data.get("properties", {})
                properties.update({"document_id": document_id, "source_file": document_name})
                node_obj = EntityNode(
                    name=node_data.get("name", "").strip(),
                    entity_type=node_data.get("type", "UNKNOWN").upper(),
                    properties=properties
                )
                self.add_entity(node_obj)

            # 4. Map and bridge relational graph edges
            for rel_data in extracted_data.get("relationships", []):
                properties = rel_data.get("properties", {})
                properties.update({"document_id": document_id})
                rel_obj = EntityRelationship(
                    source=rel_data.get("source", "").strip(),
                    target=rel_data.get("target", "").strip(),
                    relation_type=rel_data.get("type", "CONNECTED_TO").upper(),
                    properties=properties
                )
                self.add_relationship(rel_obj)

            # 5. Flush state changes to local storage disk
            self.save_graph()
            
            return {
                "nodes_added": self.graph.number_of_nodes() - initial_nodes,
                "edges_added": self.graph.number_of_edges() - initial_edges
            }

        except Exception as e:
            print(f"[GRAPH LAYER FAULT] Automated SLM extraction subloop failed: {str(e)}")
            return {"nodes_added": 0, "edges_added": 0}

    def add_entity(self, node: EntityNode):
        """Adds or updates an entity node within the NetworkX layer."""
        if not node.name:
            return
        if self.graph.has_node(node.name):
            existing_props = self.graph.nodes[node.name].get("properties", {})
            existing_props.update(node.properties)
            self.graph.nodes[node.name]["properties"] = existing_props
        else:
            self.graph.add_node(node.name, type=node.entity_type, properties=node.properties)

    def add_relationship(self, rel: EntityRelationship):
        """Injects a directed edge between two validated entity nodes."""
        if not rel.source or not rel.target:
            return
        if not self.graph.has_node(rel.source):
            self.add_entity(EntityNode(name=rel.source, entity_type="UNKNOWN", properties={}))
        if not self.graph.has_node(rel.target):
            self.add_entity(EntityNode(name=rel.target, entity_type="UNKNOWN", properties={}))

        if self.graph.has_edge(rel.source, rel.target):
            existing_props = self.graph.edges[rel.source, rel.target].get("properties", {})
            existing_props.update(rel.properties)
            self.graph.edges[rel.source, rel.target]["properties"] = existing_props
        else:
            self.graph.add_edge(rel.source, rel.target, type=rel.relation_type, properties=rel.properties)

    def save_graph(self):
        """Serializes the NetworkX graph matrix into a clean, human-readable JSON state file."""
        graph_data = {
            "nodes": [
                {"id": node, "type": data.get("type"), "properties": data.get("properties", {})}
                for node, data in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, "type": data.get("type"), "properties": data.get("properties", {})}
                for u, v, data in self.graph.edges(data=True)
            ]
        }
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=4, ensure_ascii=False)

    def load_graph(self):
        """Hydrates the NetworkX runtime memory layer from the local JSON storage file."""
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for node_data in data.get("nodes", []):
                self.graph.add_node(node_data["id"], type=node_data.get("type"), properties=node_data.get("properties", {}))
            for edge_data in data.get("edges", []):
                self.graph.add_edge(edge_data["source"], edge_data["target"], type=edge_data.get("type"), properties=edge_data.get("properties", {}))
        except Exception:
            self.graph = nx.DiGraph()