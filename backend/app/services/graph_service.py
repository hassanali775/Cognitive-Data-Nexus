import os
import json
import networkx as nx
from typing import List, Dict, Any
from app.models.entity_model import EntityNode, EntityRelationship

class KnowledgeGraphService:
    """
    Localized Graph Orchestration Service.
    Leverages NetworkX to manage directed structural relationship matrices
    completely in-memory with local JSON persistence blocks.
    """
    def __init__(self, storage_path: str = "./graph_db/knowledge_graph.json"):
        self.storage_path = os.path.abspath(storage_path)
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        self.graph = nx.DiGraph()
        self.load_graph()

    def add_entity(self, node: EntityNode):
        """Adds or updates an entity node within the NetworkX layer."""
        if not node.name:
            return
            
        # If node exists, merge properties instead of overwriting completely
        if self.graph.has_node(node.name):
            existing_props = self.graph.nodes[node.name].get("properties", {})
            existing_props.update(node.properties)
            self.graph.nodes[node.name]["properties"] = existing_props
        else:
            self.graph.add_node(
                node.name, 
                type=node.entity_type, 
                properties=node.properties
            )

    def add_relationship(self, rel: EntityRelationship):
        """Injects a directed edge between two validated entity nodes."""
        if not rel.source or not rel.target:
            return
            
        # Ensure both nodes exist in the graph first
        if not self.graph.has_node(rel.source):
            self.add_entity(EntityNode(name=rel.source, entity_type="UNKNOWN"))
        if not self.graph.has_node(rel.target):
            self.add_entity(EntityNode(name=rel.target, entity_type="UNKNOWN"))

        # Add or update directed edge connection
        if self.graph.has_edge(rel.source, rel.target):
            existing_props = self.graph.edges[rel.source, rel.target].get("properties", {})
            existing_props.update(rel.properties)
            self.graph.edges[rel.source, rel.target]["properties"] = existing_props
        else:
            self.graph.add_edge(
                rel.source, 
                rel.target, 
                type=rel.relation_type, 
                properties=rel.properties
            )

    def save_graph(self):
        """Serializes the NetworkX graph matrix into a clean, human-readable JSON state file."""
        graph_data = {
            "nodes": [
                {"id": node, "type": data.get("type"), "properties": data.get("properties", {})}
                for node, data in self.graph.nodes(data=True)
            ],
            "edges": [
                {
                    "source": u, 
                    "target": v, 
                    "type": data.get("type"), 
                    "properties": data.get("properties", {})
                }
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
                self.graph.add_node(
                    node_data["id"], 
                    type=node_data.get("type"), 
                    properties=node_data.get("properties", {})
                )
                
            for edge_data in data.get("edges", []):
                self.graph.add_edge(
                    edge_data["source"],
                    edge_data["target"],
                    type=edge_data.get("type"),
                    properties=edge_data.get("properties", {})
                )
        except Exception:
            # Fall back to empty graph if file is corrupted
            self.graph = nx.DiGraph()