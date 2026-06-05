from typing import Dict, Any, Optional
from datetime import datetime

class EntityNode:
    """Represents a unique extracted entity node within the Knowledge Graph."""
    def __init__(self, name: str, entity_type: str, properties: Optional[Dict[str, Any]] = None):
        self.name = name.strip()
        self.entity_type = entity_type.upper()
        self.properties = properties or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.entity_type,
            "properties": self.properties
        }


class EntityRelationship:
    """Represents a directed relationship edge between two entity nodes."""
    def __init__(
        self, 
        source: str, 
        target: str, 
        relation_type: str, 
        properties: Optional[Dict[str, Any]] = None
    ):
        self.source = source.strip()
        self.target = target.strip()
        self.relation_type = relation_type.upper()
        self.properties = properties or {}
        self.properties["created_at"] = self.properties.get("created_at", datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.relation_type,
            "properties": self.properties
        }