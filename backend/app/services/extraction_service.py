import re
from typing import List, Tuple
from app.models.entity_model import EntityNode, EntityRelationship

class RuleBasedGraphExtractor:
    """
    Deterministic Semantic Link Extraction Engine.
    Parses linguistic and structural document contexts to extract entity graphs 
    without requiring external API execution dependencies.
    """
    def __init__(self):
        # Compiled patterns tracking architecture definitions and active actions
        self.action_verbs = [r"uses", r"utilizes", r"stores", r"handles", r"maps", r"manages", r"executes"]
        self.relationship_pattern = re.compile(
            rf"\s*(.+?)\s+({'|'.join(self.action_verbs)})\s+(.+?)(?:\.|\n|$)", 
            re.IGNORECASE
        )

    def extract_from_text(self, text: str) -> Tuple[List[EntityNode], List[EntityRelationship]]:
        """Scans a text sequence to construct entity structures and relation edges."""
        nodes = []
        relationships = []
        
        if not text:
            return nodes, relationships

        # Clean line iteration processing
        lines = [line.strip() for line in text.split(".") if line.strip()]
        
        for line in lines:
            match = self.relationship_pattern.search(line)
            if match:
                source_name = match.group(1).strip()
                action_type = match.group(2).strip().upper()
                target_name = match.group(3).strip()
                
                # Sanity guard: clean up minor layout junk from sentence structures
                source_name = re.sub(r'^(the|a|an)\s+', '', source_name, flags=re.IGNORECASE)
                target_name = re.sub(r'^(the|a|an)\s+', '', target_name, flags=re.IGNORECASE)
                
                # Deduplicate and build normalized domain structures
                src_node = EntityNode(name=source_name, entity_type="CONCEPT")
                tgt_node = EntityNode(name=target_name, entity_type="TECHNOLOGY")
                
                nodes.extend([src_node, tgt_node])
                relationships.append(
                    EntityRelationship(
                        source=source_name,
                        target=target_name,
                        relation_type=action_type,
                        properties={"context_raw": line}
                    )
                )
                
        return nodes, relationships