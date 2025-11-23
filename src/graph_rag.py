"""
Graph RAG Module

Implements knowledge graph-based retrieval for enhanced reasoning.
Works in dry-run mode without API keys.
"""

import os
from typing import List, Dict, Set, Tuple, Optional, Any
import warnings


class GraphRAG:
    """
    Knowledge graph-based RAG for structured reasoning.
    
    Features:
    - Entity extraction from documents
    - Relationship mapping
    - Graph-based traversal and retrieval
    - Multi-hop reasoning
    
    Supports dry-run mode with placeholder graph.
    """
    
    def __init__(self):
        """Initialize the graph RAG system."""
        self.has_api_key = bool(os.getenv("OPENAI_API_KEY"))
        self.graph: Dict[str, Dict[str, List[str]]] = {}
        self.entities: Set[str] = set()
        
        if not self.has_api_key:
            warnings.warn(
                "OPENAI_API_KEY not found. Running in dry-run mode with sample graph. "
                "Set OPENAI_API_KEY for entity extraction and full graph capabilities."
            )
            self._initialize_sample_graph()
    
    def _initialize_sample_graph(self) -> None:
        """Initialize a sample GDPR knowledge graph for dry-run mode."""
        # Sample entities and relationships
        self.graph = {
            "GDPR": {
                "regulates": ["personal_data", "data_processing"],
                "protects": ["data_subjects"],
                "requires": ["consent", "transparency"]
            },
            "data_subject": {
                "has_right_to": ["access", "erasure", "rectification", "portability"],
                "can_file": ["complaint"],
                "is_protected_by": ["GDPR"]
            },
            "controller": {
                "must_implement": ["security_measures", "privacy_by_design"],
                "must_appoint": ["DPO"],
                "is_responsible_for": ["data_processing"]
            },
            "personal_data": {
                "is_regulated_by": ["GDPR"],
                "includes": ["name", "email", "IP_address", "biometric_data"],
                "requires": ["lawful_basis"]
            }
        }
        
        self.entities = set(self.graph.keys())
        for relations in self.graph.values():
            for targets in relations.values():
                self.entities.update(targets)
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract entities from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of extracted entities
            
        TODO: Implement NER with spaCy or LLM-based extraction
        """
        if not self.has_api_key:
            # Dry-run: keyword matching
            found_entities = []
            text_lower = text.lower()
            
            for entity in self.entities:
                if entity.lower().replace("_", " ") in text_lower:
                    found_entities.append(entity)
            
            return found_entities
        
        # TODO: Production implementation
        # Use spaCy NER or OpenAI function calling for entity extraction
        return []
    
    def extract_relationships(self, entity1: str, entity2: str, context: str) -> List[str]:
        """
        Extract relationships between two entities.
        
        Args:
            entity1: First entity
            entity2: Second entity
            context: Context text
            
        Returns:
            List of relationships
            
        TODO: Implement relationship extraction with LLM
        """
        if not self.has_api_key:
            # Dry-run: check if entities are connected in sample graph
            if entity1 in self.graph:
                for relation, targets in self.graph[entity1].items():
                    if entity2 in targets:
                        return [relation]
            return []
        
        # TODO: Production implementation
        return []
    
    def add_to_graph(self, entity: str, relation: str, target: str) -> None:
        """
        Add a triple (entity, relation, target) to the graph.
        
        Args:
            entity: Source entity
            relation: Relationship type
            target: Target entity
        """
        if entity not in self.graph:
            self.graph[entity] = {}
        
        if relation not in self.graph[entity]:
            self.graph[entity][relation] = []
        
        if target not in self.graph[entity][relation]:
            self.graph[entity][relation].append(target)
        
        self.entities.add(entity)
        self.entities.add(target)
    
    def build_graph_from_documents(self, documents: List[Dict[str, str]]) -> None:
        """
        Build knowledge graph from documents.
        
        Args:
            documents: List of documents with content and metadata
            
        TODO: Implement full graph construction pipeline
        """
        if not self.has_api_key:
            print(f"Dry-run: Would build graph from {len(documents)} documents")
            print("Using sample graph instead")
            return
        
        # TODO: Production implementation
        # for doc in documents:
        #     entities = self.extract_entities(doc["content"])
        #     for i, e1 in enumerate(entities):
        #         for e2 in entities[i+1:]:
        #             relations = self.extract_relationships(e1, e2, doc["content"])
        #             for rel in relations:
        #                 self.add_to_graph(e1, rel, e2)
    
    def get_neighbors(self, entity: str, relation: Optional[str] = None) -> List[str]:
        """
        Get neighboring entities in the graph.
        
        Args:
            entity: Entity to get neighbors for
            relation: Optional relation filter
            
        Returns:
            List of neighboring entities
        """
        if entity not in self.graph:
            return []
        
        if relation:
            return self.graph[entity].get(relation, [])
        
        # Return all neighbors
        neighbors = []
        for targets in self.graph[entity].values():
            neighbors.extend(targets)
        
        return list(set(neighbors))
    
    def find_path(self, start: str, end: str, max_depth: int = 3) -> List[List[str]]:
        """
        Find paths between two entities in the graph.
        
        Args:
            start: Start entity
            end: End entity
            max_depth: Maximum path length
            
        Returns:
            List of paths (each path is a list of entities)
            
        TODO: Implement proper graph traversal (BFS/DFS)
        """
        if start not in self.graph:
            return []
        
        # Simple BFS implementation
        paths = []
        queue = [(start, [start])]
        visited = set()
        
        while queue and len(paths) < 10:  # Limit to 10 paths
            current, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            if current == end:
                paths.append(path)
                continue
            
            if current in visited:
                continue
            
            visited.add(current)
            
            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in path:  # Avoid cycles
                    queue.append((neighbor, path + [neighbor]))
        
        return paths
    
    def graph_based_retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant information using graph traversal.
        
        Args:
            query: User query
            k: Number of results to return
            
        Returns:
            List of relevant graph paths/subgraphs
        """
        # Extract entities from query
        query_entities = self.extract_entities(query)
        
        if not query_entities:
            return []
        
        results = []
        
        # For each query entity, get neighborhood
        for entity in query_entities[:3]:  # Limit to first 3 entities
            neighbors = self.get_neighbors(entity)
            
            for neighbor in neighbors[:k]:
                # Get the connecting relation
                relations = []
                if entity in self.graph:
                    for rel, targets in self.graph[entity].items():
                        if neighbor in targets:
                            relations.append(rel)
                
                results.append({
                    "entity": entity,
                    "relation": relations[0] if relations else "related_to",
                    "target": neighbor,
                    "relevance": 0.9  # Placeholder score
                })
        
        return results[:k]
    
    def multi_hop_reasoning(self, query: str, hops: int = 2) -> Dict[str, Any]:
        """
        Perform multi-hop reasoning over the knowledge graph.
        
        Args:
            query: User query
            hops: Number of hops to traverse
            
        Returns:
            Reasoning result with explanation
        """
        query_entities = self.extract_entities(query)
        
        if len(query_entities) < 2:
            return {
                "success": False,
                "message": "Need at least 2 entities for multi-hop reasoning"
            }
        
        # Find paths between entities
        paths = self.find_path(query_entities[0], query_entities[1], max_depth=hops)
        
        return {
            "success": len(paths) > 0,
            "query_entities": query_entities,
            "paths": paths,
            "explanation": f"Found {len(paths)} paths connecting entities"
        }
    
    def query(self, query: str, use_multi_hop: bool = False) -> str:
        """
        Answer query using graph-based retrieval.
        
        Args:
            query: User query
            use_multi_hop: Whether to use multi-hop reasoning
            
        Returns:
            Response
        """
        if use_multi_hop:
            result = self.multi_hop_reasoning(query)
            
            if result["success"]:
                paths_str = ", ".join([" -> ".join(p) for p in result["paths"][:3]])
                return (
                    f"[Graph RAG] Found connections: {paths_str}. "
                    f"This shows relationships between {result['query_entities']}."
                )
            else:
                return f"[Graph RAG] {result['message']}"
        else:
            results = self.graph_based_retrieve(query)
            
            if results:
                facts = [f"{r['entity']} {r['relation']} {r['target']}" for r in results[:3]]
                facts_str = "; ".join(facts)
                return f"[Graph RAG] Relevant facts: {facts_str}"
            else:
                return "[Graph RAG] No relevant information found in knowledge graph"
    
    def visualize_subgraph(self, entity: str, depth: int = 2) -> Dict[str, Any]:
        """
        Get subgraph around an entity for visualization.
        
        Args:
            entity: Center entity
            depth: Depth of subgraph
            
        Returns:
            Subgraph data structure
            
        TODO: Return format compatible with graph visualization libraries
        """
        nodes = {entity}
        edges = []
        
        def explore(current_entity: str, current_depth: int):
            if current_depth <= 0 or current_entity not in self.graph:
                return
            
            for relation, targets in self.graph[current_entity].items():
                for target in targets:
                    nodes.add(target)
                    edges.append({
                        "source": current_entity,
                        "target": target,
                        "relation": relation
                    })
                    
                    if current_depth > 1:
                        explore(target, current_depth - 1)
        
        explore(entity, depth)
        
        return {
            "nodes": list(nodes),
            "edges": edges,
            "center": entity
        }


def main():
    """Example usage of GraphRAG."""
    graph_rag = GraphRAG()
    
    # Works without API keys (uses sample graph)
    response = graph_rag.query("What rights does a data subject have under GDPR?")
    print(f"Response: {response}")
    
    # Multi-hop reasoning
    response2 = graph_rag.query("How is GDPR related to data subjects?", use_multi_hop=True)
    print(f"\nMulti-hop: {response2}")
    
    # Visualize subgraph
    subgraph = graph_rag.visualize_subgraph("data_subject", depth=2)
    print(f"\nSubgraph around 'data_subject': {len(subgraph['nodes'])} nodes, {len(subgraph['edges'])} edges")


if __name__ == "__main__":
    main()
