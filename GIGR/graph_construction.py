"""
Gloss-Internal Graph Construction
Implements Algorithm 1 from the paper
"""
import torch
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import networkx as nx
from fst_parser import Component, Role, Relation, GlossFST


@dataclass
class Edge:
    """Directed labeled edge in gloss-internal graph"""
    source: int
    target: int
    relation: Relation
    

class GlossGraph:
    """
    Directed labeled graph representation of a compound gloss
    Implements the graph structure from Section 3.3
    """
    
    def __init__(self, gloss: str):
        """
        Construct gloss-internal graph from compound gloss
        
        Args:
            gloss: Raw gloss string (e.g., "IX-2P-WAKE-UP")
        """
        self.gloss = gloss
        self.parser = GlossFST()
        
        # Parse gloss
        self.components, self.root_idx = self.parser.parse(gloss)
        self.n_nodes = len(self.components)
        
        # Build graph
        self.edges = []
        self._construct_graph()
        
    def _construct_graph(self):
        """
        Execute Algorithm 1: Gloss-internal graph construction
        """
        n = self.n_nodes
        
        # Handle single-component case
        if n == 1:
            return
        
        # Root-attached edges: foreach i in {1,...,n} \ {k}
        for i, comp in enumerate(self.components):
            if i != self.root_idx:
                edge = Edge(
                    source=i,
                    target=self.root_idx,
                    relation=comp.relation
                )
                self.edges.append(edge)
        
        # Agreement refinement edge
        self._add_refinement_edges()
    
    def _add_refinement_edges(self):
        """
        Add refinement edge between AGREEMENT and REFERENCE
        if both exist (last step of Algorithm 1)
        """
        ref_idx = None
        agr_idx = None
        
        for i, comp in enumerate(self.components):
            if comp.role == Role.REFERENCE:
                ref_idx = i
            elif comp.role == Role.AGREEMENT:
                agr_idx = i
        
        # Add refinement edge: agreement -> reference
        if ref_idx is not None and agr_idx is not None:
            edge = Edge(
                source=agr_idx,
                target=ref_idx,
                relation=Relation.REL_AGREEMENT
            )
            self.edges.append(edge)
    
    def get_adjacency_list(self) -> Dict[int, List[Tuple[int, Relation]]]:
        """
        Get adjacency list representation
        
        Returns:
            Dict mapping node_id -> [(neighbor_id, relation), ...]
        """
        adj_list = {i: [] for i in range(self.n_nodes)}
        
        for edge in self.edges:
            adj_list[edge.target].append((edge.source, edge.relation))
        
        return adj_list
    
    def get_edge_index(self) -> Tuple[torch.Tensor, List[Relation]]:
        """
        Get edge index in PyTorch Geometric format
        
        Returns:
            edge_index: [2, num_edges] tensor
            edge_types: List of relations for each edge
        """
        if not self.edges:
            return torch.empty((2, 0), dtype=torch.long), []
        
        sources = [e.source for e in self.edges]
        targets = [e.target for e in self.edges]
        edge_types = [e.relation for e in self.edges]
        
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        
        return edge_index, edge_types
    
    def to_networkx(self) -> nx.DiGraph:
        """
        Convert to NetworkX directed graph for visualization
        
        Returns:
            NetworkX DiGraph
        """
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for i, comp in enumerate(self.components):
            G.add_node(i, 
                      text=comp.text,
                      role=comp.role.value,
                      is_root=(i == self.root_idx))
        
        # Add edges with attributes
        for edge in self.edges:
            G.add_edge(edge.source, edge.target,
                      relation=edge.relation.value)
        
        return G
    
    def visualize(self, save_path: Optional[str] = None):
        """
        Visualize the graph using matplotlib
        
        Args:
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            
            G = self.to_networkx()
            
            # Create layout
            pos = nx.spring_layout(G, seed=42)
            
            # Draw nodes
            node_colors = ['red' if G.nodes[n]['is_root'] else 'lightblue' 
                          for n in G.nodes()]
            
            plt.figure(figsize=(12, 8))
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                  node_size=2000, alpha=0.9)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, 
                                  arrows=True, arrowsize=20,
                                  connectionstyle='arc3,rad=0.1')
            
            # Draw labels
            node_labels = {n: f"{G.nodes[n]['text']}\n({G.nodes[n]['role']})" 
                          for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, node_labels, font_size=10)
            
            # Draw edge labels
            edge_labels = {(u, v): G[u][v]['relation'] 
                          for u, v in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
            
            plt.title(f"Gloss-Internal Graph: {self.gloss}", fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
        except ImportError:
            print("matplotlib not available for visualization")
    
    def __repr__(self):
        return f"GlossGraph('{self.gloss}', nodes={self.n_nodes}, edges={len(self.edges)})"


class GlossGraphBatch:
    """Batch multiple gloss graphs for efficient processing"""
    
    def __init__(self, glosses: List[str]):
        """
        Create batch of gloss graphs
        
        Args:
            glosses: List of gloss strings
        """
        self.graphs = [GlossGraph(g) for g in glosses]
        self.batch_size = len(glosses)
    
    def get_batched_data(self) -> Dict[str, torch.Tensor]:
        """
        Get batched graph data for neural network processing
        
        Returns:
            Dict with batched tensors
        """
        all_edge_indices = []
        all_edge_types = []
        node_counts = []
        
        offset = 0
        for graph in self.graphs:
            edge_index, edge_types = graph.get_edge_index()
            
            # Offset node indices for batching
            if edge_index.numel() > 0:
                edge_index = edge_index + offset
                all_edge_indices.append(edge_index)
                all_edge_types.extend(edge_types)
            
            node_counts.append(graph.n_nodes)
            offset += graph.n_nodes
        
        # Concatenate all edges
        if all_edge_indices:
            batched_edge_index = torch.cat(all_edge_indices, dim=1)
        else:
            batched_edge_index = torch.empty((2, 0), dtype=torch.long)
        
        return {
            'edge_index': batched_edge_index,
            'edge_types': all_edge_types,
            'node_counts': node_counts,
            'batch_size': self.batch_size,
            'total_nodes': sum(node_counts)
        }


# Example usage
if __name__ == "__main__":
    # Test single graph
    gloss = "IX-2P-WAKE-UP"
    graph = GlossGraph(gloss)
    
    print(f"\nGraph for: {gloss}")
    print(f"Nodes: {graph.n_nodes}")
    print(f"Root: {graph.components[graph.root_idx].text}")
    print(f"Edges: {len(graph.edges)}")
    
    for edge in graph.edges:
        src = graph.components[edge.source]
        tgt = graph.components[edge.target]
        print(f"  {src.text} --[{edge.relation.value}]--> {tgt.text}")
    
    # Test batch
    print("\n" + "="*50)
    glosses = ["IX-2P-WAKE-UP", "BATH", "NOT-FINISH", "IX-1P-LOVE-3P"]
    batch = GlossGraphBatch(glosses)
    
    print(f"\nBatch of {batch.batch_size} glosses:")
    batch_data = batch.get_batched_data()
    print(f"Total nodes: {batch_data['total_nodes']}")
    print(f"Total edges: {batch_data['edge_index'].shape[1]}")
    
    # Visualize one graph
    print("\nGenerating visualization...")
    graph.visualize()
