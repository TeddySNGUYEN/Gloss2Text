"""
Graph-to-Embedding Encoding Module
Implements Section 3.4: relation-aware graph encoding
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from fst_parser import Role, Relation


class RelationEncoder(nn.Module):
    """Encode edge relation types as learnable embeddings"""
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        
        # Relation type embeddings
        self.relation_embeddings = nn.Embedding(
            num_embeddings=len(Relation),
            embedding_dim=embedding_dim
        )
        
        # Create mapping from Relation enum to indices
        self.relation_to_idx = {
            Relation.REFERENCE_REL: 0,
            Relation.AGREEMENT_REL: 1,
            Relation.MODIFICATION: 2,
            Relation.REL_AGREEMENT: 3
        }
    
    def forward(self, relations: List[Relation]) -> torch.Tensor:
        """
        Encode relations as embeddings
        
        Args:
            relations: List of Relation enums
            
        Returns:
            Tensor of shape [num_relations, embedding_dim]
        """
        indices = torch.tensor([self.relation_to_idx[r] for r in relations],
                              dtype=torch.long)
        return self.relation_embeddings(indices)


class RoleEncoder(nn.Module):
    """Encode component roles as learnable embeddings"""
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        
        self.role_embeddings = nn.Embedding(
            num_embeddings=len(Role),
            embedding_dim=embedding_dim
        )
        
        self.role_to_idx = {
            Role.REFERENCE: 0,
            Role.AGREEMENT: 1,
            Role.NEGATION: 2,
            Role.LOCATIVE: 3,
            Role.ASPECT: 4,
            Role.LEXICALROOT: 5
        }
    
    def forward(self, roles: List[Role]) -> torch.Tensor:
        """
        Encode roles as embeddings
        
        Args:
            roles: List of Role enums
            
        Returns:
            Tensor of shape [num_roles, embedding_dim]
        """
        indices = torch.tensor([self.role_to_idx[r] for r in roles],
                              dtype=torch.long)
        return self.role_embeddings(indices)


class GraphEncoder(nn.Module):
    """
    Graph-to-Embedding Encoder
    Implements Equations 3-4 from Section 3.4
    """
    
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int = 512,
                 hidden_dim: int = 512,
                 num_layers: int = 1,
                 dropout: float = 0.1):
        """
        Initialize graph encoder
        
        Args:
            vocab_size: Size of gloss vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Hidden dimension for transformations
            num_layers: Number of graph convolution layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Component text embeddings
        self.text_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Role and relation encoders
        self.role_encoder = RoleEncoder(embedding_dim)
        self.relation_encoder = RelationEncoder(embedding_dim)
        
        # Relation-specific transformations (ψ_r in Eq. 3)
        self.relation_transforms = nn.ModuleDict({
            'reference': nn.Linear(embedding_dim, embedding_dim),
            'agreement': nn.Linear(embedding_dim, embedding_dim),
            'modification': nn.Linear(embedding_dim, embedding_dim),
            'rel_agreement': nn.Linear(embedding_dim, embedding_dim)
        })
        
        # Node update function (φ in Eq. 3)
        self.update_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Attention-based pooling for graph-level representation (Eq. 4)
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                node_texts: torch.Tensor,
                node_roles: List[Role],
                edge_index: torch.Tensor,
                edge_relations: List[Relation],
                batch_vector: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode gloss-internal graph to fixed-dimensional embedding
        
        Args:
            node_texts: Tensor of shape [num_nodes] with text token IDs
            node_roles: List of Role enums for each node
            edge_index: Tensor of shape [2, num_edges] with edge connections
            edge_relations: List of Relation enums for each edge
            batch_vector: Optional tensor indicating which graph each node belongs to
            
        Returns:
            Graph embeddings of shape [num_graphs, embedding_dim]
        """
        num_nodes = node_texts.size(0)
        
        # Initial node embeddings: h^(0)_v (combine text + role)
        text_emb = self.text_embedding(node_texts)  # [num_nodes, d]
        role_emb = self.role_encoder(node_roles)     # [num_nodes, d]
        h = text_emb + role_emb                      # [num_nodes, d]
        h = self.dropout(h)
        
        # Apply graph convolution layers
        for layer in range(self.num_layers):
            h = self._graph_conv_layer(h, edge_index, edge_relations)
        
        # Pool to graph-level representation (Eq. 4)
        if batch_vector is None:
            # Single graph
            graph_emb = self._attention_pool(h)  # [1, d]
        else:
            # Multiple graphs
            graph_emb = self._batch_attention_pool(h, batch_vector)  # [batch_size, d]
        
        return graph_emb
    
    def _graph_conv_layer(self,
                         h: torch.Tensor,
                         edge_index: torch.Tensor,
                         edge_relations: List[Relation]) -> torch.Tensor:
        """
        Single graph convolution layer (Eq. 3)
        
        Implements: h^(1)_v = φ(h^(0)_v, Σ_{(u,v,r)} ψ_r(h^(0)_u))
        """
        num_nodes = h.size(0)
        num_edges = edge_index.size(1)
        
        if num_edges == 0:
            # No edges - just return original
            return h
        
        # Aggregate messages from neighbors
        aggregated = torch.zeros_like(h)  # [num_nodes, d]
        
        # For each edge, apply relation-specific transformation
        for i in range(num_edges):
            source_idx = edge_index[0, i].item()
            target_idx = edge_index[1, i].item()
            relation = edge_relations[i]
            
            # Get source node embedding
            h_u = h[source_idx]  # [d]
            
            # Apply relation-specific transformation ψ_r
            relation_key = relation.value.replace('_', '')
            if relation_key not in self.relation_transforms:
                relation_key = 'modification'  # default
            
            transformed = self.relation_transforms[relation_key](h_u)  # [d]
            
            # Aggregate to target node
            aggregated[target_idx] += transformed
        
        # Update node representations: φ(h_v, aggregated)
        combined = torch.cat([h, aggregated], dim=-1)  # [num_nodes, 2*d]
        h_new = self.update_mlp(combined)  # [num_nodes, d]
        
        # Residual connection
        h_new = h_new + h
        
        return h_new
    
    def _attention_pool(self, h: torch.Tensor) -> torch.Tensor:
        """
        Attention-based pooling for single graph (Eq. 4)
        
        z = Σ_v α_v * h_v where α_v = softmax(attention_scores)
        """
        # Compute attention scores
        attention_scores = self.attention(h)  # [num_nodes, 1]
        attention_weights = F.softmax(attention_scores, dim=0)  # [num_nodes, 1]
        
        # Weighted sum
        graph_emb = (h * attention_weights).sum(dim=0, keepdim=True)  # [1, d]
        
        return graph_emb
    
    def _batch_attention_pool(self, 
                             h: torch.Tensor,
                             batch_vector: torch.Tensor) -> torch.Tensor:
        """
        Attention-based pooling for batched graphs
        
        Args:
            h: Node embeddings [total_nodes, d]
            batch_vector: Tensor indicating graph assignment [total_nodes]
            
        Returns:
            Graph embeddings [batch_size, d]
        """
        batch_size = batch_vector.max().item() + 1
        graph_embs = []
        
        for i in range(batch_size):
            # Get nodes belonging to graph i
            mask = (batch_vector == i)
            h_i = h[mask]  # [num_nodes_i, d]
            
            # Pool this graph
            graph_emb_i = self._attention_pool(h_i)  # [1, d]
            graph_embs.append(graph_emb_i)
        
        graph_embs = torch.cat(graph_embs, dim=0)  # [batch_size, d]
        
        return graph_embs


class GlossSequenceEncoder(nn.Module):
    """
    Encode a sequence of glosses using graph-internal representations
    This creates the gloss embedding sequence Z (Eq. 5)
    """
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 512,
                 hidden_dim: int = 512,
                 num_graph_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        
        self.graph_encoder = GraphEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_graph_layers,
            dropout=dropout
        )
        
        # Positional encoding for sequence
        self.pos_encoding = PositionalEncoding(embedding_dim, dropout)
    
    def forward(self, graphs_data: List[Dict]) -> torch.Tensor:
        """
        Encode sequence of gloss graphs
        
        Args:
            graphs_data: List of dicts, each containing:
                - node_texts: [num_nodes] token IDs
                - node_roles: List[Role]
                - edge_index: [2, num_edges]
                - edge_relations: List[Relation]
        
        Returns:
            Sequence embeddings [seq_len, batch_size, d]
        """
        gloss_embeddings = []
        
        for graph_data in graphs_data:
            # Encode each gloss graph
            z_i = self.graph_encoder(
                node_texts=graph_data['node_texts'],
                node_roles=graph_data['node_roles'],
                edge_index=graph_data['edge_index'],
                edge_relations=graph_data['edge_relations']
            )  # [1, d]
            
            gloss_embeddings.append(z_i)
        
        # Stack into sequence
        Z = torch.cat(gloss_embeddings, dim=0)  # [seq_len, d]
        Z = Z.unsqueeze(1)  # [seq_len, 1, d] for batch_size=1
        
        # Add positional encoding
        Z = self.pos_encoding(Z)
        
        return Z


class PositionalEncoding(nn.Module):
    """Positional encoding for sequences"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                            (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# Example usage
if __name__ == "__main__":
    from graph_construction import GlossGraph
    
    # Create example vocabulary
    vocab = {'IX': 0, '2P': 1, 'WAKE': 2, 'UP': 3, 'BATH': 4, 
             'NOT': 5, 'FINISH': 6, 'PAD': 7}
    vocab_size = len(vocab)
    
    # Create graph encoder
    encoder = GraphEncoder(vocab_size=vocab_size, embedding_dim=128)
    
    # Test with single graph
    gloss = "IX-2P-WAKE-UP"
    graph = GlossGraph(gloss)
    
    # Prepare inputs
    node_texts = torch.tensor([vocab[c.text] for c in graph.components])
    node_roles = [c.role for c in graph.components]
    edge_index, edge_relations = graph.get_edge_index()
    
    # Encode
    print(f"Encoding graph: {gloss}")
    print(f"Nodes: {node_texts.tolist()}")
    print(f"Roles: {[r.value for r in node_roles]}")
    
    with torch.no_grad():
        graph_emb = encoder(node_texts, node_roles, edge_index, edge_relations)
    
    print(f"Graph embedding shape: {graph_emb.shape}")
    print(f"Graph embedding (first 10 dims): {graph_emb[0, :10].tolist()}")
