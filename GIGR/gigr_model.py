"""
Gloss-Internal Graph Representation (GIGR) Model
Main framework integrating graph encoding with Transformer
"""
import torch
import torch.nn as nn
from typing import List, Dict, Optional
import math

from graph_construction import GlossGraph, GlossGraphBatch
from graph_encoding import GraphEncoder, PositionalEncoding
from fst_parser import GlossFST


class GIGRTransformer(nn.Module):
    """
    Complete GIGR framework for Gloss-to-Text translation
    Implements the full pipeline from Figure 1
    """
    
    def __init__(self,
                 vocab_size: int,
                 target_vocab_size: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 4,
                 num_decoder_layers: int = 4,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.2,
                 num_graph_layers: int = 1):
        """
        Initialize GIGR model
        
        Args:
            vocab_size: Size of source gloss vocabulary
            target_vocab_size: Size of target language vocabulary
            d_model: Dimension of model (default 512 as per Table 2)
            nhead: Number of attention heads (default 8 as per Table 2)
            num_encoder_layers: Number of Transformer encoder layers (2/4/6)
            num_decoder_layers: Number of Transformer decoder layers (2/4/6)
            dim_feedforward: FFN hidden dimension (default 2048)
            dropout: Dropout rate (default 0.2 as per Table 2)
            num_graph_layers: Number of graph convolution layers
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Gloss-Internal Graph Encoder
        self.graph_encoder = GraphEncoder(
            vocab_size=vocab_size,
            embedding_dim=d_model,
            hidden_dim=d_model,
            num_layers=num_graph_layers,
            dropout=dropout
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # [seq_len, batch, d_model]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Target embedding
        self.target_embedding = nn.Embedding(target_vocab_size, d_model)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, target_vocab_size)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self,
                graphs_data: List[Dict],
                target: torch.Tensor,
                target_mask: Optional[torch.Tensor] = None,
                target_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            graphs_data: List of graph dicts for each gloss in sequence
            target: Target sequence [tgt_len, batch_size]
            target_mask: Target attention mask
            target_padding_mask: Target padding mask
            
        Returns:
            Output logits [tgt_len, batch_size, target_vocab_size]
        """
        # Step 1: Encode gloss sequence using graph representations
        # Z = (z1, z2, ..., zn) from Eq. 5
        gloss_embeddings = self._encode_gloss_sequence(graphs_data)
        # Shape: [seq_len, batch_size, d_model]
        
        # Step 2: Apply Transformer Encoder
        # H = Enc(Z)
        memory = self.transformer_encoder(gloss_embeddings)
        # Shape: [seq_len, batch_size, d_model]
        
        # Step 3: Prepare target embeddings
        target_emb = self.target_embedding(target) * math.sqrt(self.d_model)
        target_emb = self.pos_encoder(target_emb)
        
        # Step 4: Apply Transformer Decoder
        # p(y | Z) = Dec(H)
        if target_mask is None:
            target_mask = self._generate_square_subsequent_mask(target.size(0))
            target_mask = target_mask.to(target.device)
        
        decoder_output = self.transformer_decoder(
            target_emb,
            memory,
            tgt_mask=target_mask,
            tgt_key_padding_mask=target_padding_mask
        )
        # Shape: [tgt_len, batch_size, d_model]
        
        # Step 5: Project to vocabulary
        output = self.output_projection(decoder_output)
        # Shape: [tgt_len, batch_size, target_vocab_size]
        
        return output
    
    def _encode_gloss_sequence(self, graphs_data: List[Dict]) -> torch.Tensor:
        """
        Encode sequence of gloss graphs (creates Z from Eq. 5)
        
        Args:
            graphs_data: List of graph data dicts
            
        Returns:
            Gloss embedding sequence [seq_len, batch_size, d_model]
        """
        gloss_embeddings = []
        
        for graph_data in graphs_data:
            # Encode each gloss graph to single vector z_i
            z_i = self.graph_encoder(
                node_texts=graph_data['node_texts'],
                node_roles=graph_data['node_roles'],
                edge_index=graph_data['edge_index'],
                edge_relations=graph_data['edge_relations']
            )  # [1, d_model]
            
            gloss_embeddings.append(z_i)
        
        # Stack into sequence
        Z = torch.cat(gloss_embeddings, dim=0)  # [seq_len, d_model]
        Z = Z.unsqueeze(1)  # [seq_len, 1, d_model] for batch_size=1
        
        # Add positional encoding
        Z = self.pos_encoder(Z)
        
        return Z
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask
    
    def encode(self, graphs_data: List[Dict]) -> torch.Tensor:
        """
        Encode gloss sequence (for inference)
        
        Args:
            graphs_data: List of graph data dicts
            
        Returns:
            Encoder memory [seq_len, batch_size, d_model]
        """
        gloss_embeddings = self._encode_gloss_sequence(graphs_data)
        memory = self.transformer_encoder(gloss_embeddings)
        return memory
    
    def decode(self,
               target: torch.Tensor,
               memory: torch.Tensor,
               target_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode step (for inference)
        
        Args:
            target: Target sequence [tgt_len, batch_size]
            memory: Encoder output [src_len, batch_size, d_model]
            target_mask: Target mask
            
        Returns:
            Output logits [tgt_len, batch_size, target_vocab_size]
        """
        target_emb = self.target_embedding(target) * math.sqrt(self.d_model)
        target_emb = self.pos_encoder(target_emb)
        
        if target_mask is None:
            target_mask = self._generate_square_subsequent_mask(target.size(0))
            target_mask = target_mask.to(target.device)
        
        decoder_output = self.transformer_decoder(
            target_emb,
            memory,
            tgt_mask=target_mask
        )
        
        output = self.output_projection(decoder_output)
        return output
    
    @torch.no_grad()
    def generate(self,
                 graphs_data: List[Dict],
                 max_len: int = 100,
                 bos_idx: int = 1,
                 eos_idx: int = 2) -> torch.Tensor:
        """
        Greedy decoding for inference
        
        Args:
            graphs_data: List of graph data dicts for source glosses
            max_len: Maximum generation length
            bos_idx: Beginning-of-sentence token index
            eos_idx: End-of-sentence token index
            
        Returns:
            Generated sequence [seq_len]
        """
        self.eval()
        
        # Encode source
        memory = self.encode(graphs_data)
        
        # Initialize with BOS
        generated = torch.tensor([[bos_idx]], dtype=torch.long)
        
        for _ in range(max_len):
            # Decode
            output = self.decode(generated, memory)
            
            # Get next token
            next_token = output[-1, 0, :].argmax(dim=-1, keepdim=True)
            
            # Append
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=0)
            
            # Check for EOS
            if next_token.item() == eos_idx:
                break
        
        return generated.squeeze(1)  # [seq_len]


class GlossGraphPreprocessor:
    """
    Preprocess raw gloss sequences into graph data for model input
    """
    
    def __init__(self, vocab: Dict[str, int]):
        """
        Args:
            vocab: Mapping from gloss component text to token ID
        """
        self.vocab = vocab
        self.parser = GlossFST()
    
    def process_gloss_sequence(self, gloss_sequence: List[str]) -> List[Dict]:
        """
        Convert gloss sequence to list of graph data dicts
        
        Args:
            gloss_sequence: List of gloss strings
            
        Returns:
            List of dicts with graph data for each gloss
        """
        graphs_data = []
        
        for gloss in gloss_sequence:
            # Construct graph
            graph = GlossGraph(gloss)
            
            # Convert component texts to token IDs
            node_texts = []
            for comp in graph.components:
                token_id = self.vocab.get(comp.text, self.vocab.get('<UNK>', 0))
                node_texts.append(token_id)
            
            node_texts = torch.tensor(node_texts, dtype=torch.long)
            node_roles = [comp.role for comp in graph.components]
            edge_index, edge_relations = graph.get_edge_index()
            
            graphs_data.append({
                'node_texts': node_texts,
                'node_roles': node_roles,
                'edge_index': edge_index,
                'edge_relations': edge_relations
            })
        
        return graphs_data


# Example usage
if __name__ == "__main__":
    # Define vocabularies
    source_vocab = {
        'IX': 0, '2P': 1, '3P': 2, '1P': 3,
        'WAKE': 4, 'UP': 5, 'BATH': 6, 'NEED': 7, 'YOU': 8,
        'NOT': 9, 'FINISH': 10,
        '<PAD>': 11, '<UNK>': 12
    }
    
    target_vocab = {
        '<PAD>': 0, '<BOS>': 1, '<EOS>': 2,
        'you': 3, 'need': 4, 'to': 5, 'take': 6, 'a': 7, 'bath': 8,
        'did': 9, 'wake': 10, 'up': 11
    }
    
    # Create model
    model = GIGRTransformer(
        vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        d_model=128,  # Small for demo
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        num_graph_layers=1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Prepare input
    preprocessor = GlossGraphPreprocessor(source_vocab)
    gloss_sequence = ["IX-2P-WAKE-UP", "BATH", "NEED", "YOU"]
    graphs_data = preprocessor.process_gloss_sequence(gloss_sequence)
    
    print(f"\nInput: {gloss_sequence}")
    print(f"Graphs created: {len(graphs_data)}")
    
    # Example forward pass
    target = torch.tensor([
        [1],  # <BOS>
        [3],  # you
        [4],  # need
        [5],  # to
    ], dtype=torch.long)
    
    with torch.no_grad():
        output = model(graphs_data, target)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected: [target_len={target.size(0)}, batch_size=1, vocab_size={len(target_vocab)}]")
    
    # Test generation
    print("\n" + "="*50)
    print("Testing greedy generation...")
    generated = model.generate(
        graphs_data,
        max_len=20,
        bos_idx=target_vocab['<BOS>'],
        eos_idx=target_vocab['<EOS>']
    )
    
    # Convert to words
    idx_to_word = {v: k for k, v in target_vocab.items()}
    generated_words = [idx_to_word[idx.item()] for idx in generated]
    print(f"Generated: {' '.join(generated_words)}")
