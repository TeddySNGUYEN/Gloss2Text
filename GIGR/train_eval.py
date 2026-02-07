"""
Training and Evaluation Utilities for GIGR
Implements training loop, evaluation metrics, and dataset handling
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import time
from collections import defaultdict
import numpy as np

from gigr_model import GIGRTransformer, GlossGraphPreprocessor
from graph_construction import GlossGraph


class GlossToTextDataset(Dataset):
    """Dataset for Gloss-to-Text translation"""
    
    def __init__(self,
                 gloss_sequences: List[List[str]],
                 target_sequences: List[List[str]],
                 source_vocab: Dict[str, int],
                 target_vocab: Dict[str, int]):
        """
        Args:
            gloss_sequences: List of gloss sequences (already tokenized)
            target_sequences: List of target word sequences
            source_vocab: Source vocabulary mapping
            target_vocab: Target vocabulary mapping
        """
        self.gloss_sequences = gloss_sequences
        self.target_sequences = target_sequences
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        
        self.preprocessor = GlossGraphPreprocessor(source_vocab)
        
        # Special tokens
        self.bos_idx = target_vocab['<BOS>']
        self.eos_idx = target_vocab['<EOS>']
        self.pad_idx = target_vocab['<PAD>']
    
    def __len__(self):
        return len(self.gloss_sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get single example
        
        Returns:
            Dict with:
                - graphs_data: List of graph dicts
                - target: Target token IDs with BOS/EOS
        """
        gloss_seq = self.gloss_sequences[idx]
        target_seq = self.target_sequences[idx]
        
        # Process gloss sequence to graphs
        graphs_data = self.preprocessor.process_gloss_sequence(gloss_seq)
        
        # Convert target to IDs with BOS/EOS
        target_ids = [self.bos_idx]
        for word in target_seq:
            target_ids.append(
                self.target_vocab.get(word, self.target_vocab.get('<UNK>', 0))
            )
        target_ids.append(self.eos_idx)
        
        return {
            'graphs_data': graphs_data,
            'target': torch.tensor(target_ids, dtype=torch.long)
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for DataLoader
    Currently supports batch_size=1 for simplicity
    """
    # For now, just return first item (batch_size=1)
    return batch[0]


class GIGRTrainer:
    """Trainer for GIGR model"""
    
    def __init__(self,
                 model: GIGRTransformer,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 pad_idx: int = 0):
        """
        Initialize trainer
        
        Args:
            model: GIGR model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer (e.g., AdamW)
            criterion: Loss function (e.g., CrossEntropyLoss)
            device: Device to use
            pad_idx: Padding token index
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.pad_idx = pad_idx
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self) -> float:
        """
        Train for one epoch
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in self.train_loader:
            graphs_data = batch['graphs_data']
            target = batch['target'].to(self.device)  # [seq_len]
            
            # Prepare input/output for teacher forcing
            # Input: [BOS, w1, w2, ..., wn]
            # Target: [w1, w2, ..., wn, EOS]
            input_target = target[:-1].unsqueeze(1)  # [seq_len-1, 1]
            output_target = target[1:]  # [seq_len-1]
            
            # Move graph data to device
            for graph_data in graphs_data:
                graph_data['node_texts'] = graph_data['node_texts'].to(self.device)
                graph_data['edge_index'] = graph_data['edge_index'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(graphs_data, input_target)  # [seq_len-1, 1, vocab_size]
            
            # Compute loss
            output = output.squeeze(1)  # [seq_len-1, vocab_size]
            loss = self.criterion(output, output_target)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> float:
        """
        Validate model
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in self.val_loader:
            graphs_data = batch['graphs_data']
            target = batch['target'].to(self.device)
            
            input_target = target[:-1].unsqueeze(1)
            output_target = target[1:]
            
            # Move to device
            for graph_data in graphs_data:
                graph_data['node_texts'] = graph_data['node_texts'].to(self.device)
                graph_data['edge_index'] = graph_data['edge_index'].to(self.device)
            
            # Forward
            output = self.model(graphs_data, input_target)
            output = output.squeeze(1)
            loss = self.criterion(output, output_target)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def train(self, num_epochs: int):
        """
        Train for multiple epochs
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"Training on device: {self.device}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("="*60)
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Time: {epoch_time:.2f}s")
        
        print("="*60)
        print("Training complete!")


class BLEUEvaluator:
    """BLEU score evaluator (simplified implementation)"""
    
    @staticmethod
    def compute_bleu(references: List[List[str]], 
                     hypotheses: List[List[str]],
                     max_n: int = 4) -> Dict[str, float]:
        """
        Compute BLEU scores
        
        Args:
            references: List of reference sequences (each is list of words)
            hypotheses: List of hypothesis sequences
            max_n: Maximum n-gram order (default 4 for BLEU-4)
            
        Returns:
            Dict with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
        """
        from collections import Counter
        
        scores = {}
        
        for n in range(1, max_n + 1):
            precision_sum = 0
            count_sum = 0
            
            for ref, hyp in zip(references, hypotheses):
                # Get n-grams
                ref_ngrams = Counter(
                    tuple(ref[i:i+n]) for i in range(len(ref) - n + 1)
                )
                hyp_ngrams = Counter(
                    tuple(hyp[i:i+n]) for i in range(len(hyp) - n + 1)
                )
                
                # Compute clipped counts
                clipped_counts = sum(
                    min(hyp_ngrams[ng], ref_ngrams[ng]) 
                    for ng in hyp_ngrams
                )
                
                precision_sum += clipped_counts
                count_sum += max(len(hyp) - n + 1, 0)
            
            # Compute precision
            precision = precision_sum / count_sum if count_sum > 0 else 0
            scores[f'BLEU-{n}'] = precision * 100  # Convert to percentage
        
        return scores


class LatencyProfiler:
    """
    Profile end-to-end inference latency
    Implements Equation 8 from the paper
    """
    
    def __init__(self, model: GIGRTransformer, device: str):
        self.model = model
        self.device = device
        self.timings = defaultdict(list)
    
    @torch.no_grad()
    def profile(self,
                gloss_sequence: List[str],
                preprocessor: GlossGraphPreprocessor,
                num_runs: int = 100) -> Dict[str, float]:
        """
        Profile latency for single sequence
        
        Args:
            gloss_sequence: Input gloss sequence
            preprocessor: Preprocessor for graphs
            num_runs: Number of runs for averaging
            
        Returns:
            Dict with timing breakdown
        """
        self.model.eval()
        
        timings = {
            'preprocessing': [],
            'graph_construction': [],
            'encoding': [],
            'decoding': [],
            'total': []
        }
        
        for _ in range(num_runs):
            # Preprocessing
            start = time.time()
            graphs_data = preprocessor.process_gloss_sequence(gloss_sequence)
            timings['preprocessing'].append((time.time() - start) * 1000)  # ms
            
            # Move to device
            for graph_data in graphs_data:
                graph_data['node_texts'] = graph_data['node_texts'].to(self.device)
                graph_data['edge_index'] = graph_data['edge_index'].to(self.device)
            
            # Encoding
            start = time.time()
            memory = self.model.encode(graphs_data)
            torch.cuda.synchronize() if self.device == 'cuda' else None
            timings['encoding'].append((time.time() - start) * 1000)
            
            # Decoding (one step)
            target = torch.tensor([[1]], dtype=torch.long).to(self.device)  # BOS
            start = time.time()
            _ = self.model.decode(target, memory)
            torch.cuda.synchronize() if self.device == 'cuda' else None
            timings['decoding'].append((time.time() - start) * 1000)
        
        # Compute averages
        results = {}
        for key in timings:
            if timings[key]:
                results[f'{key}_mean'] = np.mean(timings[key])
                results[f'{key}_std'] = np.std(timings[key])
        
        # Total latency
        results['total_mean'] = (
            results['preprocessing_mean'] +
            results['encoding_mean'] +
            results['decoding_mean']
        )
        
        return results


# Example usage
if __name__ == "__main__":
    # Create dummy data
    source_vocab = {
        'IX': 0, '2P': 1, '3P': 2, '1P': 3,
        'WAKE': 4, 'UP': 5, 'BATH': 6, 'NEED': 7, 'YOU': 8,
        '<PAD>': 9, '<UNK>': 10
    }
    
    target_vocab = {
        '<PAD>': 0, '<BOS>': 1, '<EOS>': 2,
        'you': 3, 'need': 4, 'to': 5, 'take': 6, 'a': 7, 'bath': 8,
        'did': 9, 'wake': 10, 'up': 11, '<UNK>': 12
    }
    
    # Dummy dataset
    train_glosses = [
        ['IX-2P-WAKE-UP'],
        ['BATH', 'NEED', 'YOU']
    ]
    train_targets = [
        ['did', 'you', 'wake', 'up'],
        ['you', 'need', 'to', 'take', 'a', 'bath']
    ]
    
    # Create dataset
    train_dataset = GlossToTextDataset(
        train_glosses, train_targets,
        source_vocab, target_vocab
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Create model
    model = GIGRTransformer(
        vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2
    )
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=target_vocab['<PAD>'])
    
    trainer = GIGRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=train_loader,  # Using same for demo
        optimizer=optimizer,
        criterion=criterion,
        device='cpu',
        pad_idx=target_vocab['<PAD>']
    )
    
    # Train
    print("Starting training demo...")
    trainer.train(num_epochs=3)
    
    # Evaluate BLEU
    print("\n" + "="*60)
    print("Computing BLEU scores...")
    
    references = [['did', 'you', 'wake', 'up']]
    hypotheses = [['you', 'wake', 'up']]  # Dummy prediction
    
    evaluator = BLEUEvaluator()
    bleu_scores = evaluator.compute_bleu(references, hypotheses)
    
    for metric, score in bleu_scores.items():
        print(f"{metric}: {score:.2f}")
    
    # Profile latency
    print("\n" + "="*60)
    print("Profiling latency...")
    
    preprocessor = GlossGraphPreprocessor(source_vocab)
    profiler = LatencyProfiler(model, device='cpu')
    
    latency_results = profiler.profile(
        gloss_sequence=['IX-2P-WAKE-UP'],
        preprocessor=preprocessor,
        num_runs=10
    )
    
    print(f"End-to-end latency: {latency_results['total_mean']:.2f} ± "
          f"{latency_results.get('encoding_std', 0):.2f} ms")
