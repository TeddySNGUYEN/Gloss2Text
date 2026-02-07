# GIGR Implementation - Complete Usage Guide

## Overview

This document provides a comprehensive guide to using the GIGR (Gloss-Internal Graph Construction and Encoding) implementation for Sign Language Translation.

## Installation

1. **Install dependencies:**
```bash
cd gigr
pip install -r requirements.txt
```

2. **Verify installation:**
```bash
python test_gigr.py
```

## Components Overview

### 1. FST Parser (`fst_parser.py`)

The FST parser implements deterministic morphological analysis of compound glosses.

**Usage:**
```python
from gigr import GlossFST

parser = GlossFST()
components, root_idx = parser.parse("IX-2P-WAKE-UP")

# Access parsed components
for comp in components:
    print(f"{comp.text}: {comp.role.value}")
    
# Output:
# IX: REFERENCE
# 2P: AGREEMENT
# WAKE: LEXICALROOT
# UP: ASPECT
```

**Supported Patterns:**
- Reference: `IX-*`, `INDEX-*`
- Agreement: `1P`, `2P`, `3P`
- Negation: `NOT`, `NO`
- Locative: `LOC`, `LOCATION`
- Aspect: Custom markers (UP, DOWN, CONT, FINISH, etc.)

### 2. Graph Construction (`graph_construction.py`)

Builds directed labeled graphs from parsed gloss components.

**Basic Usage:**
```python
from gigr import GlossGraph

# Create graph for a compound gloss
graph = GlossGraph("IX-2P-WAKE-UP")

# Access graph properties
print(f"Nodes: {graph.n_nodes}")
print(f"Root: {graph.components[graph.root_idx].text}")

# Iterate over edges
for edge in graph.edges:
    src = graph.components[edge.source]
    tgt = graph.components[edge.target]
    print(f"{src.text} --[{edge.relation.value}]--> {tgt.text}")
```

**Visualization:**
```python
# Save graph visualization
graph.visualize(save_path="graph.png")

# Or display interactively
graph.visualize()
```

**Batch Processing:**
```python
from gigr import GlossGraphBatch

glosses = ["IX-2P-WAKE-UP", "BATH", "NOT-FINISH"]
batch = GlossGraphBatch(glosses)

# Get batched data for neural networks
batch_data = batch.get_batched_data()
print(f"Total nodes: {batch_data['total_nodes']}")
print(f"Total edges: {batch_data['edge_index'].shape[1]}")
```

### 3. Graph Encoding (`graph_encoding.py`)

Encodes graphs to fixed-dimensional embeddings using relation-aware aggregation.

**Basic Usage:**
```python
from gigr import GraphEncoder
import torch

# Create encoder
encoder = GraphEncoder(
    vocab_size=1000,
    embedding_dim=512,
    hidden_dim=512,
    num_layers=1,
    dropout=0.1
)

# Prepare graph data
graph = GlossGraph("IX-2P-WAKE-UP")
node_texts = torch.tensor([vocab[c.text] for c in graph.components])
node_roles = [c.role for c in graph.components]
edge_index, edge_relations = graph.get_edge_index()

# Encode to embedding
graph_embedding = encoder(
    node_texts=node_texts,
    node_roles=node_roles,
    edge_index=edge_index,
    edge_relations=edge_relations
)
# Output shape: [1, 512]
```

**Multi-layer Encoding:**
```python
# Use multiple graph convolution layers
encoder = GraphEncoder(
    vocab_size=1000,
    embedding_dim=512,
    num_layers=2  # Stack multiple GNN layers
)
```

### 4. GIGR Model (`gigr_model.py`)

Complete Transformer-based translation model with graph encoding.

**Model Creation:**
```python
from gigr import GIGRTransformer

model = GIGRTransformer(
    vocab_size=1000,           # Source gloss vocabulary size
    target_vocab_size=5000,    # Target language vocabulary size
    d_model=512,               # Model dimension
    nhead=8,                   # Number of attention heads
    num_encoder_layers=4,      # Transformer encoder layers
    num_decoder_layers=4,      # Transformer decoder layers
    dim_feedforward=2048,      # FFN hidden dimension
    dropout=0.2,               # Dropout rate
    num_graph_layers=1         # Graph convolution layers
)
```

**Preprocessing:**
```python
from gigr import GlossGraphPreprocessor

# Create preprocessor with vocabulary
vocab = {'IX': 0, '2P': 1, 'WAKE': 2, 'UP': 3, ...}
preprocessor = GlossGraphPreprocessor(vocab)

# Process gloss sequence
gloss_sequence = ["IX-2P-WAKE-UP", "BATH", "NEED"]
graphs_data = preprocessor.process_gloss_sequence(gloss_sequence)
```

**Training:**
```python
# Forward pass (teacher forcing)
target = torch.tensor([[1, 3, 4, 5, 2]])  # [BOS, words..., EOS]
input_target = target[:, :-1].transpose(0, 1)
output_target = target[:, 1:]

output_logits = model(
    graphs_data=graphs_data,
    target=input_target
)

# Compute loss
loss = criterion(
    output_logits.view(-1, target_vocab_size),
    output_target.view(-1)
)
```

**Inference:**
```python
# Generate translation
generated = model.generate(
    graphs_data=graphs_data,
    max_len=50,
    bos_idx=1,  # <BOS> token
    eos_idx=2   # <EOS> token
)

# Convert to words
words = [idx_to_word[idx.item()] for idx in generated]
```

### 5. Training and Evaluation (`train_eval.py`)

**Dataset Creation:**
```python
from gigr import GlossToTextDataset
from torch.utils.data import DataLoader

# Prepare data
gloss_sequences = [
    ['IX-2P-WAKE-UP'],
    ['BATH', 'NEED', 'YOU']
]
target_sequences = [
    ['did', 'you', 'wake', 'up'],
    ['you', 'need', 'to', 'take', 'a', 'bath']
]

# Create dataset
dataset = GlossToTextDataset(
    gloss_sequences=gloss_sequences,
    target_sequences=target_sequences,
    source_vocab=source_vocab,
    target_vocab=target_vocab
)

# Create dataloader
loader = DataLoader(dataset, batch_size=1, shuffle=True)
```

**Training Loop:**
```python
from gigr import GIGRTrainer
import torch.nn as nn

# Setup optimizer and loss
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# Create trainer
trainer = GIGRTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    device='cuda'
)

# Train
trainer.train(num_epochs=10)
```

**BLEU Evaluation:**
```python
from gigr import BLEUEvaluator

evaluator = BLEUEvaluator()

references = [['did', 'you', 'wake', 'up']]
hypotheses = [['you', 'wake', 'up']]

scores = evaluator.compute_bleu(references, hypotheses, max_n=4)
print(f"BLEU-4: {scores['BLEU-4']:.2f}")
```

**Latency Profiling:**
```python
from gigr import LatencyProfiler

profiler = LatencyProfiler(model, device='cuda')

results = profiler.profile(
    gloss_sequence=['IX-2P-WAKE-UP'],
    preprocessor=preprocessor,
    num_runs=100
)

print(f"Average latency: {results['total_mean']:.2f} ms")
```

## Complete Example Pipeline

See `example_full.py` for a complete end-to-end example that demonstrates:

1. Data preparation
2. Vocabulary building
3. Model creation
4. Training
5. Evaluation with BLEU
6. Latency profiling
7. Graph visualization

**Run it:**
```bash
python example_full.py
```

## Hyperparameter Recommendations

Based on Table 2 from the paper:

**Small Model (for testing):**
```python
model = GIGRTransformer(
    d_model=256,
    nhead=8,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=1024,
    dropout=0.2
)
```

**Medium Model (paper baseline):**
```python
model = GIGRTransformer(
    d_model=512,
    nhead=8,
    num_encoder_layers=4,
    num_decoder_layers=4,
    dim_feedforward=2048,
    dropout=0.2
)
```

**Large Model (best performance):**
```python
model = GIGRTransformer(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.2
)
```

**Training Configuration:**
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,
    betas=(0.9, 0.999),
    eps=1e-8
)

# Train for 10 epochs with batch size 32
```

## Common Patterns

### Pattern 1: Simple Translation

```python
# 1. Create model
model = GIGRTransformer(vocab_size, target_vocab_size)

# 2. Load pretrained weights
checkpoint = torch.load('model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# 3. Prepare input
preprocessor = GlossGraphPreprocessor(vocab)
graphs_data = preprocessor.process_gloss_sequence(glosses)

# 4. Generate
output = model.generate(graphs_data, max_len=50)
```

### Pattern 2: Fine-tuning

```python
# 1. Load pretrained model
model = GIGRTransformer(...)
model.load_state_dict(torch.load('pretrained.pt'))

# 2. Freeze graph encoder (optional)
for param in model.graph_encoder.parameters():
    param.requires_grad = False

# 3. Fine-tune on new data
trainer = GIGRTrainer(model, train_loader, val_loader, optimizer, criterion)
trainer.train(num_epochs=5)
```

### Pattern 3: Inference Only

```python
# Efficient inference setup
model.eval()
torch.set_grad_enabled(False)

# Process multiple glosses
for gloss_seq in test_data:
    graphs_data = preprocessor.process_gloss_sequence(gloss_seq)
    output = model.generate(graphs_data)
    print(decode_output(output))
```

## Advanced Usage

### Custom FST Rules

Extend the FST with custom patterns:

```python
from gigr import GlossFST

parser = GlossFST()

# Add custom aspect markers
parser.aspect_markers.update(['NEW_ASPECT_1', 'NEW_ASPECT_2'])

# Add custom patterns
parser.custom_patterns = [r'^MY-PATTERN$']
```

### Custom Graph Encoding

Modify graph encoding behavior:

```python
from gigr import GraphEncoder

class CustomGraphEncoder(GraphEncoder):
    def _graph_conv_layer(self, h, edge_index, edge_relations):
        # Custom graph convolution logic
        return super()._graph_conv_layer(h, edge_index, edge_relations)
```

### Multi-task Learning

Extend for multiple objectives:

```python
class MultiTaskGIGR(GIGRTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Additional task head
        self.pos_tagger = nn.Linear(self.d_model, num_pos_tags)
    
    def forward(self, graphs_data, target, return_pos=False):
        # Main translation
        output = super().forward(graphs_data, target)
        
        if return_pos:
            # POS tagging
            memory = self.encode(graphs_data)
            pos_logits = self.pos_tagger(memory)
            return output, pos_logits
        
        return output
```

## Troubleshooting

### Issue: Out of Memory

**Solution:** Reduce batch size or model size
```python
# Use smaller model
model = GIGRTransformer(d_model=256, num_encoder_layers=2)

# Or use gradient accumulation
for i, batch in enumerate(loader):
    loss = compute_loss(batch)
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Issue: Slow Training

**Solution:** Enable mixed precision training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in loader:
    with autocast():
        output = model(graphs_data, target)
        loss = criterion(output, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Issue: Poor BLEU Scores

**Solutions:**
1. Increase model capacity (more layers/dimensions)
2. Train longer (more epochs)
3. Use learning rate scheduling
4. Increase training data
5. Verify preprocessing is correct

## Performance Tips

1. **Use GPU:** Ensure model and data are on GPU
2. **Batch processing:** Process multiple glosses when possible
3. **Mixed precision:** Use FP16 for faster training
4. **Gradient checkpointing:** For very deep models
5. **Data parallelism:** Use multiple GPUs with DataParallel

## Citation

If you use this implementation, please cite:

```bibtex
@article{nguyen2026gigr,
  title={Gloss-Internal Graph Construction and Encoding for Sign Language Translation},
  author={Nguyen-Xuan, Sam and Nguyen, Han},
  journal={Journal Not Specified},
  year={2026}
}
```

## Support

For issues or questions:
- GitHub Issues: (if repository is public)
- Email: samnx2@fe.edu.vn
