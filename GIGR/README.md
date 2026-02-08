## Overview

This repository implements a novel approach to Gloss-to-Text translation that represents compound glosses as directed, labeled graphs rather than flat token sequences. The framework captures internal linguistic structure (reference, agreement, aspect, etc.) and integrates seamlessly with Transformer encoder-decoder architectures.

### Key Features

- **Gloss-Internal Graph Construction**: Parses compound glosses using a deterministic Finite State Transducer (FST)
- **Relation-Aware Graph Encoding**: Encodes graphs using graph neural network principles with relation-specific transformations
- **Transformer Integration**: Seamlessly integrates with standard Transformer encoder-decoder models
- **State-of-the-art Results**: Achieves gains of up to 3.2 BLEU-4 over Linear Gloss Sequence Encoding (LGSE) on ASLG-PC12

## Architecture

The GIGR framework consists of four main components:

```
Raw Gloss Text → FST Parser → Graph Construction → Graph Encoding → Transformer
```

1. **FST Parser** (`fst_parser.py`): Deterministic finite-state transducer for morphological analysis
2. **Graph Construction** (`graph_construction.py`): Builds directed labeled graphs from parsed components
3. **Graph Encoding** (`graph_encoding.py`): Encodes graphs to fixed-dimensional embeddings
4. **GIGR Model** (`gigr_model.py`): Complete Transformer-based translation model

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- Python 3.10+
- PyTorch 2.1+
- NetworkX (for graph visualization)
- NumPy
- Matplotlib (optional, for visualization)

## Quick Start

### Basic Usage

```python
from gigr_model import GIGRTransformer, GlossGraphPreprocessor
from graph_construction import GlossGraph

# 1. Visualize a gloss-internal graph
gloss = "IX-2P-WAKE-UP"
graph = GlossGraph(gloss)
graph.visualize()  # Saves visualization

# 2. Create and use the model
model = GIGRTransformer(
    vocab_size=1000,
    target_vocab_size=5000,
    d_model=512,
    nhead=8,
    num_encoder_layers=4,
    num_decoder_layers=4
)

# 3. Prepare input data
vocab = {'IX': 0, '2P': 1, 'WAKE': 2, 'UP': 3, ...}
preprocessor = GlossGraphPreprocessor(vocab)

gloss_sequence = ["IX-2P-WAKE-UP", "BATH"]
graphs_data = preprocessor.process_gloss_sequence(gloss_sequence)

# 4. Generate translation
generated = model.generate(
    graphs_data,
    max_len=50,
    bos_idx=1,
    eos_idx=2
)
```

### Complete Training Example

Run the complete training and evaluation pipeline:

```bash
python example_full.py
```

This demonstrates:
- Data preparation
- Vocabulary building
- Model training
- BLEU evaluation
- Latency profiling
- Graph visualization

## Module Documentation

### FST Parser (`fst_parser.py`)

Implements the deterministic finite-state transducer for gloss morphological analysis according to Table 1 in the paper.

**Key Classes:**
- `GlossFST`: Main parser class
- `Role`: Enum of functional roles (REFERENCE, AGREEMENT, ASPECT, etc.)
- `Relation`: Enum of edge relations

**Example:**
```python
from fst_parser import GlossFST

parser = GlossFST()
components, root_idx = parser.parse("IX-2P-WAKE-UP")

# Output:
# components[0]: IX (REFERENCE)
# components[1]: 2P (AGREEMENT)
# components[2]: WAKE (LEXICALROOT)
# components[3]: UP (ASPECT)
```

### Graph Construction (`graph_construction.py`)

Implements Algorithm 1 from the paper for building directed labeled graphs.

**Key Classes:**
- `GlossGraph`: Single gloss graph representation
- `GlossGraphBatch`: Batch multiple graphs

**Example:**
```python
from graph_construction import GlossGraph

graph = GlossGraph("IX-2P-WAKE-UP")
print(f"Nodes: {graph.n_nodes}")
print(f"Edges: {len(graph.edges)}")

# Get PyTorch Geometric format
edge_index, edge_types = graph.get_edge_index()
```

### Graph Encoding (`graph_encoding.py`)

Implements the graph-to-embedding encoding with relation-aware aggregation (Equations 3-4).

**Key Classes:**
- `GraphEncoder`: Main graph encoding module
- `RelationEncoder`: Encodes edge relation types
- `RoleEncoder`: Encodes node functional roles

**Example:**
```python
from graph_encoding import GraphEncoder

encoder = GraphEncoder(
    vocab_size=1000,
    embedding_dim=512,
    num_layers=1
)

graph_embedding = encoder(
    node_texts=tensor([0, 1, 2, 3]),
    node_roles=[Role.REFERENCE, Role.AGREEMENT, ...],
    edge_index=edge_index,
    edge_relations=[Relation.AGREEMENT_REL, ...]
)
```

### Training and Evaluation (`train_eval.py`)

Provides training loop, evaluation metrics, and profiling utilities.

**Key Classes:**
- `GIGRTrainer`: Training loop with validation
- `BLEUEvaluator`: BLEU-1 through BLEU-4 computation
- `LatencyProfiler`: End-to-end latency profiling

**Example:**
```python
from train_eval import GIGRTrainer

trainer = GIGRTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion
)

trainer.train(num_epochs=10)
```

## Hyperparameters

Default hyperparameters follow Table 2 from the paper:

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 5×10⁻⁵ |
| Batch Size | 32 |
| Epochs | 10 |
| Dropout | 0.2 |
| Attention Heads | 8 |
| Encoder/Decoder Layers | 2/4/6 |
| Embedding Dimension | 512 |
| FFN Hidden Dimension | 2048 |

## Results

On **ASLG-PC12** test set:
- BLEU-4: **78.90** (+3.2 over LGSE baseline, +7.0 over RBGD)
- chrF: **85.70**

On **PHOENIX-2014T** test set:
- BLEU-4: **26.80** (+2.4 over LGSE baseline)
- chrF: **43.85**

End-to-end latency (batch_size=32):
- 2 layers: ~22 ms/sample
- 4 layers: ~35 ms/sample
- 6 layers: ~58 ms/sample

## Graph Structure Examples

### Simple Gloss: "BATH"
```
Node: BATH (LEXICALROOT)
Edges: None
```

### Compound Gloss: "IX-2P-WAKE-UP"
```
Nodes:
  0: IX (REFERENCE)
  1: 2P (AGREEMENT)
  2: WAKE (LEXICALROOT) [ROOT]
  3: UP (ASPECT)

Edges:
  IX --[reference]--> WAKE
  2P --[agreement]--> WAKE
  UP --[modification]--> WAKE
  2P --[rel_agreement]--> IX  (refinement)
```

## File Structure

```
gigr/
├── fst_parser.py           # FST for morphological analysis
├── graph_construction.py   # Graph building (Algorithm 1)
├── graph_encoding.py       # Graph-to-embedding encoding
├── gigr_model.py          # Complete GIGR + Transformer model
├── train_eval.py          # Training and evaluation utilities
├── example_full.py        # Complete usage example
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Limitations

1. **Rule-based FST**: Graph construction uses hand-crafted rules derived from glossing conventions. May require adaptation for different annotation schemes.

2. **Batch size**: Current implementation optimized for batch_size=1. Batching multiple sequences requires padding and graph alignment.

3. **Gloss-to-Text only**: Framework focuses on the Gloss-to-Text stage. End-to-end video-to-text would require integration with sign recognition models.

## Future Work

- Hybrid FST-neural approaches for automatic rule learning
- Extension to inter-gloss dependencies (discourse structure)
- End-to-end video-based sign language translation
- Support for more sign languages and annotation schemes

## License

This code is released under the Creative Commons Attribution 4.0 International License, consistent with the paper submission.

