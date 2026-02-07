# GIGR Implementation - Project Summary

## What is GIGR?

GIGR (Gloss-Internal Graph Construction and Encoding) is a novel framework for Sign Language Translation that treats compound glosses as **directed, labeled graphs** rather than flat token sequences.

### The Problem

Traditional approaches treat glosses like "IX-2P-WAKE-UP" as either:
1. **Atomic tokens** → High vocabulary sparsity
2. **Flat sequences** → Loses internal structure

Example: WordPiece might split "IX-2P-WAKE-UP" into [IX], [-], [2], [P], [-], [WAKE], [UP]
- This loses the relationships: 2P modifies WAKE-UP, IX points to 2P, etc.

### The GIGR Solution

GIGR explicitly models internal structure:

```
IX-2P-WAKE-UP becomes:

    IX ─────reference────┐
     ↑                   ↓
     │                 WAKE (root)
     │                   ↑
     └──rel_agreement──2P
                         ↑
                         │
                        UP─modification─┘
```

This graph captures:
- WAKE is the lexical root
- IX provides reference (indexical pointing)
- 2P provides agreement (second person)
- UP provides aspect (completion)
- 2P refines IX (agreement depends on reference)

## Implementation Overview

### Core Components

1. **FST Parser** (`fst_parser.py`)
   - Deterministic finite-state transducer
   - Parses compound glosses into components
   - Assigns functional roles (REFERENCE, AGREEMENT, ASPECT, etc.)
   - O(n) time complexity

2. **Graph Constructor** (`graph_construction.py`)
   - Builds directed labeled graphs from parsed components
   - Implements Algorithm 1 from the paper
   - Supports visualization and batch processing

3. **Graph Encoder** (`graph_encoding.py`)
   - Encodes graphs to fixed-dimensional embeddings
   - Relation-aware message passing (Equation 3)
   - Attention-based pooling (Equation 4)

4. **GIGR Model** (`gigr_model.py`)
   - Complete Transformer-based translation model
   - Integrates graph encoding with standard Transformer
   - No modification to Transformer architecture needed

5. **Training/Evaluation** (`train_eval.py`)
   - Training loop with validation
   - BLEU-1 through BLEU-4 evaluation
   - Latency profiling (Equation 8)

## Key Results from the Paper

### ASLG-PC12 Dataset
- **BLEU-4: 78.90** (vs 75.70 baseline, +3.2 improvement)
- **chrF: 85.70** (vs 83.60 baseline)
- 7.0 BLEU-4 improvement over rule-based decomposition

### PHOENIX-2014T Dataset
- **BLEU-4: 26.80** (vs 24.40 baseline, +2.4 improvement)
- **chrF: 43.85** (vs 42.95 baseline)
- Consistent gains across all metrics

### Latency (batch_size=32)
- 2 layers: ~22 ms/sample
- 4 layers: ~35 ms/sample
- 6 layers: ~58 ms/sample

## File Structure

```
gigr/
├── fst_parser.py           # FST for morphological analysis
├── graph_construction.py   # Graph building (Algorithm 1)
├── graph_encoding.py       # Graph-to-embedding (Eq. 3-4)
├── gigr_model.py          # Complete GIGR+Transformer
├── train_eval.py          # Training and evaluation
├── example_full.py        # Complete usage example
├── test_gigr.py          # Test suite
├── __init__.py           # Package initialization
├── requirements.txt      # Dependencies
└── README.md            # Documentation
```

## Quick Start

### Installation
```bash
cd gigr
pip install -r requirements.txt
```

### Basic Usage
```python
from gigr import GlossGraph, GIGRTransformer

# Visualize a gloss graph
graph = GlossGraph("IX-2P-WAKE-UP")
graph.visualize()

# Create translation model
model = GIGRTransformer(
    vocab_size=1000,
    target_vocab_size=5000,
    d_model=512,
    nhead=8,
    num_encoder_layers=4,
    num_decoder_layers=4
)

# Train and translate
# (see example_full.py for complete pipeline)
```

### Run Complete Example
```bash
python example_full.py
```

## Technical Highlights

### 1. Linguistically-Grounded FST
The FST implements established findings from sign language morphology:
- Referential markers precede agreement (Meir 2002)
- Agreement depends on reference establishment
- Aspectual markers follow predicates (Liddell 2003)

### 2. Efficient Graph Encoding
- Single-layer GNN: Fast inference
- Multi-layer GNN: Better structure exploitation
- Attention pooling: Weighted node aggregation

### 3. Modular Design
- Drop-in replacement for linear encoders
- No changes to Transformer architecture
- Compatible with existing training pipelines

## Advantages Over Baselines

### vs Rule-Based Gloss Decomposition (RBGD)
- RBGD: IX-2P-WAKE-UP → [IX, 2P, WAKE, UP] (flat)
- GIGR: Preserves dependencies and relations
- **+7.0 BLEU-4 on ASLG-PC12**

### vs Linear Gloss Sequence Encoding (LGSE)
- LGSE: Subword tokenization (statistical, structure-agnostic)
- GIGR: Explicit linguistic structure
- **+3.2 BLEU-4 on ASLG-PC12**
- **+2.4 BLEU-4 on PHOENIX-2014T**

## When to Use GIGR

**Use GIGR when:**
- Working with compound glosses (hyphen-separated)
- Dataset has high gloss structural complexity
- Need interpretable intermediate representations
- Want to preserve linguistic dependencies

**Consider alternatives when:**
- Glosses are mostly atomic (single words)
- Computational resources are very limited
- Annotation conventions differ significantly

## Limitations

1. **Rule-based FST**: Requires manual rules for new annotation schemes
2. **Batch size**: Current implementation optimized for batch_size=1
3. **Gloss-to-Text only**: Doesn't include video-based sign recognition

## Future Extensions

1. **Hybrid FST-Neural**: Learn decomposition rules from data
2. **Inter-gloss graphs**: Model discourse-level dependencies
3. **End-to-end video**: Integrate with sign recognition
4. **Multilingual**: Extend to more sign languages

## Research Context

This implementation is based on:

**Paper:** "Gloss-Internal Graph Construction and Encoding for Sign Language Translation"  
**Authors:** Sam Nguyen-Xuan, Han Nguyen  
**Affiliation:** Swinburne Vietnam, FPT University & University of South Florida  
**Date:** February 6, 2026  
**Status:** Submitted to Journal Not Specified

The paper addresses a key limitation in sign language translation: existing methods treat glosses as flat sequences, ignoring their internal compositional structure. GIGR explicitly models this structure as graphs, leading to more accurate and robust translation.

## Dependencies

**Core:**
- PyTorch >= 2.1.0
- NumPy >= 1.24.0
- NetworkX >= 3.0 (for graphs)

**Optional:**
- Matplotlib >= 3.7.0 (for visualization)
- CUDA (for GPU acceleration)

## Citation

```bibtex
@article{nguyen2026gigr,
  title={Gloss-Internal Graph Construction and Encoding for Sign Language Translation},
  author={Nguyen-Xuan, Sam and Nguyen, Han},
  journal={Journal Not Specified},
  year={2026}
}
```

## Contact

**Sam Nguyen-Xuan**  
Department of Computer Science  
Swinburne Vietnam, FPT University  
Email: samnx2@fe.edu.vn

## License

Creative Commons Attribution 4.0 International License

## Acknowledgments

This implementation faithfully reproduces the methodology described in the paper and includes all key components: FST-based parsing, graph construction (Algorithm 1), relation-aware encoding (Equations 3-4), and integration with Transformer models.

---

**Generated:** February 2026  
**Version:** 1.0.0  
**Implementation Status:** Complete and tested
