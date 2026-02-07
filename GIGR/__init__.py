"""
GIGR: Gloss-Internal Graph Construction and Encoding for Sign Language Translation

This package implements the GIGR framework for Sign Language Translation,
which represents compound glosses as directed, labeled graphs.

Main modules:
- fst_parser: Finite State Transducer for morphological analysis
- graph_construction: Building directed labeled graphs from glosses
- graph_encoding: Graph-to-embedding encoding with relation-aware aggregation
- gigr_model: Complete GIGR + Transformer model for translation
- train_eval: Training loop, evaluation metrics, and profiling

Example usage:
    >>> from gigr import GlossGraph, GIGRTransformer
    >>> 
    >>> # Visualize a gloss graph
    >>> graph = GlossGraph("IX-2P-WAKE-UP")
    >>> graph.visualize()
    >>> 
    >>> # Create translation model
    >>> model = GIGRTransformer(
    ...     vocab_size=1000,
    ...     target_vocab_size=5000,
    ...     d_model=512
    ... )
"""

__version__ = "1.0.0"
__author__ = "Sam Nguyen-Xuan, Han Nguyen"
__email__ = "samnx2@fe.edu.vn"

# Import main classes for convenient access
from .fst_parser import GlossFST, Role, Relation, Component
from .graph_construction import GlossGraph, GlossGraphBatch, Edge
from .graph_encoding import GraphEncoder, GlossSequenceEncoder
from .gigr_model import GIGRTransformer, GlossGraphPreprocessor
from .train_eval import (
    GlossToTextDataset,
    GIGRTrainer,
    BLEUEvaluator,
    LatencyProfiler
)

__all__ = [
    # FST parser
    'GlossFST',
    'Role',
    'Relation',
    'Component',
    
    # Graph construction
    'GlossGraph',
    'GlossGraphBatch',
    'Edge',
    
    # Graph encoding
    'GraphEncoder',
    'GlossSequenceEncoder',
    
    # Model
    'GIGRTransformer',
    'GlossGraphPreprocessor',
    
    # Training and evaluation
    'GlossToTextDataset',
    'GIGRTrainer',
    'BLEUEvaluator',
    'LatencyProfiler',
]


def print_system_info():
    """Print system and package information"""
    import torch
    import sys
    
    print("="*60)
    print("GIGR System Information")
    print("="*60)
    print(f"GIGR Version: {__version__}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*60)


# Print info on import (optional - can comment out)
# print_system_info()
