"""
Quick Test Script for GIGR Implementation
Verifies that all components work correctly
"""
import torch
import sys

def test_fst_parser():
    """Test FST parser"""
    print("\n[Test 1] FST Parser")
    print("-" * 50)
    
    from gigr.fst_parser import GlossFST
    
    parser = GlossFST()
    test_glosses = [
        "IX-2P-WAKE-UP",
        "BATH",
        "NOT-FINISH",
        "IX-1P-LOVE-3P"
    ]
    
    for gloss in test_glosses:
        components, root_idx = parser.parse(gloss)
        print(f"\n  Gloss: {gloss}")
        print(f"  Root: {components[root_idx].text}")
        for comp in components:
            print(f"    {comp.text} -> {comp.role.value}")
    
    print("\n  ✓ FST Parser test passed")
    return True


def test_graph_construction():
    """Test graph construction"""
    print("\n[Test 2] Graph Construction")
    print("-" * 50)
    
    from gigr.graph_construction import GlossGraph
    
    gloss = "IX-2P-WAKE-UP"
    graph = GlossGraph(gloss)
    
    print(f"\n  Gloss: {gloss}")
    print(f"  Nodes: {graph.n_nodes}")
    print(f"  Edges: {len(graph.edges)}")
    print(f"  Root: {graph.components[graph.root_idx].text}")
    
    print("\n  Edge structure:")
    for edge in graph.edges:
        src = graph.components[edge.source]
        tgt = graph.components[edge.target]
        print(f"    {src.text} --[{edge.relation.value}]--> {tgt.text}")
    
    # Test edge index format
    edge_index, edge_types = graph.get_edge_index()
    print(f"\n  Edge index shape: {edge_index.shape}")
    
    print("\n  ✓ Graph Construction test passed")
    return True


def test_graph_encoding():
    """Test graph encoding"""
    print("\n[Test 3] Graph Encoding")
    print("-" * 50)
    
    from gigr.graph_construction import GlossGraph
    from gigr.graph_encoding import GraphEncoder
    from gigr.fst_parser import Role
    
    # Create simple vocab
    vocab = {'IX': 0, '2P': 1, 'WAKE': 2, 'UP': 3}
    
    # Create encoder
    encoder = GraphEncoder(
        vocab_size=len(vocab),
        embedding_dim=64,
        num_layers=1
    )
    
    # Create graph
    gloss = "IX-2P-WAKE-UP"
    graph = GlossGraph(gloss)
    
    # Prepare inputs
    node_texts = torch.tensor([vocab.get(c.text, 0) for c in graph.components])
    node_roles = [c.role for c in graph.components]
    edge_index, edge_relations = graph.get_edge_index()
    
    # Encode
    with torch.no_grad():
        graph_emb = encoder(node_texts, node_roles, edge_index, edge_relations)
    
    print(f"\n  Input gloss: {gloss}")
    print(f"  Number of nodes: {len(node_texts)}")
    print(f"  Graph embedding shape: {graph_emb.shape}")
    print(f"  Expected: [1, 64]")
    
    assert graph_emb.shape == (1, 64), "Unexpected embedding shape"
    
    print("\n  ✓ Graph Encoding test passed")
    return True


def test_gigr_model():
    """Test complete GIGR model"""
    print("\n[Test 4] GIGR Model")
    print("-" * 50)
    
    from gigr.gigr_model import GIGRTransformer, GlossGraphPreprocessor
    
    # Create vocabularies
    source_vocab = {
        'IX': 0, '2P': 1, 'WAKE': 2, 'UP': 3,
        'BATH': 4, '<PAD>': 5, '<UNK>': 6
    }
    
    target_vocab = {
        '<PAD>': 0, '<BOS>': 1, '<EOS>': 2,
        'you': 3, 'wake': 4, 'up': 5, '<UNK>': 6
    }
    
    # Create model
    model = GIGRTransformer(
        vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        dropout=0.1
    )
    
    print(f"\n  Model created")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Prepare input
    preprocessor = GlossGraphPreprocessor(source_vocab)
    gloss_sequence = ["IX-2P-WAKE-UP"]
    graphs_data = preprocessor.process_gloss_sequence(gloss_sequence)
    
    # Test generation
    with torch.no_grad():
        generated = model.generate(
            graphs_data,
            max_len=10,
            bos_idx=target_vocab['<BOS>'],
            eos_idx=target_vocab['<EOS>']
        )
    
    print(f"\n  Input: {gloss_sequence}")
    print(f"  Generated sequence length: {len(generated)}")
    print(f"  Generated token IDs: {generated.tolist()}")
    
    print("\n  ✓ GIGR Model test passed")
    return True


def test_training_components():
    """Test training utilities"""
    print("\n[Test 5] Training Components")
    print("-" * 50)
    
    from gigr.train_eval import BLEUEvaluator
    
    # Test BLEU evaluator
    evaluator = BLEUEvaluator()
    
    references = [
        ['did', 'you', 'wake', 'up'],
        ['you', 'need', 'a', 'bath']
    ]
    
    hypotheses = [
        ['you', 'wake', 'up'],
        ['need', 'bath']
    ]
    
    scores = evaluator.compute_bleu(references, hypotheses, max_n=4)
    
    print(f"\n  BLEU Scores:")
    for metric, score in scores.items():
        print(f"    {metric}: {score:.2f}")
    
    assert all(0 <= score <= 100 for score in scores.values()), "Invalid BLEU scores"
    
    print("\n  ✓ Training Components test passed")
    return True


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("GIGR Implementation Test Suite")
    print("="*60)
    
    tests = [
        test_fst_parser,
        test_graph_construction,
        test_graph_encoding,
        test_gigr_model,
        test_training_components
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n  ✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"\n  Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n  ✓ All tests passed successfully!")
    else:
        print("\n  ✗ Some tests failed")
        sys.exit(1)
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Add parent directory to path
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    run_all_tests()
