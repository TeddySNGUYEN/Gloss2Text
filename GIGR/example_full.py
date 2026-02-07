"""
Complete Example: Training and Evaluating GIGR Model
Demonstrates the full pipeline from data preparation to evaluation
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json

from gigr_model import GIGRTransformer, GlossGraphPreprocessor
from train_eval import GlossToTextDataset, GIGRTrainer, BLEUEvaluator, collate_fn, LatencyProfiler
from graph_construction import GlossGraph


def build_vocabulary(sequences: list, min_freq: int = 1) -> dict:
    """Build vocabulary from sequences"""
    from collections import Counter
    
    # Flatten and count
    all_tokens = []
    for seq in sequences:
        all_tokens.extend(seq)
    
    counter = Counter(all_tokens)
    
    # Create vocab
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = idx
            idx += 1
    
    return vocab


def prepare_aslg_style_data():
    """
    Prepare sample data in ASLG-PC12 style
    This is a minimal example - real data would come from the dataset
    """
    # Sample gloss sequences (compound glosses with hyphens)
    train_glosses = [
        ['IX-2P-WAKE-UP'],
        ['BATH', 'NEED', 'YOU'],
        ['YOUR', 'NEW', 'CAR', 'COLOR', 'WHAT'],
        ['LONG', 'SEE', 'NO', 'HOW', 'YOU'],
        ['YOU', 'WORK', 'WHERE'],
        ['IX-1P-LOVE-3P', 'MUSIC'],
        ['NOT-FINISH', 'HOMEWORK'],
        ['IX-3P', 'SMART', 'VERY'],
    ]
    
    # Corresponding English translations
    train_targets = [
        ['did', 'you', 'wake', 'up'],
        ['you', 'need', 'to', 'take', 'a', 'bath'],
        ['what', 'color', 'is', 'your', 'new', 'car'],
        ['long', 'time', 'no', 'see', 'how', 'are', 'you', 'doing'],
        ['where', 'do', 'you', 'work'],
        ['i', 'love', 'music'],
        ['homework', 'not', 'finished'],
        ['he', 'is', 'very', 'smart'],
    ]
    
    # Simple dev/test split
    split_idx = 6
    dev_glosses = train_glosses[split_idx:]
    dev_targets = train_targets[split_idx:]
    train_glosses = train_glosses[:split_idx]
    train_targets = train_targets[:split_idx]
    
    return {
        'train': (train_glosses, train_targets),
        'dev': (dev_glosses, dev_targets)
    }


def main():
    """Main training and evaluation pipeline"""
    
    print("="*70)
    print("GIGR Model - Complete Training and Evaluation Example")
    print("Gloss-Internal Graph Construction and Encoding")
    print("="*70)
    
    # 1. Prepare data
    print("\n[Step 1] Preparing data...")
    data = prepare_aslg_style_data()
    train_glosses, train_targets = data['train']
    dev_glosses, dev_targets = data['dev']
    
    print(f"  Training examples: {len(train_glosses)}")
    print(f"  Dev examples: {len(dev_glosses)}")
    print(f"  Sample input: {train_glosses[0]}")
    print(f"  Sample target: {' '.join(train_targets[0])}")
    
    # 2. Build vocabularies
    print("\n[Step 2] Building vocabularies...")
    
    # Build source vocab from gloss components
    all_components = []
    for gloss_seq in train_glosses + dev_glosses:
        for gloss in gloss_seq:
            # Split compound glosses
            components = gloss.replace('-', ' ').replace('+', ' ').split()
            all_components.extend(components)
    
    source_vocab = build_vocabulary([all_components])
    source_vocab['<UNK>'] = len(source_vocab)
    
    # Build target vocab
    target_vocab = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
    idx = 4
    for target_seq in train_targets + dev_targets:
        for word in target_seq:
            if word not in target_vocab:
                target_vocab[word] = idx
                idx += 1
    
    print(f"  Source vocabulary size: {len(source_vocab)}")
    print(f"  Target vocabulary size: {len(target_vocab)}")
    
    # 3. Create datasets
    print("\n[Step 3] Creating datasets...")
    
    train_dataset = GlossToTextDataset(
        train_glosses, train_targets,
        source_vocab, target_vocab
    )
    
    dev_dataset = GlossToTextDataset(
        dev_glosses, dev_targets,
        source_vocab, target_vocab
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Current implementation supports batch_size=1
        shuffle=True,
        collate_fn=collate_fn
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 4. Create model
    print("\n[Step 4] Creating GIGR model...")
    
    # Hyperparameters from Table 2 (scaled down for demo)
    model = GIGRTransformer(
        vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        d_model=256,  # 512 in paper
        nhead=8,
        num_encoder_layers=2,  # 2/4/6 in paper
        num_decoder_layers=2,
        dim_feedforward=1024,  # 2048 in paper
        dropout=0.2,
        num_graph_layers=1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    print(f"  Encoder layers: 2")
    print(f"  Decoder layers: 2")
    print(f"  Attention heads: 8")
    
    # 5. Setup training
    print("\n[Step 5] Setting up training...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    # Optimizer: AdamW with lr=5e-5 (from Table 2)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=target_vocab['<PAD>'])
    
    # Create trainer
    trainer = GIGRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=dev_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        pad_idx=target_vocab['<PAD>']
    )
    
    # 6. Train model
    print("\n[Step 6] Training model...")
    print("-" * 70)
    
    num_epochs = 10  # 10 epochs as per Table 2
    trainer.train(num_epochs=num_epochs)
    
    # 7. Evaluate with BLEU
    print("\n[Step 7] Evaluating with BLEU scores...")
    
    model.eval()
    idx_to_word = {v: k for k, v in target_vocab.items()}
    
    references = []
    hypotheses = []
    
    preprocessor = GlossGraphPreprocessor(source_vocab)
    
    print("  Generating translations...")
    for i, (gloss_seq, target_seq) in enumerate(zip(dev_glosses, dev_targets)):
        # Prepare input
        graphs_data = preprocessor.process_gloss_sequence(gloss_seq)
        for graph_data in graphs_data:
            graph_data['node_texts'] = graph_data['node_texts'].to(device)
            graph_data['edge_index'] = graph_data['edge_index'].to(device)
        
        # Generate
        with torch.no_grad():
            generated = model.generate(
                graphs_data,
                max_len=50,
                bos_idx=target_vocab['<BOS>'],
                eos_idx=target_vocab['<EOS>']
            )
        
        # Convert to words
        generated_words = []
        for idx in generated[1:-1]:  # Skip BOS/EOS
            word = idx_to_word.get(idx.item(), '<UNK>')
            if word not in ['<BOS>', '<EOS>', '<PAD>']:
                generated_words.append(word)
        
        references.append(target_seq)
        hypotheses.append(generated_words)
        
        if i == 0:  # Show first example
            print(f"\n  Example translation:")
            print(f"    Source: {' '.join(gloss_seq)}")
            print(f"    Reference: {' '.join(target_seq)}")
            print(f"    Generated: {' '.join(generated_words)}")
    
    # Compute BLEU
    evaluator = BLEUEvaluator()
    bleu_scores = evaluator.compute_bleu(references, hypotheses, max_n=4)
    
    print("\n  BLEU Scores on Dev Set:")
    print("  " + "-" * 30)
    for metric in ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']:
        print(f"  {metric}: {bleu_scores[metric]:.2f}")
    
    # 8. Profile latency
    print("\n[Step 8] Profiling end-to-end latency...")
    
    profiler = LatencyProfiler(model, device=device)
    
    test_gloss = dev_glosses[0]
    latency_results = profiler.profile(
        gloss_sequence=test_gloss,
        preprocessor=preprocessor,
        num_runs=50
    )
    
    print(f"\n  Latency Analysis (averaged over 50 runs):")
    print("  " + "-" * 50)
    print(f"  Preprocessing:  {latency_results['preprocessing_mean']:.2f} ± "
          f"{latency_results['preprocessing_std']:.2f} ms")
    print(f"  Encoding:       {latency_results['encoding_mean']:.2f} ± "
          f"{latency_results['encoding_std']:.2f} ms")
    print(f"  Decoding (1 step): {latency_results['decoding_mean']:.2f} ± "
          f"{latency_results['decoding_std']:.2f} ms")
    print(f"  Total:          {latency_results['total_mean']:.2f} ms")
    
    # 9. Visualize a gloss graph
    print("\n[Step 9] Visualizing gloss-internal graph...")
    
    sample_gloss = "IX-2P-WAKE-UP"
    graph = GlossGraph(sample_gloss)
    
    print(f"\n  Graph structure for '{sample_gloss}':")
    print(f"  Nodes: {graph.n_nodes}")
    print(f"  Root: {graph.components[graph.root_idx].text}")
    print(f"  Edges:")
    
    for edge in graph.edges:
        src = graph.components[edge.source]
        tgt = graph.components[edge.target]
        print(f"    {src.text} --[{edge.relation.value}]--> {tgt.text}")
    
    # Try to visualize (if matplotlib available)
    try:
        print("\n  Generating graph visualization...")
        graph.visualize(save_path='/home/claude/graph_viz.png')
        print("  Saved to: /home/claude/graph_viz.png")
    except:
        print("  (Visualization skipped - matplotlib not available)")
    
    # 10. Save model
    print("\n[Step 10] Saving model...")
    
    save_path = '/home/claude/gigr_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'source_vocab': source_vocab,
        'target_vocab': target_vocab,
        'config': {
            'd_model': 256,
            'nhead': 8,
            'num_encoder_layers': 2,
            'num_decoder_layers': 2,
            'dim_feedforward': 1024,
            'dropout': 0.2,
            'num_graph_layers': 1
        }
    }, save_path)
    
    print(f"  Model saved to: {save_path}")
    
    # Summary
    print("\n" + "="*70)
    print("Training and Evaluation Complete!")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  Best Dev BLEU-4: {bleu_scores['BLEU-4']:.2f}")
    print(f"  Inference Latency: {latency_results['total_mean']:.2f} ms/sample")
    print(f"  Model Parameters: {total_params:,}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
