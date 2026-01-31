import os
import torch

from .utils import set_seed, get_device
from .data import CSVGlossTextDataset, split_full_csv, build_vocabs, make_loader
from .rule_based import rule_based_translate
from .metrics import bleu_chrf_from_texts

from .linear_baseline import LinearBoWSeq2Seq, train_one_epoch_linear, eval_loss_linear
from .transformer_no_graph import TransformerNoGraph, train_one_epoch_nograph, eval_loss_nograph
from .graph import build_graph_vocabs
from .gigp_model import GIGPTransformer, train_one_epoch_gigp, eval_loss_gigp

from .train_eval import eval_bleu_chrf_linear, eval_bleu_chrf_nograph, eval_bleu_chrf_gigp
from .latency import benchmark_all, print_latency

def eval_rule_based(test_ds):
    hyps, refs = [], []
    for gloss_tokens, _, _, tgt in test_ds:
        hyps.append(rule_based_translate(gloss_tokens))
        refs.append(tgt)
    return bleu_chrf_from_texts(hyps, refs)

def run_pipeline(
    data_dir: str,
    use_cuda: bool = True,
    make_splits: bool = False,
    full_csv: str = "train.csv",
    batch_size: int = 16,
    lr: float = 5e-5,
    seed: int = 42,
    epochs_linear: int = 3,
    epochs_nograph: int = 3,
    epochs_gigp: int = 3,
    d_model: int = 512,
    nhead: int = 8,
    enc_layers: int = 4,
    dec_layers: int = 4,
    ffn_dim: int = 2048,
    dropout: float = 0.1,
    eval_max_len: int = 64,
    lat_warmup: int = 30,
    lat_runs: int = 200,
):
    set_seed(seed)
    device = get_device(use_cuda)
    print("Device:", device)

    if make_splits:
        split_full_csv(data_dir, full_csv=full_csv, seed=seed)

    train_path = os.path.join(data_dir, "train.csv")
    dev_path   = os.path.join(data_dir, "dev.csv")
    test_path  = os.path.join(data_dir, "test.csv")
    for p in [train_path, dev_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    train_ds = CSVGlossTextDataset(train_path)
    dev_ds   = CSVGlossTextDataset(dev_path)
    test_ds  = CSVGlossTextDataset(test_path)
    print(f"train/dev/test = {len(train_ds)}/{len(dev_ds)}/{len(test_ds)}")

    src_vocab, tgt_vocab = build_vocabs(train_ds, min_freq=1)
    print(f"src_vocab={len(src_vocab)} tgt_vocab={len(tgt_vocab)}")

    train_loader = make_loader(train_ds, src_vocab, tgt_vocab, device=device, batch_size=batch_size, shuffle=True)
    dev_loader   = make_loader(dev_ds,   src_vocab, tgt_vocab, device=device, batch_size=batch_size, shuffle=False)
    test_loader  = make_loader(test_ds,  src_vocab, tgt_vocab, device=device, batch_size=batch_size, shuffle=False)

    # 1) Rule-based
    rb_bleu, rb_chrf = eval_rule_based(test_ds)
    print(f"[Rule-based] BLEU={rb_bleu:.2f} chrF={rb_chrf:.2f}")

    # 2) Linear baseline
    linear = LinearBoWSeq2Seq(src_vocab, tgt_vocab, d_model=d_model, max_len=128).to(device)
    opt_lin = torch.optim.Adam(linear.parameters(), lr=lr)
    for ep in range(1, epochs_linear + 1):
        tr = train_one_epoch_linear(linear, train_loader, opt_lin, tgt_vocab)
        dv = eval_loss_linear(linear, dev_loader, tgt_vocab)
        print(f"[Linear] epoch {ep:02d} train_loss={tr:.4f} dev_loss={dv:.4f}")
    lin_bleu, lin_chrf = eval_bleu_chrf_linear(linear, test_loader, tgt_vocab, max_len=eval_max_len)
    print(f"[Linear] BLEU={lin_bleu:.2f} chrF={lin_chrf:.2f}")

    # 3) Transformer (no graph)
    nograph = TransformerNoGraph(
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        d_model=d_model, nhead=nhead,
        enc_layers=enc_layers, dec_layers=dec_layers,
        ffn_dim=ffn_dim, dropout=dropout
    ).to(device)
    opt_ng = torch.optim.Adam(nograph.parameters(), lr=lr)
    for ep in range(1, epochs_nograph + 1):
        tr = train_one_epoch_nograph(nograph, train_loader, opt_ng, tgt_vocab)
        dv = eval_loss_nograph(nograph, dev_loader, tgt_vocab)
        print(f"[NoGraphTr] epoch {ep:02d} train_loss={tr:.4f} dev_loss={dv:.4f}")
    ng_bleu, ng_chrf = eval_bleu_chrf_nograph(nograph, test_loader, tgt_vocab, max_len=eval_max_len)
    print(f"[NoGraphTr] BLEU={ng_bleu:.2f} chrF={ng_chrf:.2f}")

    # 4) GIGP
    node_vocab, role_vocab, rel_vocab = build_graph_vocabs(train_ds, min_freq=1)
    print(f"node={len(node_vocab)} role={len(role_vocab)} rel={len(rel_vocab)}")

    gigp = GIGPTransformer(
        node_vocab=node_vocab, role_vocab=role_vocab, rel_vocab=rel_vocab, tgt_vocab=tgt_vocab,
        d_model=d_model, nhead=nhead, enc_layers=enc_layers, dec_layers=dec_layers,
        ffn_dim=ffn_dim, dropout=dropout
    ).to(device)
    opt_gigp = torch.optim.Adam(gigp.parameters(), lr=lr)
    for ep in range(1, epochs_gigp + 1):
        tr = train_one_epoch_gigp(gigp, train_loader, opt_gigp, tgt_vocab)
        dv = eval_loss_gigp(gigp, dev_loader, tgt_vocab)
        print(f"[GIGP]   epoch {ep:02d} train_loss={tr:.4f} dev_loss={dv:.4f}")
    gigp_bleu, gigp_chrf = eval_bleu_chrf_gigp(gigp, test_loader, tgt_vocab, device=device, max_len=eval_max_len)
    print(f"[GIGP] BLEU={gigp_bleu:.2f} chrF={gigp_chrf:.2f}")

    # 5) End-to-end latency
    lat = benchmark_all(
        device=device,
        test_ds=test_ds,
        test_loader=test_loader,
        linear_model=linear,
        nograph_model=nograph,
        gigp_model=gigp,
        max_len=eval_max_len,
        warmup=lat_warmup,
        runs=lat_runs,
    )
    print_latency(lat)

    return {
        "device": device,
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "linear": linear,
        "nograph": nograph,
        "gigp": gigp,
        "latency": lat,
    }
