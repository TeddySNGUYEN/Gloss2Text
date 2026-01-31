import torch
from .metrics import bleu_chrf_from_texts, decode_ids_to_text

from .linear_baseline import greedy_decode_linear
from .gigp_model import greedy_decode_gigp
from .transformer_no_graph import greedy_decode_nograph

@torch.no_grad()
def eval_bleu_chrf_linear(model, loader, tgt_vocab, max_len=64):
    model.eval()
    hyps, refs = [], []
    for src_pad, _, _, _, _, _, tgt_texts in loader:
        out_ids = greedy_decode_linear(model, src_pad, max_len=max_len)
        for b in range(out_ids.size(0)):
            seq = out_ids[b].tolist()
            if seq and seq[0] == tgt_vocab.bos_id:
                seq = seq[1:]
            if tgt_vocab.eos_id in seq:
                seq = seq[:seq.index(tgt_vocab.eos_id)]
            hyps.append(decode_ids_to_text(seq, tgt_vocab))
        refs.extend(tgt_texts)
    return bleu_chrf_from_texts(hyps, refs)

@torch.no_grad()
def eval_bleu_chrf_nograph(model, loader, tgt_vocab, max_len=64):
    model.eval()
    hyps, refs = [], []
    for src_pad, src_lens, _, _, _, _, tgt_texts in loader:
        out_ids = greedy_decode_nograph(model, src_pad, src_lens, max_len=max_len)
        for b in range(out_ids.size(0)):
            seq = out_ids[b].tolist()
            if seq and seq[0] == tgt_vocab.bos_id:
                seq = seq[1:]
            if tgt_vocab.eos_id in seq:
                seq = seq[:seq.index(tgt_vocab.eos_id)]
            hyps.append(decode_ids_to_text(seq, tgt_vocab))
        refs.extend(tgt_texts)
    return bleu_chrf_from_texts(hyps, refs)

@torch.no_grad()
def eval_bleu_chrf_gigp(model, loader, tgt_vocab, device, max_len=64):
    model.eval()
    hyps, refs = [], []
    for _, _, _, _, raw_gloss_tokens, _, tgt_texts in loader:
        out_ids = greedy_decode_gigp(model, raw_gloss_tokens, device=device, max_len=max_len)
        for b in range(out_ids.size(0)):
            seq = out_ids[b].tolist()
            if seq and seq[0] == tgt_vocab.bos_id:
                seq = seq[1:]
            if tgt_vocab.eos_id in seq:
                seq = seq[:seq.index(tgt_vocab.eos_id)]
            hyps.append(decode_ids_to_text(seq, tgt_vocab))
        refs.extend(tgt_texts)
    return bleu_chrf_from_texts(hyps, refs)
