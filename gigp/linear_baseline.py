import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import Vocab
from .utils import AvgMeter

class LinearBoWSeq2Seq(nn.Module):
    def __init__(self, src_vocab: Vocab, tgt_vocab: Vocab, d_model=512, max_len=128):
        super().__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        self.src_proj = nn.Linear(len(src_vocab), d_model, bias=False)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.out = nn.Linear(d_model, len(tgt_vocab))

    def forward(self, src_ids: torch.Tensor, tgt_inp_ids: torch.Tensor):
        B, S = src_ids.shape
        B2, T = tgt_inp_ids.shape
        assert B == B2

        bow = torch.zeros((B, len(self.src_vocab)), device=src_ids.device, dtype=torch.float32)
        bow.scatter_add_(1, src_ids, torch.ones_like(src_ids, dtype=torch.float32))
        bow = bow / (bow.sum(dim=1, keepdim=True) + 1e-6)

        h = self.src_proj(bow)
        pos = torch.arange(T, device=src_ids.device).clamp(max=self.max_len - 1)
        pos = pos.unsqueeze(0).expand(B, -1)
        hp = h.unsqueeze(1) + self.pos_emb(pos)

        return self.out(hp)

@torch.inference_mode()
def greedy_decode_linear(model: LinearBoWSeq2Seq, src_ids: torch.Tensor, max_len=64):
    device = src_ids.device
    B = src_ids.size(0)
    ys = torch.full((B, 1), model.tgt_vocab.bos_id, dtype=torch.long, device=device)
    for _ in range(max_len):
        logits = model(src_ids, ys)
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        ys = torch.cat([ys, next_tok], dim=1)
        if torch.all(next_tok.squeeze(1) == model.tgt_vocab.eos_id):
            break
    return ys

def train_one_epoch_linear(model, loader, optimizer, tgt_vocab: Vocab, grad_clip=1.0):
    model.train()
    meter = AvgMeter()
    for src_pad, _, tgt_pad, _, _, _, _ in loader:
        tgt_inp = tgt_pad[:, :-1]
        tgt_out = tgt_pad[:, 1:]
        logits = model(src_pad, tgt_inp)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=tgt_vocab.pad_id
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        meter.update(loss.item(), n=tgt_pad.size(0))
    return meter.avg

@torch.no_grad()
def eval_loss_linear(model, loader, tgt_vocab: Vocab):
    model.eval()
    meter = AvgMeter()
    for src_pad, _, tgt_pad, _, _, _, _ in loader:
        tgt_inp = tgt_pad[:, :-1]
        tgt_out = tgt_pad[:, 1:]
        logits = model(src_pad, tgt_inp)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=tgt_vocab.pad_id
        )
        meter.update(loss.item(), n=tgt_pad.size(0))
    return meter.avg
