import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import Vocab
from .utils import AvgMeter

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class TransformerNoGraph(nn.Module):
    def __init__(
        self,
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        d_model=512,
        nhead=8,
        enc_layers=4,
        dec_layers=4,
        ffn_dim=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        self.src_emb = nn.Embedding(len(src_vocab), d_model, padding_idx=src_vocab.pad_id)
        self.src_pos = PositionalEncoding(d_model, dropout)

        self.tgt_emb = nn.Embedding(len(tgt_vocab), d_model, padding_idx=tgt_vocab.pad_id)
        self.tgt_pos = PositionalEncoding(d_model, dropout)

        self.tr = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.out = nn.Linear(d_model, len(tgt_vocab))

    @staticmethod
    def lengths_to_padmask(lengths: torch.Tensor, max_len: int):
        B = lengths.size(0)
        idx = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(B, -1)
        return idx >= lengths.unsqueeze(1)

    def encode(self, src_ids: torch.Tensor, src_lens: torch.Tensor):
        x = self.src_pos(self.src_emb(src_ids))
        src_kpm = self.lengths_to_padmask(src_lens, x.size(1))
        mem = self.tr.encoder(x, src_key_padding_mask=src_kpm)
        return mem, src_kpm

    def forward(self, src_ids: torch.Tensor, src_lens: torch.Tensor, tgt_inp_ids: torch.Tensor):
        device = src_ids.device
        mem, src_kpm = self.encode(src_ids, src_lens)

        tgt = self.tgt_pos(self.tgt_emb(tgt_inp_ids))
        tgt_kpm = (tgt_inp_ids == self.tgt_vocab.pad_id)

        T = tgt_inp_ids.size(1)
        tgt_causal = torch.triu(torch.ones((T, T), device=device), diagonal=1).bool()

        dec = self.tr.decoder(
            tgt, mem,
            tgt_mask=tgt_causal,
            tgt_key_padding_mask=tgt_kpm,
            memory_key_padding_mask=src_kpm,
        )
        return self.out(dec)

@torch.inference_mode()
def greedy_decode_nograph(model: TransformerNoGraph, src_ids: torch.Tensor, src_lens: torch.Tensor, max_len=64):
    device = src_ids.device
    model.eval()
    mem, src_kpm = model.encode(src_ids, src_lens)

    B = src_ids.size(0)
    ys = torch.full((B, 1), model.tgt_vocab.bos_id, dtype=torch.long, device=device)

    for _ in range(max_len):
        tgt = model.tgt_pos(model.tgt_emb(ys))
        tgt_kpm = (ys == model.tgt_vocab.pad_id)
        T = ys.size(1)
        tgt_causal = torch.triu(torch.ones((T, T), device=device), diagonal=1).bool()

        dec = model.tr.decoder(
            tgt, mem,
            tgt_mask=tgt_causal,
            tgt_key_padding_mask=tgt_kpm,
            memory_key_padding_mask=src_kpm,
        )
        logits = model.out(dec)
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        ys = torch.cat([ys, next_tok], dim=1)
        if torch.all(next_tok.squeeze(1) == model.tgt_vocab.eos_id):
            break

    return ys

def train_one_epoch_nograph(model: TransformerNoGraph, loader, optimizer, tgt_vocab: Vocab, grad_clip=1.0):
    model.train()
    meter = AvgMeter()
    for src_pad, src_lens, tgt_pad, _, _, _, _ in loader:
        tgt_inp = tgt_pad[:, :-1]
        tgt_out = tgt_pad[:, 1:]
        logits = model(src_pad, src_lens, tgt_inp)
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
def eval_loss_nograph(model: TransformerNoGraph, loader, tgt_vocab: Vocab):
    model.eval()
    meter = AvgMeter()
    for src_pad, src_lens, tgt_pad, _, _, _, _ in loader:
        tgt_inp = tgt_pad[:, :-1]
        tgt_out = tgt_pad[:, 1:]
        logits = model(src_pad, src_lens, tgt_inp)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=tgt_vocab.pad_id
        )
        meter.update(loss.item(), n=tgt_pad.size(0))
    return meter.avg
