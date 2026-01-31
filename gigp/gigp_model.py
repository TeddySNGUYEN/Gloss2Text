import math
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import Vocab
from .graph import Graph, build_gloss_graph
from .utils import AvgMeter

class GraphEncoder(nn.Module):
    def __init__(self, node_vocab: Vocab, role_vocab: Vocab, rel_vocab: Vocab, d_model: int, cache_size=200000):
        super().__init__()
        self.node_vocab = node_vocab
        self.role_vocab = role_vocab
        self.rel_vocab = rel_vocab
        self.d_model = d_model

        self.node_emb = nn.Embedding(len(node_vocab), d_model, padding_idx=node_vocab.pad_id)
        self.role_emb = nn.Embedding(len(role_vocab), d_model, padding_idx=role_vocab.pad_id)

        self.rel_linears = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(len(rel_vocab))])

        self.phi = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.ln = nn.LayerNorm(d_model)
        self.pool_score = nn.Linear(d_model, 1)

        self._cache: Dict[str, Graph] = {}
        self.cache_size = cache_size

    def _get_graph(self, gloss_tok: str) -> Graph:
        g = self._cache.get(gloss_tok)
        if g is not None:
            return g
        g = build_gloss_graph(gloss_tok)
        if len(self._cache) < self.cache_size:
            self._cache[gloss_tok] = g
        return g

    def forward(self, raw_gloss_tokens: List[List[str]], device: torch.device):
        B = len(raw_gloss_tokens)
        lens = torch.tensor([len(x) for x in raw_gloss_tokens], dtype=torch.long, device=device)
        Smax = int(lens.max().item())
        Z = torch.zeros((B, Smax, self.d_model), device=device)

        for b in range(B):
            for s, gloss_tok in enumerate(raw_gloss_tokens[b]):
                G = self._get_graph(gloss_tok)

                node_ids = torch.tensor([self.node_vocab.stoi.get(t, self.node_vocab.unk_id) for t in G.node_tokens],
                                        dtype=torch.long, device=device)
                role_ids = torch.tensor([self.role_vocab.stoi.get(r, self.role_vocab.unk_id) for r in G.node_roles],
                                        dtype=torch.long, device=device)

                h0 = self.node_emb(node_ids) + self.role_emb(role_ids)
                N = h0.size(0)
                agg = torch.zeros((N, self.d_model), device=device)

                if len(G.edges) > 0:
                    src_idx = torch.tensor([e[0] for e in G.edges], dtype=torch.long, device=device)
                    dst_idx = torch.tensor([e[1] for e in G.edges], dtype=torch.long, device=device)
                    rel_ids = torch.tensor([self.rel_vocab.stoi.get(e[2], self.rel_vocab.unk_id) for e in G.edges],
                                           dtype=torch.long, device=device)

                    msgs = []
                    for k in range(src_idx.size(0)):
                        rid = int(rel_ids[k].item())
                        msgs.append(self.rel_linears[rid](h0[src_idx[k]]))
                    msgs = torch.stack(msgs, dim=0)
                    agg.index_add_(0, dst_idx, msgs)

                h1 = self.phi(torch.cat([h0, agg], dim=-1))
                h = self.ln(h0 + h1)

                w = self.pool_score(h).squeeze(-1)
                a = torch.softmax(w, dim=0).unsqueeze(-1)
                Z[b, s] = torch.sum(a * h, dim=0)

        return Z, lens

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

class GIGPTransformer(nn.Module):
    def __init__(self, node_vocab, role_vocab, rel_vocab, tgt_vocab,
                 d_model=512, nhead=8, enc_layers=4, dec_layers=4,
                 ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.tgt_vocab = tgt_vocab
        self.graph_enc = GraphEncoder(node_vocab, role_vocab, rel_vocab, d_model)
        self.src_pos = PositionalEncoding(d_model, dropout)
        self.tgt_emb = nn.Embedding(len(tgt_vocab), d_model, padding_idx=tgt_vocab.pad_id)
        self.tgt_pos = PositionalEncoding(d_model, dropout)

        self.tr = nn.Transformer(
            d_model=d_model, nhead=nhead,
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

    def encode(self, raw_gloss_tokens, device):
        Z, src_lens = self.graph_enc(raw_gloss_tokens, device=device)
        Z = self.src_pos(Z)
        src_kpm = self.lengths_to_padmask(src_lens, Z.size(1))
        mem = self.tr.encoder(Z, src_key_padding_mask=src_kpm)
        return mem, src_kpm

    def forward(self, raw_gloss_tokens, tgt_inp_ids):
        device = tgt_inp_ids.device
        mem, src_kpm = self.encode(raw_gloss_tokens, device=device)

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
def greedy_decode_gigp(model: GIGPTransformer, raw_gloss_tokens, device, max_len=64):
    model.eval()
    mem, src_kpm = model.encode(raw_gloss_tokens, device=device)

    B = len(raw_gloss_tokens)
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

def train_one_epoch_gigp(model, loader, optimizer, tgt_vocab: Vocab, grad_clip=1.0):
    model.train()
    meter = AvgMeter()
    for _, _, tgt_pad, _, raw_gloss_tokens, _, _ in loader:
        tgt_inp = tgt_pad[:, :-1]
        tgt_out = tgt_pad[:, 1:]
        logits = model(raw_gloss_tokens, tgt_inp)
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
def eval_loss_gigp(model, loader, tgt_vocab: Vocab):
    model.eval()
    meter = AvgMeter()
    for _, _, tgt_pad, _, raw_gloss_tokens, _, _ in loader:
        tgt_inp = tgt_pad[:, :-1]
        tgt_out = tgt_pad[:, 1:]
        logits = model(raw_gloss_tokens, tgt_inp)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=tgt_vocab.pad_id
        )
        meter.update(loss.item(), n=tgt_pad.size(0))
    return meter.avg
