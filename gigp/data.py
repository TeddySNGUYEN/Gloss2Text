import os, csv
from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .vocab import Vocab

def split_full_csv(
    data_dir: str,
    full_csv: str = "train.csv",
    train_out: str = "train.csv",
    dev_out: str = "dev.csv",
    test_out: str = "test.csv",
    train_ratio=0.8,
    dev_ratio=0.1,
    test_ratio=0.1,
    seed=42,
):
    full_path = os.path.join(data_dir, full_csv)
    assert os.path.exists(full_path), f"Missing: {full_path}"
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6

    df = pd.read_csv(full_path)
    if "gloss" not in df.columns:
        raise ValueError(f"CSV must contain 'gloss'. Columns: {list(df.columns)}")
    if "text" not in df.columns:
        raise ValueError(f"CSV must contain 'text'. Columns: {list(df.columns)}")

    df = df[["gloss", "text"]].dropna()
    df["gloss"] = df["gloss"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[(df["gloss"] != "") & (df["text"] != "")]

    train_df, temp_df = train_test_split(df, test_size=1.0-train_ratio, random_state=seed, shuffle=True)
    test_portion = test_ratio / (dev_ratio + test_ratio)
    dev_df, test_df = train_test_split(temp_df, test_size=test_portion, random_state=seed, shuffle=True)

    train_df.to_csv(os.path.join(data_dir, train_out), index=False)
    dev_df.to_csv(os.path.join(data_dir, dev_out), index=False)
    test_df.to_csv(os.path.join(data_dir, test_out), index=False)

class CSVGlossTextDataset(Dataset):
    def __init__(self, csv_path: str):
        assert os.path.exists(csv_path), f"Missing: {csv_path}"
        self.rows: List[Tuple[str, str]] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"No header in CSV: {csv_path}")
            fields = set(reader.fieldnames)
            if "gloss" not in fields or "text" not in fields:
                raise ValueError(f"CSV must contain columns gloss,text. Got: {reader.fieldnames}")
            for row in reader:
                g = (row.get("gloss") or "").strip()
                t = (row.get("text") or "").strip()
                if g and t:
                    self.rows.append((g, t))
        if not self.rows:
            raise ValueError(f"No valid rows in {csv_path}")

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        gloss, text = self.rows[idx]
        return gloss.split(), text.split(), gloss, text

def build_vocabs(train_ds: Dataset, min_freq=1):
    src_vocab = Vocab(min_freq=min_freq)
    tgt_vocab = Vocab(min_freq=min_freq)
    for gloss_tokens, tgt_tokens, _, _ in train_ds:
        src_vocab.add_sentence(gloss_tokens)
        tgt_vocab.add_sentence(tgt_tokens)
    src_vocab.build()
    tgt_vocab.build()
    return src_vocab, tgt_vocab

def collate_batch(batch, src_vocab: Vocab, tgt_vocab: Vocab, device, max_src_len=128, max_tgt_len=128):
    raw_gloss_tokens = []
    src_texts, tgt_texts = [], []
    src_ids_list, tgt_ids_list = [], []

    for gloss_tokens, tgt_tokens, gloss_str, tgt_str in batch:
        gloss_tokens = gloss_tokens[:max_src_len]
        tgt_tokens = tgt_tokens[:max_tgt_len]

        raw_gloss_tokens.append(gloss_tokens)
        src_texts.append(gloss_str)
        tgt_texts.append(tgt_str)

        src_ids = src_vocab.encode(gloss_tokens)
        tgt_ids = [tgt_vocab.bos_id] + tgt_vocab.encode(tgt_tokens) + [tgt_vocab.eos_id]

        src_ids_list.append(torch.tensor(src_ids, dtype=torch.long))
        tgt_ids_list.append(torch.tensor(tgt_ids, dtype=torch.long))

    src_pad = nn.utils.rnn.pad_sequence(src_ids_list, batch_first=True, padding_value=src_vocab.pad_id).to(device)
    tgt_pad = nn.utils.rnn.pad_sequence(tgt_ids_list, batch_first=True, padding_value=tgt_vocab.pad_id).to(device)

    src_lens = torch.tensor([len(x) for x in src_ids_list], dtype=torch.long, device=device)
    tgt_lens = torch.tensor([len(x) for x in tgt_ids_list], dtype=torch.long, device=device)

    return src_pad, src_lens, tgt_pad, tgt_lens, raw_gloss_tokens, src_texts, tgt_texts

def make_loader(ds, src_vocab, tgt_vocab, device, batch_size=16, shuffle=False, max_src_len=128, max_tgt_len=128):
    def _collate(batch):
        return collate_batch(batch, src_vocab, tgt_vocab, device, max_src_len, max_tgt_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate, num_workers=0)
