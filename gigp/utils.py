import random
import numpy as np
import torch

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(use_cuda: bool = True) -> torch.device:
    return torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

def cuda_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

class AvgMeter:
    def __init__(self):
        self.sum = 0.0
        self.cnt = 0
    def update(self, v, n=1):
        self.sum += float(v) * n
        self.cnt += n
    @property
    def avg(self):
        return self.sum / max(1, self.cnt)
