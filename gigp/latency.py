import time
import numpy as np
import torch

from .utils import cuda_sync
from .rule_based import rule_based_translate
from .linear_baseline import greedy_decode_linear
from .transformer_no_graph import greedy_decode_nograph
from .gigp_model import greedy_decode_gigp

def _time_it(fn, device: torch.device, warmup=30, runs=200):
    for _ in range(warmup):
        fn()
    cuda_sync(device)

    ms = []
    for _ in range(runs):
        cuda_sync(device)
        t0 = time.perf_counter()
        fn()
        cuda_sync(device)
        t1 = time.perf_counter()
        ms.append((t1 - t0) * 1000.0)
    return np.array(ms, dtype=np.float64)

def _summarize(ms: np.ndarray):
    return {
        "mean_ms": float(ms.mean()),
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
    }

def benchmark_all(
    device: torch.device,
    test_ds,
    test_loader,
    linear_model=None,
    nograph_model=None,
    gigp_model=None,
    max_len=64,
    warmup=30,
    runs=200,
):
    results = {}

    # Rule-based (CPU) - use same batch size as loader when possible
    B0 = int(next(iter(test_loader))[0].size(0))
    B0 = min(B0, len(test_ds))
    idxs = np.random.choice(len(test_ds), size=B0, replace=False)

    def rb_fn():
        for i in idxs:
            gloss_tokens, _, _, _ = test_ds[i]
            _ = rule_based_translate(gloss_tokens)

    ms = _time_it(rb_fn, device=torch.device("cpu"), warmup=warmup, runs=runs)
    s = _summarize(ms)
    s["batch_size"] = B0
    s["samples_per_sec"] = float((B0 * 1000.0) / s["mean_ms"])
    results["Rule-based (CPU)"] = s

    fixed = next(iter(test_loader))
    src_pad, src_lens, _, _, raw_gloss_tokens, _, _ = fixed
    B = int(src_pad.size(0))

    if linear_model is not None:
        linear_model.eval()
        @torch.inference_mode()
        def lin_fn():
            _ = greedy_decode_linear(linear_model, src_pad, max_len=max_len)
        ms = _time_it(lin_fn, device=device, warmup=warmup, runs=runs)
        s = _summarize(ms)
        s["batch_size"] = B
        s["samples_per_sec"] = float((B * 1000.0) / s["mean_ms"])
        results["Linear-BoW (GPU)"] = s

    if nograph_model is not None:
        nograph_model.eval()
        @torch.inference_mode()
        def ng_fn():
            _ = greedy_decode_nograph(nograph_model, src_pad, src_lens, max_len=max_len)
        ms = _time_it(ng_fn, device=device, warmup=warmup, runs=runs)
        s = _summarize(ms)
        s["batch_size"] = B
        s["samples_per_sec"] = float((B * 1000.0) / s["mean_ms"])
        results["Transformer (no-graph)"] = s

    if gigp_model is not None:
        gigp_model.eval()
        @torch.inference_mode()
        def gigp_fn():
            _ = greedy_decode_gigp(gigp_model, raw_gloss_tokens, device=device, max_len=max_len)
        ms = _time_it(gigp_fn, device=device, warmup=warmup, runs=runs)
        s = _summarize(ms)
        s["batch_size"] = B
        s["samples_per_sec"] = float((B * 1000.0) / s["mean_ms"])
        results["GIGP (GPU)"] = s

    return results

def print_latency(results: dict):
    print("\n=== End-to-End Latency (decode-to-EOS) ===")
    for name, r in results.items():
        B = r["batch_size"]
        mean_s = r["mean_ms"] / B
        p95_s = r["p95_ms"] / B
        print(
            f"{name:22s} | "
            f"mean={r['mean_ms']:.2f} ms/batch ({mean_s:.3f} ms/sample) | "
            f"p95={r['p95_ms']:.2f} ms/batch ({p95_s:.3f} ms/sample) | "
            f"throughput={r['samples_per_sec']:.1f} samples/s | B={B}"
        )
