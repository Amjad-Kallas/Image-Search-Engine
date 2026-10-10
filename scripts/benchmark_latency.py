"""Parameter count and single-query latency (encode + FAISS search) per model, on the COCO index.

Run:  uv run python -m scripts.benchmark_latency
"""
import json
import statistics
import time
from pathlib import Path

import torch
from PIL import Image

from src import index as idx
from src.embed import Embedder

MODELS = [
    ("ViT-B-32", "laion2b_s34b_b79k"),
    ("ViT-L-14", "laion2b_s32b_b82k"),
    ("ViT-B-16-SigLIP", "webli"),
    ("ViT-SO400M-14-SigLIP", "webli"),
    ("ViT-L-16-SigLIP-256", "webli"),
]
N, WARMUP = 30, 3
img = Image.open(idx.IMAGE_DIRS["pets"] / "Birman_10.jpg")
out = {}
for name, pre in MODELS:
    emb = Embedder(name, pre)
    index, _ = idx.load("coco", emb.tag)
    params = sum(p.numel() for p in emb.model.parameters())

    def timed(fn):
        for _ in range(WARMUP):
            fn()
        ts = []
        for _ in range(N):
            if emb.device == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            fn()
            if emb.device == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1000)
        return statistics.median(ts)

    text_ms = timed(lambda: index.search(emb.encode_texts(["a red double decker bus"]), 10))
    image_ms = timed(lambda: index.search(emb.encode_pil([img]), 10))
    out[emb.tag] = {"params_m": round(params / 1e6), "text_query_ms": round(text_ms, 1), "image_query_ms": round(image_ms, 1)}
    print(emb.tag, out[emb.tag], flush=True)
    del emb
    torch.cuda.empty_cache()
(Path(idx.ROOT) / "results" / "latency.json").write_text(json.dumps(out, indent=2))
