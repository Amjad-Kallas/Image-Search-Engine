"""Build / load a FAISS index (exact cosine search via inner product on normalised vectors)."""
import json
from pathlib import Path

import faiss
import numpy as np

from src.embed import Embedder

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"

IMAGE_DIRS = {
    "coco": DATA_DIR / "coco" / "val2017",
    "pets": DATA_DIR / "pets" / "jpg",
}


def _dir(dataset: str, tag: str) -> Path:
    return INDEX_DIR / dataset / tag


def build(embedder: Embedder, dataset: str) -> Path:
    paths = sorted(IMAGE_DIRS[dataset].glob("*.jpg"))
    vecs = embedder.encode_images(paths)
    out = _dir(dataset, embedder.tag)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", vecs)
    (out / "files.json").write_text(json.dumps([p.name for p in paths]))
    faiss.write_index(_make_index(vecs), str(out / "index.faiss"))
    return out


def _make_index(vecs: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index


def load(dataset: str, tag: str):
    d = _dir(dataset, tag)
    return faiss.read_index(str(d / "index.faiss")), json.loads((d / "files.json").read_text())


def load_embeddings(dataset: str, tag: str) -> np.ndarray:
    return np.load(_dir(dataset, tag) / "embeddings.npy")
