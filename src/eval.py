"""Retrieval metrics. Text->image: each COCO caption is a query, its source image the single relevant item."""
import json
from pathlib import Path

import numpy as np

from src import index as idx
from src.embed import Embedder

CAPTIONS = idx.DATA_DIR / "coco" / "captions_val2017.json"
KS = (1, 5, 10)


def load_queries(files: list[str]):
    ann = json.loads(CAPTIONS.read_text())
    file_of = {im["id"]: im["file_name"] for im in ann["images"]}
    pos = {f: i for i, f in enumerate(files)}
    texts = [a["caption"].strip() for a in ann["annotations"]]
    gold = np.array([pos[file_of[a["image_id"]]] for a in ann["annotations"]])
    return texts, gold


def text_to_image(embedder: Embedder) -> dict:
    index, files = idx.load("coco", embedder.tag)
    texts, gold = load_queries(files)
    q = embedder.encode_texts(texts)
    _, ids = index.search(q, max(KS))
    hits = ids == gold[:, None]
    # rank of the gold image (1-based); inf if outside top max(KS)
    first = np.where(hits.any(1), hits.argmax(1) + 1, np.inf)
    res = {f"recall@{k}": float((first <= k).mean()) for k in KS}
    res["precision@10"] = float(hits.sum(1).mean() / 10)  # at most 1 relevant -> bounded by 0.1
    res["mrr@10"] = float(np.where(np.isfinite(first), 1 / first, 0).mean())
    res["n_queries"] = len(texts)
    return res


def pet_breed(filename: str) -> str:
    return filename.rsplit("_", 1)[0].lower()


def image_to_image_pets(embedder: Embedder, ks=(1, 5, 10)) -> dict:
    """Leave-one-out: every image queries the rest; relevant = same breed."""
    index, files = idx.load("pets", embedder.tag)
    vecs = idx.load_embeddings("pets", embedder.tag)
    labels = np.array([pet_breed(f) for f in files])
    n_rel_total = np.array([(labels == l).sum() - 1 for l in labels])  # excluding the query itself
    _, ids = index.search(vecs, max(ks) + 1)
    # drop the query itself (should be rank 1; filter by id to be safe against ties)
    ranked = np.array([[j for j in row if j != i][: max(ks)] for i, row in enumerate(ids)])
    rel = labels[ranked] == labels[:, None]
    res = {}
    for k in ks:
        hits = rel[:, :k].sum(1)
        res[f"precision@{k}"] = float((hits / k).mean())
        res[f"recall@{k}"] = float((hits / n_rel_total).mean())
    res["n_queries"] = len(files)
    return res


def pets_map_and_confusions(embedder: Embedder, top_pairs: int = 10) -> dict:
    """mAP over the full ranking, plus per-breed top-1 accuracy and most-confused breed pairs."""
    vecs = idx.load_embeddings("pets", embedder.tag)
    files = idx.load("pets", embedder.tag)[1]
    names, labels = np.unique([pet_breed(f) for f in files], return_inverse=True)
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -np.inf)  # exclude the query itself
    order = np.argsort(-sims, axis=1)[:, :-1]
    rel = labels[order] == labels[:, None]
    cum = rel.cumsum(1)
    ranks = np.arange(1, rel.shape[1] + 1)
    ap = ((cum / ranks) * rel).sum(1) / rel.sum(1)
    top1_wrong = ~rel[:, 0]
    per_breed = {names[b]: float(rel[labels == b, 0].mean()) for b in range(len(names))}
    pairs = {}
    for i in np.where(top1_wrong)[0]:
        key = (names[labels[i]], names[labels[order[i, 0]]])
        pairs[key] = pairs.get(key, 0) + 1
    worst = sorted(pairs.items(), key=lambda kv: -kv[1])[:top_pairs]
    return {
        "mAP": float(ap.mean()),
        "worst_breeds_top1": dict(sorted(per_breed.items(), key=lambda kv: kv[1])[:5]),
        "most_confused": [f"{a} -> {b}: {n}" for (a, b), n in worst],
    }


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from src.embed import DEFAULT_MODEL

    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL[0])
    p.add_argument("--pretrained", default=DEFAULT_MODEL[1])
    a = p.parse_args()
    emb = Embedder(a.model, a.pretrained)
    out = {
        "model": emb.tag,
        "text2img_coco": text_to_image(emb),
        "img2img_pets": {**image_to_image_pets(emb), **pets_map_and_confusions(emb)},
    }
    results = idx.ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / f"{emb.tag}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
