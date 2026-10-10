"""Generate report figures + analysis stats from saved results and indices.

Run:  uv run --with matplotlib python -m scripts.make_figures
Writes PNGs to docs/images/ and results/analysis_stats.json.
"""
import json
import random
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src import index as idx
from src.embed import Embedder
from src.eval import CAPTIONS, load_queries, pet_breed

ROOT = idx.ROOT
OUT = ROOT / "docs" / "images"
RES = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

BEST = ("ViT-L-16-SigLIP-256", "webli")
# fixed order = by parameter count; colour = model family
MODELS = [
    ("ViT-B-32__laion2b_s34b_b79k", "ViT-B/32", "LAION"),
    ("ViT-B-16-SigLIP__webli", "ViT-B/16", "SigLIP"),
    ("ViT-L-14__laion2b_s32b_b82k", "ViT-L/14", "LAION"),
    ("ViT-L-16-SigLIP-256__webli", "ViT-L/16-256", "SigLIP"),
    ("ViT-SO400M-14-SigLIP__webli", "ViT-SO400M/14", "SigLIP"),
]
FAMILY_COLOR = {"LAION": "#2a78d6", "SigLIP": "#eb6834"}  # validated pair (light surface)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.edgecolor": GRID,
    }
)

results = {tag: json.loads((RES / f"{tag}.json").read_text()) for tag, _, _ in MODELS}
latency = json.loads((RES / "latency.json").read_text())


# ---------- charts ----------
def fig_model_comparison():
    panels = [
        ("COCO text → image\nrecall@1", lambda r: r["text2img_coco"]["recall@1"]),
        ("Pets image → image\nprecision@1", lambda r: r["img2img_pets"]["precision@1"]),
        ("Pets image → image\nmAP", lambda r: r["img2img_pets"]["mAP"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    ys = np.arange(len(MODELS))[::-1]
    for ax, (title, getter) in zip(axes, panels):
        vals = [getter(results[t]) for t, _, _ in MODELS]
        ax.barh(ys, vals, height=0.5, color=[FAMILY_COLOR[f] for _, _, f in MODELS])
        for y, v in zip(ys, vals):
            ax.text(v + 0.012, y, f"{v:.3f}", va="center", color=INK, fontsize=10.5)
        ax.set_xlim(0, 1.0)
        ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
        ax.set_yticks(ys, [n for _, n, _ in MODELS])
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLOR.values()]
    fig.legend(handles, ["LAION-2B CLIP", "SigLIP (WebLI)"], loc="upper right", frameon=False, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig-model-comparison.png", dpi=150)
    plt.close(fig)


def fig_cost_quality():
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    offsets = {"ViT-B/32": (8, -14), "ViT-B/16": (8, 6), "ViT-L/14": (-10, -18), "ViT-L/16-256": (10, 6), "ViT-SO400M/14": (-12, -20)}
    for tag, name, fam in MODELS:
        x = latency[tag]["image_query_ms"]
        y = results[tag]["text2img_coco"]["recall@1"]
        ax.scatter(x, y, s=130, color=FAMILY_COLOR[fam], edgecolor=SURFACE, linewidth=2, zorder=3)
        dx, dy = offsets[name]
        ax.annotate(
            f"{name}\n{latency[tag]['params_m']}M params",
            (x, y), xytext=(dx, dy), textcoords="offset points",
            ha="left" if dx > 0 else "right", fontsize=10, color=INK,
        )
    ax.set_xlabel("Image-query latency (ms, median, GTX 1060)  →  slower")
    ax.set_ylabel("COCO text → image recall@1  →  better")
    ax.set_xlim(0, 140)
    ax.set_ylim(0.36, 0.54)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=9) for c in FAMILY_COLOR.values()]
    ax.legend(handles, ["LAION-2B CLIP", "SigLIP (WebLI)"], frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig-cost-quality.png", dpi=150)
    plt.close(fig)


def fig_pets_confusions(top_n=10):
    rows = results["ViT-L-16-SigLIP-256__webli"]["img2img_pets"]["most_confused"][:top_n]
    parsed = []
    for r in rows:
        pair, n = r.rsplit(": ", 1)
        a, b = pair.split(" -> ")
        parsed.append((f"{pretty(a)} → {pretty(b)}", int(n)))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ys = np.arange(len(parsed))[::-1]
    ax.barh(ys, [n for _, n in parsed], height=0.55, color=FAMILY_COLOR["LAION"])
    for y, (_, n) in zip(ys, parsed):
        ax.text(n + 0.4, y, str(n), va="center", fontsize=10.5)
    ax.set_yticks(ys, [p for p, _ in parsed])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("queries whose top-1 result was the other breed")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig-pets-confusions.png", dpi=150)
    plt.close(fig)


# ---------- image grids ----------
def pretty(breed: str) -> str:
    return breed.replace("_", " ").title()


def square(path: Path, size=240) -> Image.Image:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return im.crop((left, top, left + s, top + s)).resize((size, size))


def tile(ax, path, title, title_color=INK, bold=False):
    ax.imshow(square(path))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=10, color=title_color, pad=4, fontweight="bold" if bold else "normal")


def coco_rows(rows, fname, coco_dir):
    n = len(rows)
    fig, axes = plt.subplots(n, 6, figsize=(14, 2.55 * n), gridspec_kw={"width_ratios": [1.5, 1, 1, 1, 1, 1]})
    axes = np.atleast_2d(axes)
    for r, row in enumerate(rows):
        axes[r, 0].axis("off")
        axes[r, 0].text(0, 0.5, "“" + textwrap.fill(row["caption"], 30) + "”", fontsize=11.5, va="center", color=INK)
        tile(axes[r, 1], coco_dir / row["gold_file"], f"expected image\n(ranked #{row['gold_rank']})", "#2a78d6", True)
        for c, (f, s) in enumerate(row["top"][:4]):
            ok = f == row["gold_file"]
            tile(axes[r, c + 2], coco_dir / f, f"#{c + 1} · {s:.2f}" + ("  ✓ match" if ok else ""), INK, ok)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=120, pil_kwargs={"quality": 85})
    plt.close(fig)


def pets_misses(pairs, vecs, files, labels, names, fname):
    fig, axes = plt.subplots(len(pairs), 5, figsize=(11.5, 2.5 * len(pairs)))
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -np.inf)
    for r, (a, b) in enumerate(pairs):
        cand = [i for i in np.where(names[labels] == a)[0] if names[labels[sims[i].argmax()]] == b]
        q = cand[0]
        top = np.argsort(-sims[q])[:4]
        tile(axes[r, 0], idx.IMAGE_DIRS["pets"] / files[q], "query:\n" + textwrap.fill(pretty(a), 20), "#2a78d6", True)
        for c, j in enumerate(top):
            ok = labels[j] == labels[q]
            tile(axes[r, c + 1], idx.IMAGE_DIRS["pets"] / files[j], textwrap.fill(pretty(names[labels[j]]), 20) + ("  ✓" if ok else "  ✗"), INK, not ok)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=120, pil_kwargs={"quality": 85})
    plt.close(fig)


# ---------- analysis ----------
def main():
    fig_model_comparison()
    fig_cost_quality()
    fig_pets_confusions()

    emb = Embedder(*BEST)
    index, files = idx.load("coco", emb.tag)
    coco_vecs = idx.load_embeddings("coco", emb.tag)
    texts, gold = load_queries(files)
    q = emb.encode_texts(texts)
    _, ids = index.search(q, 10)
    top1 = ids[:, 0]
    hit = top1 == gold

    # how wrong are the rank-1 misses? compare COCO object categories of top-1 vs gold image
    inst = json.loads((CAPTIONS.parent / "instances_val2017.json").read_text())
    fname_of = {im["id"]: im["file_name"] for im in inst["images"]}
    cats = {f: set() for f in files}
    for a in inst["annotations"]:
        cats[fname_of[a["image_id"]]].add(a["category_id"])
    cat_sets = [cats[f] for f in files]

    def share(i, j):
        a, b = cat_sets[i], cat_sets[j]
        return (len(a & b) > 0), (len(a & b) / len(a | b) if a | b else 0.0)

    miss = np.where(~hit)[0]
    m = np.array([share(gold[i], top1[i]) for i in miss])
    rng = np.random.default_rng(0)
    rp = rng.integers(0, len(files), size=(100000, 2))
    rp = rp[rp[:, 0] != rp[:, 1]]
    base = np.array([share(i, j) for i, j in rp])
    stats = {
        "model": emb.tag,
        "n_queries": int(len(texts)),
        "top1_hit_rate": float(hit.mean()),
        "n_top1_misses": int(len(miss)),
        "miss_top1_shares_a_category": float(m[:, 0].mean()),
        "miss_top1_mean_jaccard": float(m[:, 1].mean()),
        "random_pair_shares_a_category": float(base[:, 0].mean()),
        "random_pair_mean_jaccard": float(base[:, 1].mean()),
    }
    (RES / "analysis_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    # example rows
    def gold_rank(i):
        s = coco_vecs @ q[i]
        return int((s > s[gold[i]]).sum()) + 1

    def make_row(i):
        s = coco_vecs @ q[i]
        order = np.argsort(-s)[:4]
        return {
            "caption": texts[i],
            "gold_file": files[gold[i]],
            "gold_rank": gold_rank(i),
            "top": [(files[j], float(s[j])) for j in order],
        }

    rnd = random.Random(7)
    hits_i = rnd.sample([i for i in np.where(hit)[0] if 25 < len(texts[i]) < 70], 3)
    miss_i = rnd.sample([i for i in np.where(gold_rank_vec(q, coco_vecs, gold) > 10)[0] if 25 < len(texts[i]) < 70], 4)
    coco_dir = idx.IMAGE_DIRS["coco"]
    coco_rows([make_row(i) for i in hits_i], "fig-coco-hits.jpg", coco_dir)
    coco_rows([make_row(i) for i in miss_i], "fig-coco-misses.jpg", coco_dir)

    # Pets failures (best model, leave-one-out on stored embeddings)
    pf = idx.load("pets", emb.tag)[1]
    pvecs = idx.load_embeddings("pets", emb.tag)
    names, labels = np.unique([pet_breed(f) for f in pf], return_inverse=True)
    pets_misses(
        [("ragdoll", "birman"), ("beagle", "basset_hound"), ("american_pit_bull_terrier", "staffordshire_bull_terrier"), ("british_shorthair", "russian_blue")],
        pvecs, pf, labels, names, "fig-pets-misses.jpg",
    )


def gold_rank_vec(q, vecs, gold):
    """Rank of the gold image for every query (1 = best)."""
    ranks = np.empty(len(q), dtype=np.int32)
    for s in range(0, len(q), 2000):
        sims = q[s : s + 2000] @ vecs.T
        g = sims[np.arange(len(sims)), gold[s : s + 2000]]
        ranks[s : s + 2000] = (sims > g[:, None]).sum(1) + 1
    return ranks


if __name__ == "__main__":
    main()
