"""Build docs/report.html (self-contained) from scripts/report_template.html.

Fills {{placeholders}} from results/*.json, inlines scripts/report.css, and embeds docs/images/* as base64.
Run:  uv run python -m scripts.build_report
"""
import base64
import json
import re
from pathlib import Path

from src import index as idx

ROOT = idx.ROOT
RES = ROOT / "results"
SCRIPTS = ROOT / "scripts"
DOCS = ROOT / "docs"

MODELS = [  # (tag, display name, family) -- same order as the figures: by parameter count
    ("ViT-B-32__laion2b_s34b_b79k", "ViT-B/32", "LAION-2B"),
    ("ViT-B-16-SigLIP__webli", "ViT-B/16 SigLIP", "WebLI"),
    ("ViT-L-14__laion2b_s32b_b82k", "ViT-L/14", "LAION-2B"),
    ("ViT-L-16-SigLIP-256__webli", "ViT-L/16-256 SigLIP", "WebLI"),
    ("ViT-SO400M-14-SigLIP__webli", "ViT-SO400M/14 SigLIP", "WebLI"),
]
R = {t: json.loads((RES / f"{t}.json").read_text()) for t, _, _ in MODELS}
LAT = json.loads((RES / "latency.json").read_text())
ST = json.loads((RES / "analysis_stats.json").read_text())

BASE, B16, L14, BEST, SO = (m[0] for m in (MODELS[0], MODELS[1], MODELS[2], MODELS[3], MODELS[4]))
pct = lambda x: f"{x * 100:.1f}%"
f3 = lambda x: f"{x:.3f}"

values = {
    "n_queries": f"{ST['n_queries']:,}",
    "coco_r1_base": pct(R[BASE]["text2img_coco"]["recall@1"]),
    "coco_r1_best": pct(R[BEST]["text2img_coco"]["recall@1"]),
    "coco_r10_best": pct(R[BEST]["text2img_coco"]["recall@10"]),
    "coco_mrr_best": f3(R[BEST]["text2img_coco"]["mrr@10"]),
    "pets_map_base": f3(R[BASE]["img2img_pets"]["mAP"]),
    "pets_map_best": f3(R[BEST]["img2img_pets"]["mAP"]),
    "pets_p1_best": pct(R[BEST]["img2img_pets"]["precision@1"]),
    "pets_p1_base": pct(R[BASE]["img2img_pets"]["precision@1"]),
    "b16_r1": pct(R[B16]["text2img_coco"]["recall@1"]),
    "l14_r1": pct(R[L14]["text2img_coco"]["recall@1"]),
    "so_r1": pct(R[SO]["text2img_coco"]["recall@1"]),
    "b16_map": f3(R[B16]["img2img_pets"]["mAP"]),
    "l14_map": f3(R[L14]["img2img_pets"]["mAP"]),
    "b16_params": f"{LAT[B16]['params_m']}M",
    "l14_params": f"{LAT[L14]['params_m']}M",
    "best_params": f"{LAT[BEST]['params_m']}M",
    "so_params": f"{LAT[SO]['params_m']}M",
    "b16_lat": f"{LAT[B16]['image_query_ms']:.0f} ms",
    "l14_lat": f"{LAT[L14]['image_query_ms']:.0f} ms",
    "best_lat": f"{LAT[BEST]['image_query_ms']:.0f} ms",
    "so_lat": f"{LAT[SO]['image_query_ms']:.0f} ms",
    "top1_hit": pct(ST["top1_hit_rate"]),
    "n_miss": f"{ST['n_top1_misses']:,}",
    "miss_cat": pct(ST["miss_top1_shares_a_category"]),
    "rand_cat": pct(ST["random_pair_shares_a_category"]),
    "miss_jac": f"{ST['miss_top1_mean_jaccard']:.2f}",
    "rand_jac": f"{ST['random_pair_mean_jaccard']:.2f}",
}


def table_main() -> str:
    head = "<tr><th>Model</th><th>Weights</th><th>COCO R@1</th><th>R@5</th><th>R@10</th><th>MRR@10</th><th>Pets P@1</th><th>P@10</th><th>mAP</th></tr>"
    rows = []
    for tag, name, fam in MODELS:
        t, i = R[tag]["text2img_coco"], R[tag]["img2img_pets"]
        cells = [t["recall@1"], t["recall@5"], t["recall@10"], t["mrr@10"], i["precision@1"], i["precision@10"], i["mAP"]]
        rows.append(f"<tr><td>{name}</td><td>{fam}</td>" + "".join(f"<td>{c:.3f}</td>" for c in cells) + "</tr>")
    return f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def table_latency() -> str:
    head = "<tr><th>Model</th><th>Parameters</th><th>Text query</th><th>Image query</th></tr>"
    rows = [
        f"<tr><td>{name}</td><td>{LAT[tag]['params_m']}M</td><td>{LAT[tag]['text_query_ms']:.0f} ms</td><td>{LAT[tag]['image_query_ms']:.0f} ms</td></tr>"
        for tag, name, _ in MODELS
    ]
    return f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


values["TABLE_MAIN"] = table_main()
values["TABLE_LATENCY"] = table_latency()
values["CSS"] = (SCRIPTS / "report.css").read_text(encoding="utf8")

html = (SCRIPTS / "report_template.html").read_text(encoding="utf8")
missing = set(re.findall(r"\{\{(\w+)\}\}", html)) - set(values)
assert not missing, f"template placeholders without a value: {missing}"
html = re.sub(r"\{\{(\w+)\}\}", lambda m: values[m.group(1)], html)


def embed(m):
    path = DOCS / "images" / m.group(1)
    mime = "image/png" if path.suffix == ".png" else "image/jpeg"
    return f'src="data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"'


html = re.sub(r'src="images/([^"]+)"', embed, html)
(DOCS / "report.html").write_text(html, encoding="utf8")
print(f"wrote docs/report.html ({len(html) / 1024:.0f} KB)")
