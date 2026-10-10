# Image Search Engine

Search an image collection by **text description** or by **uploading an image**, using CLIP-style embeddings
(text and images live in one shared vector space, so both query types hit the same index).
The project includes a proper retrieval evaluation and a comparison of five embedding models.

- **Text → image:** type "a red double decker bus", get the closest images.
- **Image → image:** upload a photo, get visually similar ones.
- **Stack:** `open_clip` (PyTorch), FAISS (exact cosine search), FastAPI, Streamlit.

## Results

All numbers are produced by `src/eval.py` and stored in [results/](results/).

**Text → image** on COCO val2017: 5,000 images, 25,014 captions used as queries, the caption's own image is the single correct answer.
**Image → image** on Oxford-IIIT Pets: 7,389 images, every image queries the other 7,388, a result is correct if it is the same breed (37 breeds).

| Model | Weights | COCO R@1 | R@5 | R@10 | MRR@10 | Pets P@1 | P@10 | mAP |
|---|---|---|---|---|---|---|---|---|
| ViT-B/32 | LAION-2B | 0.388 | 0.648 | 0.747 | 0.499 | 0.872 | 0.800 | 0.476 |
| ViT-L/14 | LAION-2B | 0.457 | 0.704 | 0.792 | 0.563 | 0.927 | 0.884 | 0.607 |
| ViT-B/16 SigLIP | WebLI | 0.472 | 0.721 | 0.806 | 0.578 | 0.924 | 0.873 | 0.579 |
| ViT-SO400M/14 SigLIP | WebLI | 0.508 | 0.746 | 0.823 | 0.610 | 0.951 | 0.918 | 0.641 |
| **ViT-L/16-256 SigLIP** | WebLI | **0.515** | **0.753** | **0.829** | **0.616** | **0.953** | **0.921** | **0.669** |

R@k = recall@k, P@k = precision@k, MRR = mean reciprocal rank (cut off at 10), mAP = mean average precision over the full ranking.

What the numbers say:

- Model quality matters a lot: going from ViT-B/32 to the best model lifts COCO recall@1 from 38.8% to 51.5% and Pets mAP from 0.476 to 0.669.
- At comparable cost, SigLIP beats LAION CLIP. ViT-B/16 SigLIP (about 200M parameters) beats the much larger LAION ViT-L/14 on COCO text→image.
- The largest model is not the best: SO400M/14 scores slightly below L/16-256. SO400M runs at 224 px input and L/16-256 at 256 px; resolution is a plausible reason, but I did not test it.
- Most Pets errors are between genuinely similar breeds. With ViT-B/32 the top confusions are ragdoll↔birman, beagle→basset hound, British shorthair→Russian blue (see `most_confused` in the results JSON).

### Metric notes

- COCO text→image has exactly one relevant image per query, so precision@10 is capped at 0.1 and is not reported in the table. Recall@k and MRR are the informative metrics there.
- On Pets each breed has roughly 200 images, so recall@k is capped near `k/200` (about 5% at k=10) and is also uninformative; precision@k and mAP are used instead. The JSON files still contain it for completeness.

## Limitations

- **Possible training-data overlap.** The models are pretrained on web-scale data (LAION-2B, WebLI) that may include COCO or Pets images. The numbers show how well this pipeline retrieves with these models, not performance on guaranteed-unseen data.
- **Single gold image per caption.** COCO has several images that fit a generic caption equally well, but only the source image counts as correct, so true quality is higher than the recall numbers suggest.
- **Pets relevance is breed-only.** Two images of different breeds count as a miss even when they look alike.
- **Small, curated datasets.** 5k and 7k images; no stress test of latency or index scaling. Search is exact (`IndexFlatIP`), which is fine at this size but would need an approximate index for millions of images.
- Evaluation covers the embedding models only, not the API or UI.

## Setup

Requires [uv](https://docs.astral.sh/uv/). The project pins Python 3.12 (uv fetches it) and installs PyTorch from the CUDA 12.6 wheel index, which still supports older NVIDIA GPUs such as the GTX 1060. A GPU is optional; CPU works for queries and is just slower for building indices.

```bash
uv sync
```

### Data

Download into `data/` (not tracked by git):

```
data/
├── coco/
│   ├── val2017/                    # http://images.cocodataset.org/zips/val2017.zip
│   ├── captions_val2017.json       # from annotations_trainval2017.zip
│   └── instances_val2017.json
└── pets/
    └── jpg/                        # Oxford-IIIT Pet images; breed is parsed from the filename
```

### Build indices and evaluate

```bash
# default model: ViT-B-32 / laion2b_s34b_b79k
uv run python build_index.py --dataset coco
uv run python build_index.py --dataset pets
uv run python -m src.eval                      # writes results/<model>.json

# another model
uv run python build_index.py --dataset coco --model ViT-L-16-SigLIP-256 --pretrained webli
uv run python build_index.py --dataset pets --model ViT-L-16-SigLIP-256 --pretrained webli
uv run python -m src.eval --model ViT-L-16-SigLIP-256 --pretrained webli
```

Indices are saved under `index/<dataset>/<model>/`. Embedding 5k images takes a few minutes with ViT-B/32 on a GTX 1060; the larger models take noticeably longer (roughly 8 to 12 minutes per dataset for the biggest ones).

## Run the app

```bash
uv run uvicorn api.main:app        # terminal 1: REST API on :8000 (docs at /docs)
uv run streamlit run app.py        # terminal 2: UI
```

The API serves the model from the `SEARCH_MODEL` / `SEARCH_PRETRAINED` environment variables (default ViT-B/32) and needs the matching index to exist.

| Endpoint | Description |
|---|---|
| `GET /health` | Loaded model and available datasets |
| `POST /search/text` | JSON `{query, k, dataset}` → top-k images with scores |
| `POST /search/image` | Multipart upload `file`, plus `k`, `dataset` → top-k images with scores |
| `GET /images/{dataset}/{filename}` | Serves an indexed image (filenames are whitelisted) |

## Project layout

```
src/embed.py     model loading, image/text encoding (L2-normalised vectors)
src/index.py     build/load FAISS index per dataset and model
src/search.py    SearchEngine.by_text / by_image
src/eval.py      recall@k, precision@k, MRR, mAP, per-breed confusion analysis
api/main.py      FastAPI service
app.py           Streamlit UI (talks to the API)
build_index.py   CLI to embed a dataset and save its index
results/         evaluation output per model (JSON)
```

## How it works

1. A pretrained CLIP/SigLIP model encodes every image into a vector; vectors are L2-normalised and stored in a FAISS inner-product index, so the score is cosine similarity.
2. A text query goes through the model's text encoder, an image query through its image encoder; both are looked up in the same index and the top-k are returned ranked by similarity.
3. Evaluation replays labelled queries (COCO captions, Pets breeds) through the same search and measures how often the right images come back.
