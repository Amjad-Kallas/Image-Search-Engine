"""REST API: text and image search over the prebuilt FAISS indices.

Run:  uv run uvicorn api.main:app
Env:  SEARCH_MODEL, SEARCH_PRETRAINED, SEARCH_DEVICE (defaults: ViT-B-32 / laion2b_s34b_b79k / auto)
"""
import io
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from src import index as idx
from src.embed import DEFAULT_MODEL, Embedder
from src.search import SearchEngine

MAX_K = 50
state: dict = {"engines": {}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["embedder"] = Embedder(
        os.getenv("SEARCH_MODEL", DEFAULT_MODEL[0]),
        os.getenv("SEARCH_PRETRAINED", DEFAULT_MODEL[1]),
        device=os.getenv("SEARCH_DEVICE") or None,
    )
    yield
    state.clear()


app = FastAPI(title="Image Search Engine", lifespan=lifespan)


def get_engine(dataset: str) -> SearchEngine:
    if dataset not in idx.IMAGE_DIRS:
        raise HTTPException(404, f"unknown dataset '{dataset}', choose from {list(idx.IMAGE_DIRS)}")
    engines = state["engines"]
    if dataset not in engines:
        try:
            engines[dataset] = SearchEngine(state["embedder"], dataset)
        except Exception as e:  # index not built for this model/dataset
            raise HTTPException(503, f"index for '{dataset}' not available: {e}")
    return engines[dataset]


class Hit(BaseModel):
    file: str
    score: float
    url: str


class SearchResponse(BaseModel):
    model: str
    dataset: str
    results: list[Hit]


class TextQuery(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(10, ge=1, le=MAX_K)
    dataset: str = "coco"


def _respond(dataset: str, hits) -> SearchResponse:
    return SearchResponse(
        model=state["embedder"].tag,
        dataset=dataset,
        results=[Hit(file=f, score=s, url=f"/images/{dataset}/{f}") for f, s in hits],
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": state["embedder"].tag, "datasets": list(idx.IMAGE_DIRS)}


@app.post("/search/text", response_model=SearchResponse)
def search_text(body: TextQuery):
    engine = get_engine(body.dataset)
    return _respond(body.dataset, engine.by_text(body.query, body.k))


@app.post("/search/image", response_model=SearchResponse)
def search_image(
    file: UploadFile = File(...),
    k: int = Form(10, ge=1, le=MAX_K),
    dataset: str = Form("coco"),
):
    engine = get_engine(dataset)
    try:
        image = Image.open(io.BytesIO(file.file.read()))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "uploaded file is not a valid image")
    return _respond(dataset, engine.by_image(image, k))


@app.get("/images/{dataset}/{filename}")
def get_image(dataset: str, filename: str):
    engine = get_engine(dataset)
    if filename not in engine.file_set:  # whitelist: blocks path traversal
        raise HTTPException(404, "image not found")
    return FileResponse(idx.IMAGE_DIRS[dataset] / filename)
