"""Text -> image and image -> image search over a built index."""
from PIL import Image

from src import index as idx
from src.embed import Embedder


class SearchEngine:
    def __init__(self, embedder: Embedder, dataset: str = "coco"):
        self.embedder = embedder
        self.dataset = dataset
        self.index, self.files = idx.load(dataset, embedder.tag)
        self.file_set = set(self.files)

    def _search(self, vec, k):
        scores, ids = self.index.search(vec, k)
        return [(self.files[i], float(s)) for i, s in zip(ids[0], scores[0])]

    def by_text(self, query: str, k: int = 10):
        return self._search(self.embedder.encode_texts([query]), k)

    def by_image(self, image: Image.Image, k: int = 10):
        return self._search(self.embedder.encode_pil([image]), k)
