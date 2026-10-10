"""CLIP model loading and image/text encoding (L2-normalised float32 vectors)."""
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

DEFAULT_MODEL = ("ViT-B-32", "laion2b_s34b_b79k")


class Embedder:
    def __init__(self, model_name=DEFAULT_MODEL[0], pretrained=DEFAULT_MODEL[1], device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.tag = f"{model_name}__{pretrained}"

    @torch.no_grad()
    def _normalise(self, feats: torch.Tensor) -> np.ndarray:
        feats = feats.float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype("float32")

    @torch.no_grad()
    def encode_pil(self, images: list[Image.Image]) -> np.ndarray:
        batch = torch.stack([self.preprocess(im.convert("RGB")) for im in images]).to(self.device)
        with torch.autocast(self.device, enabled=self.device == "cuda"):
            return self._normalise(self.model.encode_image(batch))

    def encode_images(self, paths: list[Path], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in tqdm(range(0, len(paths), batch_size), desc="images"):
            imgs = [Image.open(p) for p in paths[i : i + batch_size]]
            out.append(self.encode_pil(imgs))
        return np.concatenate(out)

    @torch.no_grad()
    def encode_texts(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            tokens = self.tokenizer(texts[i : i + batch_size]).to(self.device)
            with torch.autocast(self.device, enabled=self.device == "cuda"):
                out.append(self._normalise(self.model.encode_text(tokens)))
        return np.concatenate(out)
