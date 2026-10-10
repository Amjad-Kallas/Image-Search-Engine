import argparse

from src import index
from src.embed import DEFAULT_MODEL, Embedder

p = argparse.ArgumentParser()
p.add_argument("--dataset", choices=list(index.IMAGE_DIRS), required=True)
p.add_argument("--model", default=DEFAULT_MODEL[0])
p.add_argument("--pretrained", default=DEFAULT_MODEL[1])
a = p.parse_args()
print("saved to", index.build(Embedder(a.model, a.pretrained), a.dataset))
