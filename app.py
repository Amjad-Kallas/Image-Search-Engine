"""Streamlit UI. Needs the API running:  uv run uvicorn api.main:app
Run:  uv run streamlit run app.py
"""
import os

import requests
import streamlit as st

API = os.getenv("SEARCH_API", "http://127.0.0.1:8000")
COLS = 5

st.set_page_config(page_title="Image Search", layout="wide")
st.title("Image Search Engine")


@st.cache_data(ttl=30)
def health():
    return requests.get(f"{API}/health", timeout=5).json()


try:
    info = health()
except requests.RequestException:
    st.error(f"API not reachable at {API}. Start it with: uv run uvicorn api.main:app")
    st.stop()

with st.sidebar:
    dataset = st.selectbox("Dataset", info["datasets"])
    k = st.slider("Results", 5, 50, 10, step=5)
    st.caption(f"Model: {info['model']}")


def show(resp: requests.Response):
    if not resp.ok:
        st.error(resp.json().get("detail", resp.text))
        return
    data = resp.json()
    results = data["results"]
    for row in range(0, len(results), COLS):
        cols = st.columns(COLS)
        for col, hit in zip(cols, results[row : row + COLS]):
            col.image(f"{API}{hit['url']}", width="stretch")
            col.caption(f"{hit['score']:.3f}")


text_tab, image_tab = st.tabs(["Search by text", "Search by image"])

with text_tab:
    query = st.text_input("Describe what you are looking for", placeholder="a dog catching a frisbee")
    if query:
        show(requests.post(f"{API}/search/text", json={"query": query, "k": k, "dataset": dataset}, timeout=60))

with image_tab:
    upload = st.file_uploader("Upload a query image", type=["jpg", "jpeg", "png", "webp"])
    if upload:
        st.image(upload, width=250, caption="Query")
        resp = requests.post(
            f"{API}/search/image",
            files={"file": (upload.name, upload.getvalue(), upload.type)},
            data={"k": k, "dataset": dataset},
            timeout=60,
        )
        show(resp)
