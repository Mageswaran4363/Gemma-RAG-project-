"""Streamlit UI:  streamlit run app.py"""
import shutil
from pathlib import Path

import streamlit as st

from rag import RAG, Embedder, Generator, VectorIndex, build_index

DOCS, INDEX = Path("docs"), Path("index")

st.set_page_config(page_title="Gemma RAG", page_icon="🔎", layout="wide")
st.title("Ask your documents")
st.caption("Retrieval: google/embeddinggemma-2 · Generation: small local LLM")


@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder():
    return Embedder()


@st.cache_resource(show_spinner="Loading answer model…")
def get_generator():
    return Generator()


with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("Add .txt / .md / .pdf", type=["txt", "md", "pdf"],
                             accept_multiple_files=True)
    if files:
        DOCS.mkdir(exist_ok=True)
        for f in files:
            (DOCS / f.name).write_bytes(f.getbuffer())
        st.success(f"Saved {len(files)} file(s). Rebuild the index.")
    st.write(sorted(p.name for p in DOCS.glob("*") if p.is_file()))
    if st.button("Build / rebuild index", type="primary"):
        shutil.rmtree(INDEX, ignore_errors=True)
        with st.spinner("Embedding chunks…"):
            build_index(DOCS, get_embedder()).save(INDEX)
        st.success("Index ready")
    k = st.slider("Passages to retrieve", 1, 10, 4)
    use_llm = st.toggle("Generate an answer", value=True,
                        help="Off = retrieval only (fast, no LLM download)")

if not (INDEX / "vectors.npy").exists():
    st.info("Click **Build / rebuild index** in the sidebar to get started.")
    st.stop()

rag = RAG(VectorIndex.load(INDEX), get_embedder(), get_generator() if use_llm else None)

question = st.chat_input("Ask a question about your documents")
if question:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = rag.ask(question, k=k)
        st.write(result["answer"])
        with st.expander("Sources"):
            for i, (chunk, score) in enumerate(result["sources"], 1):
                st.markdown(f"**[{i}] {chunk.source}** — similarity {score:.3f}")
                st.caption(chunk.text)
