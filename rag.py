"""Minimal RAG pipeline built on google/embeddinggemma-2.

Steps: load docs -> chunk -> embed (with task prefixes) -> cosine search -> generate answer.
The vector index is a plain NumPy matrix saved to disk, so there is no vector DB to install.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

EMBED_MODEL = "google/embeddinggemma-2"
GEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # ungated, ~6 GB download; use Qwen2.5-0.5B-Instruct for a tiny/fast option
EMBED_DIM = 768  # can be 512 / 256 / 128 thanks to Matryoshka truncation


# ---------------------------------------------------------------- documents
@dataclass
class Chunk:
    source: str
    title: str
    text: str


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, max_words: int = 180, overlap: int = 40) -> list[str]:
    """Split on paragraphs, then pack into ~max_words windows with overlap."""
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    if not words or words == [""]:
        return []
    step = max(1, max_words - overlap)
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), step)
            if words[i:i + max_words]]


def load_chunks(folder: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(folder).rglob("*")):
        if path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue
        title = path.stem.replace("_", " ").replace("-", " ")
        for piece in chunk_text(read_file(path)):
            chunks.append(Chunk(source=path.name, title=title, text=piece))
    return chunks


# ---------------------------------------------------------------- embeddings
class Embedder:
    """Wraps EmbeddingGemma 2 and applies the model card's task prefixes."""

    def __init__(self, model_name: str = EMBED_MODEL, dim: int = EMBED_DIM):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, truncate_dim=dim)

    def embed_documents(self, chunks: list[Chunk]) -> np.ndarray:
        # Model card format for documents: "title: {title} | text: {content}"
        texts = [f"title: {c.title or 'none'} | text: {c.text}" for c in chunks]
        return self.model.encode(texts, normalize_embeddings=True,
                                 batch_size=16, show_progress_bar=True)

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], prompt_name="SearchQuery",
                                 normalize_embeddings=True)[0]


# ---------------------------------------------------------------- index
class VectorIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        self.chunks = chunks
        self.vectors = np.asarray(vectors, dtype=np.float32)

    def search(self, query_vec: np.ndarray, k: int = 4) -> list[tuple[Chunk, float]]:
        scores = self.vectors @ np.asarray(query_vec, dtype=np.float32)  # cosine (normalised)
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        (folder / "chunks.json").write_text(json.dumps([asdict(c) for c in self.chunks], indent=1))

    @classmethod
    def load(cls, folder: str | Path) -> "VectorIndex":
        folder = Path(folder)
        chunks = [Chunk(**c) for c in json.loads((folder / "chunks.json").read_text())]
        return cls(chunks, np.load(folder / "vectors.npy"))


def build_index(doc_folder: str | Path, embedder: Embedder) -> VectorIndex:
    chunks = load_chunks(doc_folder)
    if not chunks:
        raise ValueError(f"No .txt/.md/.pdf files found in {doc_folder}")
    return VectorIndex(chunks, embedder.embed_documents(chunks))


# ---------------------------------------------------------------- generation
def build_prompt(question: str, hits: list[tuple[Chunk, float]]) -> list[dict]:
    context = "\n\n".join(f"[{i + 1}] ({c.source}) {c.text}" for i, (c, _) in enumerate(hits))
    return [
        {"role": "system", "content":
            "Answer using ONLY the context. Cite sources like [1]. "
            "If the context does not contain the answer, say you don't know."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


class Generator:
    def __init__(self, model_name: str = GEN_MODEL):
        from transformers import pipeline
        self.pipe = pipeline("text-generation", model=model_name, device_map="auto")

    def answer(self, messages: list[dict], max_new_tokens: int = 300) -> str:
        out = self.pipe(messages, max_new_tokens=max_new_tokens, do_sample=False)
        return out[0]["generated_text"][-1]["content"].strip()


class RAG:
    def __init__(self, index: VectorIndex, embedder: Embedder, generator: Generator | None):
        self.index, self.embedder, self.generator = index, embedder, generator

    def ask(self, question: str, k: int = 4) -> dict:
        hits = self.index.search(self.embedder.embed_query(question), k=k)
        answer = (self.generator.answer(build_prompt(question, hits))
                  if self.generator else "(generation disabled — showing retrieved passages only)")
        return {"answer": answer, "sources": hits}
