"""Command-line version.

    python cli.py build                 # embed everything in ./docs
    python cli.py ask "your question"   # retrieve + answer
    python cli.py ask "..." --no-llm    # retrieval only
"""
import argparse

from rag import RAG, Embedder, Generator, VectorIndex, build_index

p = argparse.ArgumentParser()
p.add_argument("cmd", choices=["build", "ask"])
p.add_argument("question", nargs="?")
p.add_argument("-k", type=int, default=4)
p.add_argument("--docs", default="docs")
p.add_argument("--index", default="index")
p.add_argument("--no-llm", action="store_true")
a = p.parse_args()

embedder = Embedder()
if a.cmd == "build":
    idx = build_index(a.docs, embedder)
    idx.save(a.index)
    print(f"Indexed {len(idx.chunks)} chunks -> {a.index}/")
else:
    if not a.question:
        p.error("ask needs a question")
    rag = RAG(VectorIndex.load(a.index), embedder, None if a.no_llm else Generator())
    res = rag.ask(a.question, k=a.k)
    print("\n" + res["answer"] + "\n\nSources:")
    for i, (c, s) in enumerate(res["sources"], 1):
        print(f"  [{i}] {c.source}  ({s:.3f})  {c.text[:90]}…")
