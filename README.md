# Gemma RAG — ask questions about your own documents

A small retrieval-augmented generation (RAG) app. Drop `.txt`, `.md` or `.pdf` files into `docs/`, then ask questions in plain English. Answers cite the passages they came from.

- **Retrieval:** `google/embeddinggemma-2`, using the model card's task prefixes (`SearchQuery` for questions, `title: … | text: …` for documents)
- **Vector store:** a NumPy matrix with cosine similarity, so there is no database to install
- **Answering:** `Qwen/Qwen2.5-0.5B-Instruct`, a small model that runs on CPU (change it with `GEN_MODEL` in `rag.py`)

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py          # web UI: click "Build / rebuild index", then ask
# or
python cli.py build
python cli.py ask "Which role offers visa sponsorship and what does it pay?"
python cli.py ask "What chunk size is recommended?" --no-llm   # retrieval only
```

The first run downloads the models from Hugging Face (about 1.5 GB for the embedder and 1 GB for the answer model).

## Files

| File | Purpose |
|---|---|
| `rag.py` | Core pipeline: load docs, chunk, embed, search, prompt, generate |
| `app.py` | Streamlit chat UI with file upload |
| `cli.py` | Command-line version |
| `docs/` | Sample documents (two example job adverts and a RAG explainer) |

## Ideas to extend it

- Use `EMBED_DIM = 256` to get vectors 3× smaller with little loss in quality
- Add an evaluation script that reports recall@k on a handful of question/answer pairs
- Swap in FAISS or Chroma when you have thousands of documents
- Index images too, since EmbeddingGemma 2 puts text and images in the same vector space
