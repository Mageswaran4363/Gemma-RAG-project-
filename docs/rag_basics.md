# What is Retrieval-Augmented Generation?
Retrieval-augmented generation (RAG) combines a search step with a language model. First, documents are split into chunks and converted into embeddings, which are vectors that capture meaning. When a user asks a question, the question is embedded too, and the most similar chunks are retrieved using cosine similarity.
The retrieved chunks are placed into the prompt as context, and the language model writes an answer grounded in those passages. This reduces hallucination and lets the model use private or recent information it was never trained on.
Good chunk sizes are usually 100 to 300 words with some overlap. Embedding quality matters more than the choice of language model for retrieval accuracy.
