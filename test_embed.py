from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-2")
q = model.encode("Which job sponsors visas?", prompt_name="SearchQuery")
docs = model.encode(["title: none | text: Visa sponsorship is available.",
                     "title: none | text: The office is in Chelmsford."])
print(model.similarity(q, docs))
