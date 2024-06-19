import numpy as np
from api_requests import CosineSimilarityRequest
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


def embed(sentences):
    if isinstance(sentences, str):
        sentences = [sentences]
    embeddings = model.encode(sentences, show_progress_bar=False)
    return embeddings


def evaluate_cosine_similarity(request: CosineSimilarityRequest):
    v1 = embed(request.string1).detach().numpy().flatten()
    v2 = embed(request.string2).detach().numpy().flatten()
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
