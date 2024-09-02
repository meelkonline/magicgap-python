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
    try:
        # Embed both input strings
        v1 = embed(request.string1)[0]  # Assuming embed returns a list of embeddings
        v2 = embed(request.string2)[0]

        # Compute the norms to ensure they are not zero
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        # Check if norms are zero to prevent division by zero
        if norm_v1 == 0 or norm_v2 == 0:
            raise ValueError("One of the vectors is a zero vector, cannot compute cosine similarity.")

        # Calculate cosine similarity
        cosine_similarity = np.dot(v1, v2) / (norm_v1 * norm_v2)
        return cosine_similarity

    except Exception as e:
        raise
