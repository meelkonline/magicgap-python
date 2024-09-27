import faiss
import numpy as np

from api_requests import QueryRequest, UpsertRequest
from vector_functions import embed

# Initialize FAISS index
d = 384  # Dimension of vectors
index = faiss.IndexIDMap(faiss.IndexFlatL2(d))


def handle_faiss_upsert(request: UpsertRequest):
    vector = embed(request.text).cpu().numpy()
    vector = np.array([vector], dtype='float32').reshape(1, -1)  # Ensure it's a 2D array with shape (1, d)
    index.add_with_ids(vector, np.array([request.id], dtype='int64'))


#   todo:batch upsert pour dataset/etc...


def handle_faiss_query(request: QueryRequest):
    vector = embed(request.text)
    distances, indices = index.search(np.array([vector], dtype='float32'), request.top_k)
    return {"indices": indices[0].tolist(), "distances": distances[0].tolist()}
