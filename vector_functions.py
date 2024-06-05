from transformers import AutoModel, AutoTokenizer
import numpy as np
from api_requests import CosineSimilarityRequest

model_name = "sentence-transformers/bert-base-nli-mean-tokens"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def embed(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings


def evaluate_cosine_similarity(request: CosineSimilarityRequest):

    v1 = embed(request.string1).detach().numpy().flatten()
    v2 = embed(request.string2).detach().numpy().flatten()

    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
