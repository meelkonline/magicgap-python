from transformers import AutoModel, AutoTokenizer
import numpy as np
from sentence_transformers import SentenceTransformer
from api_requests import CosineSimilarityRequest
from nlp_functions import spatie_extract_phrases

model_name = "sentence-transformers/bert-base-nli-mean-tokens"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def embed(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

    # sans doute plus fin, mais trop lent / iterations/s, à tester sur TAS/autres...
    # model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')
    # embeddings = model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
    # return embeddings


def evaluate_cosine_similarity(request: CosineSimilarityRequest):
    str1 = spatie_extract_phrases(request.string1)
    str2 = spatie_extract_phrases(request.string2)

    v1 = embed(str1).detach().numpy().flatten()
    v2 = embed(str2).detach().numpy().flatten()

    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
