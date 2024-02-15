import pandas as pd

from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from textblob import TextBlob

from api_requests import UpsertRequest, VectorizeRequest, SearchRequest, UpdateRequest, DeleteRequest, \
    CosineSimilarityRequest
from pinecone_functions import pinecone_upsert, pinecone_search, pinecone_update, pinecone_delete

from nlp_functions import spatie_extract_phrases, evaluate_toxicity, get_lang
from vector_functions import evaluate_cosine_similarity, embed

app = FastAPI()


@app.post("/api/upsert")
def upsert(request: UpsertRequest):
    return pinecone_upsert(request)


@app.post("/api/update")
def update(request: UpdateRequest):
    return pinecone_update(request)


@app.post("/api/delete")
def delete(request: DeleteRequest):
    return pinecone_delete(request)


@app.post("/api/search")
def search(request: SearchRequest):
    return pinecone_search(request)


@app.get("/api/extract_entities")
def extract_entities(text):
    return spatie_extract_phrases(text)


@app.post("/api/lang")
def lang(text):
    return get_lang(text)


@app.post("/api/toxicity")
def toxicity(text):
    return evaluate_toxicity(text)  # [{'label': 'not_toxic', 'score': 0.9954179525375366}]


@app.post("/api/cosine_similarity")
def cosine_similarity(request: CosineSimilarityRequest):
    score = evaluate_cosine_similarity(request)
    return float(score)


@app.post("/api/vectorize")
def vectorize(request: VectorizeRequest):
    embeddings = embed(request.string)
    embeddings_list = embeddings.tolist()
    return embeddings_list
