from fastapi import FastAPI
from api_requests import UpsertRequest, SingleStringRequest, SearchRequest, UpdateRequest, DeleteRequest, \
    CosineSimilarityRequest, ListRequest, ExtractChunksRequest, MultipleStringRequest
from pinecone_functions import pinecone_upsert, pinecone_search, pinecone_update, pinecone_delete, pinecone_list
from nlp_functions import spatie_extract_phrases, get_lang, extract_document_chunks, \
    evaluate_sentiment
from speech_functions import phonemize_audio
from vector_functions import evaluate_cosine_similarity, embed

import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

@app.post("/api/upsert")
def upsert(request: UpsertRequest):
    return pinecone_upsert(request)


@app.post("/api/update")
def update(request: UpdateRequest):
    return pinecone_update(request)


@app.post("/api/delete")
def delete(request: DeleteRequest):
    return pinecone_delete(request)


@app.post("/api/list")
def vector_list(request: ListRequest):
    return pinecone_list(request)


@app.post("/api/search")
def search(request: SearchRequest):
    return pinecone_search(request)


@app.get("/api/extract_entities")
def extract_entities(text):
    return spatie_extract_phrases(text)


@app.post("/api/extract_chunks")
def extract_chunks(request: ExtractChunksRequest):
    return extract_document_chunks(request.filepath, request.max_words)


@app.post("/api/lang")
def lang(request: SingleStringRequest):
    return get_lang(request.string)


@app.post("/api/cosine_similarity")
def cosine_similarity(request: CosineSimilarityRequest):
    score = evaluate_cosine_similarity(request)
    return float(score)


@app.post("/api/vectorize")
def vectorize(request: SingleStringRequest):
    embeddings = embed(request.string)
    embeddings_list = embeddings.tolist()
    return embeddings_list


@app.post("/api/phonemize")
def phonemize(request: MultipleStringRequest):
    return phonemize_audio(request.strings[0], request.strings[1])


@app.post("/api/sentiment")
def sentiment(request: MultipleStringRequest):
    return evaluate_sentiment(request)

