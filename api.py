import pandas as pd

from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from textblob import TextBlob

from api_requests import UpsertRequest, VectorizeRequest, SearchRequest, UpdateRequest, DeleteRequest
from pinecone_functions import pinecone_upsert, pinecone_search, pinecone_update, pinecone_delete

from nlp_functions import spatie_extract_phrases, evaluate_toxicity, get_lang

app = FastAPI()


@app.post("/api/chunk_csv")
def chunk_csv(request: SearchRequest):
    df = pd.read_csv("real_estate_courses.csv")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # vectorizing the course description
    df['description_vector'] = df['description'].apply(lambda x: model.encode(x))
    pinecone_data = [
        {
            "id": str(index),  # Unique identifier for the course
            "values": row['description_vector'],  # Vector representation of the course description
            "metadata": {  # Additional details about the course
                "name": row['name'],
                "required_equipment": row['required equipment'],
                # Add more fields as necessary
            }
        }
        for index, row in df.iterrows()
    ]


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


@app.post("/api/vectorize")
def vectorize(request: VectorizeRequest):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(request.sentences)
    embeddings_list = [embedding.tolist() for embedding in embeddings]
    return embeddings_list
