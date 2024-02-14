import pandas as pd

from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from textblob import TextBlob

from api_requests import UpsertRequest, VectorizeRequest, SearchRequest, UpdateRequest, DeleteRequest
from pinecone_functions import pinecone_upsert, pinecone_search, pinecone_update, pinecone_delete

from spacy_functions import spatie_extract_phrases

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


@app.post("/api/vectorize")
def vectorize(request: VectorizeRequest):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(request.sentences)
    embeddings_list = [embedding.tolist() for embedding in embeddings]
    return embeddings_list


@app.get("/api/get-sentiment")
def get_sentiment(text: str):
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    analysis = TextBlob(text)
    # Sentiment property returns a namedtuple of the form Sentiment(polarity, subjectivity)
    # polarity is a float within the range [-1.0, 1.0]
    # subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective
    sentiment_score = analysis.sentiment.polarity

    return sentiment_score
