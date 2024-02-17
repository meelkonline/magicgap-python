import os
import requests
from dotenv import load_dotenv
from pinecone import Pinecone, PineconeApiException

from api_requests import UpsertRequest, SearchRequest, UpdateRequest, DeleteRequest, ListRequest


def pinecone_client():
    load_dotenv()  # Load environment variables from a .env file
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc


def pinecone_upsert(request: UpsertRequest):
    index = pinecone_client().Index(request.index_name)
    try:
        index.upsert(vectors=[{"id": request.id, "values": request.values, "metadata": request.metadata}])
    except PineconeApiException as e:
        print(f"Pinecone API Exception: {e}")


def pinecone_update(request: UpdateRequest):
    index = pinecone_client().Index(request.index_name)
    try:
        index.update(id=request.id, values=request.values, set_metadata=request.metadata)
    except PineconeApiException as e:
        print(f"Pinecone API Exception: {e}")


def pinecone_delete(request: DeleteRequest):
    index = pinecone_client().Index(request.index_name)
    try:
        # Check if 'ids' is provided and not empty
        if request.ids:
            index.delete(ids=request.ids)
        # If 'ids' is not provided or is empty, and 'filter' is provided
        elif request.filter:
            index.delete(filter=request.filter)
        else:
            print("No IDs or filter provided for deletion.")
    except PineconeApiException as e:
        print(f"Pinecone API Exception: {e}")


def pinecone_list(request: ListRequest):
    load_dotenv()  # Load environment variables from a .env file
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_HOST = os.getenv(request.index_name)
    url = PINECONE_HOST+"/vectors/list?prefix=te&limit=100"
    headers = {"accept": "application/json", "Api-Key": PINECONE_API_KEY}
    response = requests.get(url, headers=headers)
    response_data = response.json()
    return response_data.get('vectors', [])  # Default to an empty list if 'vectors' key is not found


def pinecone_search(request: SearchRequest):
    index = pinecone_client().Index(request.index_name)
    result = index.query(vector=request.vector, top_k=request.top_k, filter=request.filter, include_metadata=True)
    simplified_matches = [
        {"id": match.id, "score": match.score, "metadata": match.metadata} for match in result.matches
    ]

    return simplified_matches
