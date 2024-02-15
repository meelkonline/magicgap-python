from pydantic import BaseModel
from typing import List, Any, Optional


class SingleStringRequest(BaseModel):
    string: str


class UpsertRequest(BaseModel):
    index_name: str
    id: str
    values: List[float]
    metadata: Optional[dict] = {}


class UpdateRequest(BaseModel):
    index_name: str
    id: str
    values: List[float]
    metadata: Optional[dict] = {}


class DeleteRequest(BaseModel):
    index_name: str
    ids: Optional[List[str]]
    filter: Optional[dict] = {}


class SearchRequest(BaseModel):
    index_name: str
    vector: List[float]
    filter: Optional[dict] = {}
    top_k: Optional[int] = 5


class CosineSimilarityRequest(BaseModel):
    string1: str
    string2: str
