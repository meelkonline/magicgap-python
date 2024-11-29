from pydantic import BaseModel
from typing import List, Any, Optional

from typing_extensions import Dict, Tuple


class SingleStringRequest(BaseModel):
    string: str


class MultipleStringRequest(BaseModel):
    strings: List[str]


class PhonemizeAudioRequest(BaseModel):
    lang: str
    audiopath: str


class PhonemizeSegmentRequest(BaseModel):
    string: str
    lang: str


class PhonemizeStreamRequest(BaseModel):
    lang: str
    characters: List[str]
    start_times: List[float]
    end_times: List[float]


class SentimentRequest(BaseModel):
    lang: str
    strings: List[str]


class TranslateRequest(BaseModel):
    src_lang: str
    target_lang: str
    strings: List[str]


class UpsertRequest(BaseModel):
    id: str
    text: str


class QueryRequest(BaseModel):
    text: str
    top_k: Optional[int] = 10


class ChatRequest(BaseModel):
    messages: str
    options: Optional[dict] = {}


class CosineSimilarityRequest(BaseModel):
    target: str
    messages: List[str]


class SummarizeRequest(BaseModel):
    messages: List[str]
    max_length: int


class CompareRequest(BaseModel):
    premise: str
    hypotheses: List[Tuple[int, str]]  # Allow duplicate IDs
    min_score: int


class ChunkDocumentRequest(BaseModel):
    filepath: str
    threshold: float


class ChunkContentRequest(BaseModel):
    text: str
    threshold: float
