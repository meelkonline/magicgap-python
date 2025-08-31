from fastapi import FastAPI
from starlette.responses import StreamingResponse
from transformers import pipeline
from api_requests import UpsertRequest, \
    CosineSimilarityRequest, SentimentRequest, \
    TranslateRequest, ChatRequest, ChunkContentRequest, QueryRequest, SummarizeRequest, \
    CompareRequest, AudioStreamRequest, ChunkSimpleDocumentRequest, MultipleStringRequest
from audio_functions import kokoro_stream_audio, kokoro_save_audio
from comparison import compare
from faiss_functions import handle_faiss_upsert, handle_faiss_query
from llama_functions import llama32_3b_ask, llama32_3b_quiz
from nlp_functions import spatie_extract_phrases, extract_sentences
from sentiment_analysis import evaluate_sentiment
from summarization import summarize_conversation
from pdf_chunker import SimplePdfChunker
from vector_functions import evaluate_cosine_similarity, embed, smart_embed
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a', format='%(name)s - %(levelname)s - %('
                                                                                 'message)s')


@app.get("/api/extract_entities")
def extract_entities(text):
    return spatie_extract_phrases(text)


@app.post("/api/chunk_simple_document")
def chunk_simple_document(request: ChunkSimpleDocumentRequest):
    chunker = SimplePdfChunker(request.filepath)
    return chunker.chunk_document()


@app.post("/api/summarize")
def summarize(request: SummarizeRequest):
    return summarize_conversation(request.messages, request.max_length)


@app.post("/api/simple_cosine_similarity")
def simple_cosine_similarity(request: CosineSimilarityRequest):
    score = evaluate_cosine_similarity(request)
    return float(score)


@app.post("/api/vectorize")
def vectorize(request: MultipleStringRequest):
    # embeddings = embed(request.strings)
    # embeddings_list = embeddings.tolist()

    embeddings = smart_embed(request.strings)  # returns tensor [N,384] on GPU
    embeddings_list = embeddings.cpu().tolist()  # move to CPU, convert to list
    return embeddings_list


@app.post("/api/sentiment")
def sentiment(request: SentimentRequest):
    return evaluate_sentiment(request)


@app.post("/api/chat/ask")
def chat_ask(request: ChatRequest):
    result = llama32_3b_ask(request)
    return result


@app.post("/api/quiz/answer")
def quiz_answer(request: ChatRequest):
    result = llama32_3b_quiz(request)
    return result


@app.post("/api/compare_premise")
def compare_premise(request: CompareRequest):
    return compare(request)


@app.post("/api/translate")
def translate(request: TranslateRequest):
    output = ""
    src_tg = request.src_lang + '-' + request.target_lang
    match src_tg:
        case 'fr-en':
            pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
            output = pipe(request.strings)

        case 'en-fr':
            pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
            output = pipe(request.strings)

    return output


@app.post("/api/faiss/upsert")
def faiss_upsert(request: UpsertRequest):
    handle_faiss_upsert(request)
    return {"message": "Vector upserted successfully"}


@app.post("/api/faiss/query")
def faiss_query(request: QueryRequest):
    result = handle_faiss_query(request)
    return result


@app.post("/api/audio/save")
def save_audio(request: AudioStreamRequest):
    return kokoro_save_audio(request.text, request.lang, request.voice_id)


@app.post("/api/audio/stream")
async def stream_audio(request: AudioStreamRequest):
    text = request.text
    lang = request.lang
    voice_id = request.voice_id

    if not text:
        return {"error": "No text provided."}

    # Return a streaming response with text/plain so it doesn't try to parse a big JSON array
    return StreamingResponse(kokoro_stream_audio(text, lang, voice_id), media_type="text/plain")
