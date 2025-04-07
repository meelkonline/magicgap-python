from fastapi import FastAPI
from starlette.responses import StreamingResponse
from transformers import pipeline
from api_requests import UpsertRequest, SingleStringRequest, \
    CosineSimilarityRequest, SentimentRequest, \
    TranslateRequest, ChatRequest, ChunkSemanticDocumentRequest, ChunkContentRequest, QueryRequest, SummarizeRequest, \
    CompareRequest, AudioStreamRequest, ChunkSimpleDocumentRequest
from audio_functions import kokoro_stream_audio, kokoro_save_audio
from comparison import compare
from faiss_functions import handle_faiss_upsert, handle_faiss_query
from llama_functions import llama32_3b_ask, llama32_3b_quiz
from nlp_functions import spatie_extract_phrases, load_text, extract_sentences
from sentiment_analysis import evaluate_sentiment
from summarization import summarize_conversation
from pdf_chunker import SmartPdfChunker, SimplePdfChunker
from vector_functions import evaluate_cosine_similarity, embed, semantic_chunks
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a', format='%(name)s - %(levelname)s - %('
                                                                                 'message)s')


@app.get("/api/extract_entities")
def extract_entities(text):
    return spatie_extract_phrases(text)


@app.post("/api/chunk_semantic_document")
def chunk_semantic_document(request: ChunkSemanticDocumentRequest):
    text = load_text(request.filepath)
    sentences = extract_sentences(text)
    return semantic_chunks(sentences, request.threshold)


@app.post("/api/chunk_simple_document")
def chunk_simple_document(request: ChunkSimpleDocumentRequest):
    chunker = SimplePdfChunker(request.filepath)
    return chunker.chunk_document()


@app.post("/api/smart_chunk")
def smart_chunk(request: ChunkSemanticDocumentRequest):
    chunker = SmartPdfChunker(pdf_path=request.filepath)
    articles = chunker.chunk_document()

    return articles


@app.post("/api/summarize")
def summarize(request: SummarizeRequest):
    #return llama32_summarize(request)
    return summarize_conversation(request.messages, request.max_length)


@app.post("/api/chunk_content")
def chunk_content(request: ChunkContentRequest):
    sentences = extract_sentences(request.text)
    return semantic_chunks(sentences, request.threshold)


# @app.post("/api/lang")
# def lang(request: SingleStringRequest):
#     return get_lang(request.string)


@app.post("/api/simple_cosine_similarity")
def simple_cosine_similarity(request: CosineSimilarityRequest):
    score = evaluate_cosine_similarity(request)
    return float(score)


@app.post("/api/vectorize")
def vectorize(request: SingleStringRequest):
    embeddings = embed(request.string)
    embeddings_list = embeddings.tolist()
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
