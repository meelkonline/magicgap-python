from fastapi import FastAPI
from transformers import pipeline
from api_requests import UpsertRequest, SingleStringRequest, \
    CosineSimilarityRequest, SentimentRequest, \
    TranslateRequest, ChatRequest, ChunkDocumentRequest, ChunkContentRequest, QueryRequest, SummarizeRequest

from faiss_functions import handle_faiss_upsert, handle_faiss_query
from llama_functions import llama32_3b_ask
from nlp_functions import spatie_extract_phrases, get_lang, \
    evaluate_sentiment, load_text, extract_sentences, get_toxicity
from vector_functions import evaluate_cosine_similarity, embed, semantic_chunks, summarize_sentences
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a', format='%(name)s - %(levelname)s - %('
                                                                                 'message)s')


@app.get("/api/extract_entities")
def extract_entities(text):
    return spatie_extract_phrases(text)


@app.post("/api/chunk_document")
def chunk_document(request: ChunkDocumentRequest):
    text = load_text(request.filepath)
    sentences = extract_sentences(text)
    return semantic_chunks(sentences, request.threshold)


@app.post("/api/summarize")
def summarize(request: SummarizeRequest):
    return summarize_sentences(request.messages, request.max_length)


@app.post("/api/chunk_content")
def chunk_content(request: ChunkContentRequest):
    sentences = extract_sentences(request.text)
    return semantic_chunks(sentences, request.threshold)


@app.post("/api/lang")
def lang(request: SingleStringRequest):
    return get_lang(request.string)


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


@app.post("/api/toxicity")
def toxicity(request: SingleStringRequest):
    result = get_toxicity(request.string)
    return result  # Return the dictionary directly


@app.post("/api/chat/ask")
def chat_ask(request: ChatRequest):
    result = llama32_3b_ask(request)
    return result


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
