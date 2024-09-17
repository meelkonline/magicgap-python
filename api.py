from fastapi import FastAPI
from transformers import pipeline

from api_requests import UpsertRequest, SingleStringRequest, SearchRequest, UpdateRequest, DeleteRequest, \
    CosineSimilarityRequest, ListRequest, ExtractSemanticChunksRequest, SentimentRequest, \
    PhonemizeAudioRequest, TranslateRequest, ChatRequest
from llama_functions import llama_ask
from pinecone_functions import pinecone_upsert, pinecone_search, pinecone_update, pinecone_delete, pinecone_list
from nlp_functions import spatie_extract_phrases, get_lang, \
    evaluate_sentiment, load_text, extract_sentences, get_toxicity
from speech_functions import phonemize_audio
from vector_functions import evaluate_cosine_similarity, embed
from sklearn.metrics.pairwise import cosine_similarity
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


@app.post("/api/semantic_chunks")
def semantic_chunks(request: ExtractSemanticChunksRequest):
    """ Extract, embed and cluster sentences based on cosine similarity. """
    text = load_text(request.filepath)
    sentences = extract_sentences(text)
    for s in sentences:
        logging.log(0, '-> ' + s)

    embeddings = embed(sentences)

    # Example of clustering based on cosine similarity threshold
    clusters = []
    for i in range(len(sentences)):
        found_cluster = False
        for cluster in clusters:
            # if max(cosine_similarity([embeddings[i]], [embeddings[idx] for idx in cluster]))[0] > request.threshold:
            if max(cosine_similarity(
                    [embeddings[i].cpu().numpy()],
                    [embeddings[idx].cpu().numpy() for idx in cluster]
            ))[0] > request.threshold:
                cluster.append(i)
                found_cluster = True
                break
        if not found_cluster:
            clusters.append([i])

    # Extracting the clustered sentences
    clustered_sentences = [[' '.join(sentences[idx] for idx in cluster)] for cluster in clusters]
    return clustered_sentences


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


@app.post("/api/phonemize")
def phonemize(request: PhonemizeAudioRequest):
    return phonemize_audio(request.lang, request.audiopath)


@app.post("/api/sentiment")
def sentiment(request: SentimentRequest):
    return evaluate_sentiment(request)


@app.post("/api/toxicity")
def toxicity(request: SingleStringRequest):
    result = get_toxicity(request.string)
    return result  # Return the dictionary directly


@app.post("/api/chat/ask")
def chat_ask(request: ChatRequest):
    result = llama_ask(request)
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