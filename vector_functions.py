import numpy as np
import torch
import re
from api_requests import CosineSimilarityRequest
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load a smaller and faster model
embed_model_name = 'paraphrase-MiniLM-L6-v2'
embed_model = SentenceTransformer(embed_model_name, device="cuda")

def preprocess_text(text):
    # Normalize case and strip unnecessary spaces/punctuation
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text


def evaluate_cosine_similarity(request: CosineSimilarityRequest):

    # Preprocess texts
    target_text = preprocess_text(request.target)
    messages_texts = [preprocess_text(msg) for msg in request.messages]

    # Encode target and messages
    with torch.no_grad():
        target_embedding = embed_model.encode(target_text, convert_to_tensor=True, device="cuda")
        conversation_embeddings = embed_model.encode(messages_texts, convert_to_tensor=True, device="cuda")

    # Compute cosine similarity
    similarities = util.cos_sim(target_embedding, conversation_embeddings)
    print(similarities)
    # Find the message with the highest similarity
    max_similarity, _ = torch.max(similarities, dim=1)

    return max_similarity.item()


def embed(sentences):

    if isinstance(sentences, str):
        sentences = [sentences]

    with torch.no_grad():
        embeddings = embed_model.encode(sentences, convert_to_tensor=True, show_progress_bar=False, device="cuda")
    return embeddings


def semantic_chunks(sentences, threshold):
    """ Extract, embed and cluster sentences based on cosine similarity. """
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
            ))[0] > threshold:
                cluster.append(i)
                found_cluster = True
                break
        if not found_cluster:
            clusters.append([i])

    # Extracting the clustered sentences
    clustered_sentences = [[' '.join(sentences[idx] for idx in cluster)] for cluster in clusters]
    return clustered_sentences
