import numpy as np
import torch
import re
from api_requests import CosineSimilarityRequest
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_text(text):
    # Normalize case and strip unnecessary spaces/punctuation
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text


# Function to initialize and get the model
def get_model():
    global model
    if model is None:
        print("Loading multilingual model...")
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=device)
    return model


def evaluate_cosine_similarity(request: CosineSimilarityRequest):
    evaluation_model = get_model()  # Ensure the model is loaded
    # Encode target answer and conversation into vectors
    target_embedding = evaluation_model.encode(preprocess_text(request.target), convert_to_tensor=True)
    conversation_embeddings = evaluation_model.encode([preprocess_text(msg) for msg in request.messages],
                                                      convert_to_tensor=True)

    # Compute cosine similarity between the target and each message in the conversation
    similarities = util.pytorch_cos_sim(target_embedding, conversation_embeddings)

    # Find the message with the highest similarity
    max_similarity, idx = torch.max(similarities, dim=1)

    print(f"Found : '{request.messages[idx.item()]}'")

    return max_similarity.item()


def embed(sentences):
    model = get_model()  # Ensure the model is loaded
    if isinstance(sentences, str):
        sentences = [sentences]

    with torch.no_grad():
        embeddings = model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
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
