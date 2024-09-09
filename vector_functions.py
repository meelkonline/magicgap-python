import numpy as np
import torch
import re
from api_requests import CosineSimilarityRequest
from sentence_transformers import SentenceTransformer, util

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


def old_evaluate_cosine_similarity(request: CosineSimilarityRequest):
    try:
        v1 = embed(request.string1)[0]  # Assuming embed returns a list of embeddings
        v2 = embed(request.string2)[0]

        if torch.cuda.is_available():
            # Ensure tensors are moved to CPU before using them with numpy
            norm_v1 = np.linalg.norm(v1.cpu().numpy())  # Move tensor to CPU and convert to NumPy array
            norm_v2 = np.linalg.norm(v2.cpu().numpy())  # Move tensor to CPU and convert to NumPy array

            cosine_similarity = np.dot(v1.cpu().numpy(), v2.cpu().numpy()) / (norm_v1 * norm_v2)
        else:
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            if norm_v1 == 0 or norm_v2 == 0:
                raise ValueError("One of the vectors is a zero vector, cannot compute cosine similarity.")
            cosine_similarity = np.dot(v1, v2) / (norm_v1 * norm_v2)

        return cosine_similarity

    except Exception as e:
        raise
