import numpy as np
import torch
import re
from api_requests import CosineSimilarityRequest
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

dtype = torch.float16  # try bfloat16 on newer GPUs if preferred
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# embed_model_name = 'paraphrase-MiniLM-L6-v2'
# embed_model = SentenceTransformer(embed_model_name, device="cuda")

# One-time init (do this at process start)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "paraphrase-MiniLM-L6-v2"
#MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embed_model = SentenceTransformer(MODEL_NAME)
embed_model.to(DEVICE)
embed_model.eval()  # keep it warm
tokenizer = embed_model.tokenizer

# Optional: raise model cap (does not change 384-d output)
embed_model.max_seq_length = 512  # safe upper limit for MiniLM backbones


def _token_count(text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _chunk_by_tokens(text: str, chunk_size=240, overlap=48):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= chunk_size:
        return [text]
    chunks = []
    step = chunk_size - overlap
    for s in range(0, len(ids), step):
        piece_ids = ids[s:s + chunk_size]
        chunks.append(tokenizer.decode(piece_ids, skip_special_tokens=True))
    return chunks


def smart_embed(texts, short_cap=200, chunk_size=240, overlap=48,
                batch_size=256, normalize=False):
    """
    Returns a tensor of shape [N, 384] on DEVICE.
    - If input is a string, returns shape [1, 384]
    """
    if isinstance(texts, str):
        texts = [texts]

    # Build per-text chunk lists
    per_text_chunks = []
    for t in texts:
        if _token_count(t) <= short_cap:
            per_text_chunks.append([t])
        else:
            per_text_chunks.append(_chunk_by_tokens(t, chunk_size, overlap))

    # Flatten for one batched encode call
    flat = [c for chunks in per_text_chunks for c in chunks]

    with torch.inference_mode():
        embs = embed_model.encode(
            flat,
            convert_to_tensor=True,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize,
        ).to(DEVICE)

    # Re-group and mean-pool to one vector per original text
    out = []
    i = 0
    for chunks in per_text_chunks:
        n = len(chunks)
        out.append(embs[i:i + n].mean(dim=0))
        i += n
    return torch.stack(out)  # [len(texts), 384] on DEVICE


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
    # print(similarities)
    # Find the message with the highest similarity
    max_similarity, _ = torch.max(similarities, dim=1)

    return max_similarity.item()


def embed(sentences):
    if isinstance(sentences, str):
        sentences = [sentences]

    with torch.no_grad():
        embeddings = embed_model.encode(sentences, convert_to_tensor=True, show_progress_bar=False, device="cuda")
    return embeddings
