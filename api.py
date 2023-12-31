import string
import spacy
import torch

from fastapi import FastAPI, HTTPException
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModel

# Load the pre-trained model and tokenizer (e.g., DistilBERT)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

from helpers.helpers import read_file, chunk_pdf, extract_coherent_chunks, process_document, generate_chunks, \
    generate_openai_qas

app = FastAPI()


@app.get("/api/hello")
def hello(text):
    return 'hi'


@app.get("/api/vectorize")
def vectorize(text: str):
    # Tokenize and encode the text for the model
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Generate the embedding
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling to get a single vector (embedding) for the entire text
    vector = outputs.last_hidden_state.mean(dim=1).numpy()

    # Convert the NumPy array to a list (or string) for JSON serialization
    vector_list = vector.tolist()  # Convert to list
    # Or, alternatively, convert to a string if preferred
    # vector_str = json.dumps(vector_list)

    return {"vector": vector_list}


@app.get("/api/chunk-document")
def chunk_document(file_path, max_chunk_size, threshold=0.5):
    chunks = chunk_pdf(file_path, max_chunk_size)

    return chunks


@app.get("/api/smart-chunk")
def smart_chunk(file_path, description, openai_api_key):
    processed_chunks = process_document(file_path)
    chunks = []
    for chunk in processed_chunks:
        generated_chunks = generate_chunks(chunk, description, openai_api_key)
        chunks.append(generated_chunks)

    return chunks


@app.get("/api/extract-qas")
def extract_qas(content, description, openai_api_key):
    return generate_openai_qas(content, description, openai_api_key)


@app.get("/api/get-sentiment")
def get_sentiment(text: str):
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    analysis = TextBlob(text)
    # Sentiment property returns a namedtuple of the form Sentiment(polarity, subjectivity)
    # polarity is a float within the range [-1.0, 1.0]
    # subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective
    sentiment_score = analysis.sentiment.polarity

    return sentiment_score
