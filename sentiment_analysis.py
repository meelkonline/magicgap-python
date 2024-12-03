import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from api_requests import SentimentRequest

# Determine the device to use (GPU if available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# English emotion model
ENGLISH_MODEL_NAME = "michellejieli/emotion_text_classifier"
english_tokenizer = AutoTokenizer.from_pretrained(ENGLISH_MODEL_NAME)
english_model = AutoModelForSequenceClassification.from_pretrained(ENGLISH_MODEL_NAME).to(DEVICE)

# French emotion model
FRENCH_MODEL_NAME = "ac0hik/emotions_detection_french"
french_tokenizer = AutoTokenizer.from_pretrained(FRENCH_MODEL_NAME)
french_model = AutoModelForSequenceClassification.from_pretrained(FRENCH_MODEL_NAME).to(DEVICE)


def classify_emotion(texts, tokenizer, model):
    """
    Classifies emotions for a list of texts using the specified tokenizer and model.
    Returns a list of dictionaries with label and score.
    """
    try:
        # Tokenize the input texts
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Extract the top label and score for each input
        top_indices = torch.argmax(probabilities, dim=-1).cpu().tolist()
        labels = [model.config.id2label[idx] for idx in top_indices]
        scores = probabilities[range(len(probabilities)), top_indices].cpu().tolist()

        # Convert to JSON-serializable structure
        return [{"label": label, "score": score} for label, score in zip(labels, scores)]

    except Exception as e:
        # Handle and log any errors
        return {"error": str(e), "details": "An error occurred during classification."}


def evaluate_sentiment(request: SentimentRequest):
    """
    Evaluates sentiment based on the language in the request.
    """
    try:
        if request.lang == "fr":
            return classify_emotion(request.strings, french_tokenizer, french_model)
        else:
            return classify_emotion(request.strings, english_tokenizer, english_model)
    except Exception as e:
        return {"error": str(e), "details": "An error occurred during sentiment analysis."}


# r = SentimentRequest(lang="en", strings=["hello inspector, i am fine thank you"])
# s = evaluate_sentiment(r)
# print(s)