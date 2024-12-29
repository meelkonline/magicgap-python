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
    Returns a list of dictionaries with the *chosen* label and score for each text.

    If the top-scoring label is 'neutral', it tries the next one, and so on.
    If all labels end up being 'neutral', it stays with 'neutral'.
    """
    try:
        # Tokenize the input texts
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        results = []
        for i in range(probabilities.size(0)):
            # Convert tensor to list for this text
            scores_i = probabilities[i].cpu().tolist()

            # Build and sort label-score pairs in descending order by score
            label_score_pairs = [
                (model.config.id2label[j], scores_i[j]) for j in range(len(scores_i))
            ]
            label_score_pairs.sort(key=lambda x: x[1], reverse=True)

            # Find the first label that isn't 'neutral'
            chosen_label = None
            chosen_score = None
            for label, score in label_score_pairs:
                print(label.lower())
                if label.lower() != "neutral":
                    chosen_label = label
                    chosen_score = score

                    break

            # If none was found, use the top one (which will be 'neutral')
            if chosen_label is None:
                chosen_label, chosen_score = label_score_pairs[0]

            results.append({"label": chosen_label, "score": chosen_score})

        return results

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


# r = SentimentRequest(lang="fr", strings=["Bonjour Inspecteur, je vais bien je vous remercie"])
# s = evaluate_sentiment(r)
# print(s)