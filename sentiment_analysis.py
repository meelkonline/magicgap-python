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

# Emotion category mappings (French & English)
EMOTION_MAPPING = {
    "disgust": ("DisgustTrust", -1),
    "degout": ("DisgustTrust", -1),
    "fear": ("FearAnger", -1),
    "peur": ("FearAnger", -1),
    "anger": ("FearAnger", 1),
    "colere": ("FearAnger", 1),
    "sadness": ("SadnessJoy", -1),
    "tristesse": ("SadnessJoy", -1),
    "joy": ("SadnessJoy", 1),
    "joie": ("SadnessJoy", 1),
    "surprise": ("SurpriseAnticipation", -1),
    "surpris": ("SurpriseAnticipation", -1),
}


def classify_emotion(texts, tokenizer, model):
    """
    Classifies emotions for a list of texts using the specified tokenizer and model.
    Returns a list of dictionaries formatted according to the PHP function logic.
    """
    try:
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        results = []
        for i in range(probabilities.size(0)):
            scores_i = probabilities[i].cpu().tolist()

            # Build and sort label-score pairs in descending order
            raw_label_score_pairs = [
                (model.config.id2label[j].lower(), scores_i[j]) for j in range(len(scores_i))
            ]
            raw_label_score_pairs.sort(key=lambda x: x[1], reverse=True)

            # 🔍 Ensure at least one valid label exists
            if not raw_label_score_pairs or not isinstance(raw_label_score_pairs[0], tuple):
                continue  # Skip this iteration safely

            # Select the first non-neutral label and map it
            for label, score in raw_label_score_pairs:
                if label in EMOTION_MAPPING:
                    mapped_label, multiplier = EMOTION_MAPPING[label]
                    if mapped_label:
                        chosen_label = mapped_label
                        chosen_score = round(score * multiplier, 6)
                        results.append({"label": chosen_label, "score": chosen_score})
                    break  # Stop after finding a valid label

        return results

    except Exception as e:
        print(f"ERROR in classify_emotion: {e}")
        return []  # 🔥 Important: Return an empty list instead of None to prevent unpacking errors


def evaluate_sentiment(request: SentimentRequest):
    """
    Evaluates sentiment based on the language in the request.
    Returns a list of sentiment results.
    """
    try:
        if request.lang == "fr":
            return classify_emotion(request.strings, french_tokenizer, french_model)
        else:
            return classify_emotion(request.strings, english_tokenizer, english_model)
    except Exception as e:
        print(f"ERROR in evaluate_sentiment: {e}")
        return [{"error": str(e), "details": "An error occurred during sentiment analysis."}]


# Example Usage
r = SentimentRequest(lang="fr", strings=["Bonjour Inspecteur, je vais bien je vous remercie"])
s = evaluate_sentiment(r)
print(s)

r_en = SentimentRequest(lang="en", strings=["Hello Inspector, I am fine thank you"])
s_en = evaluate_sentiment(r_en)
print(s_en)

r = SentimentRequest(lang="en", strings=['im looking for a rope', ' i dropped my ring at the bottom of the well',
                                         ' juste offered to retrieve it for me', ' the rope is for him'])
s = evaluate_sentiment(r)
print(s)
