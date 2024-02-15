import spacy
import langid

from transformers import pipeline

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")


# Function to lemmatize and remove stop words from a chunk
def spatie_refine_chunk(chunk):
    # Lemmatize and remove stop words within the chunk
    refined = [token.lemma_ for token in chunk if not token.is_stop]
    # Join the refined tokens back into a string
    return " ".join(refined)


def spatie_extract_phrases(text):
    doc = nlp(text)

    # Initialize an empty list to store nouns and verbs in the order they appear
    nouns_verbs = []

    # Iterate over each token in the document
    for token in doc:
        # Check if the token is a noun or a verb, not a stop word, and not an auxiliary verb
        if token.pos_ in ["NOUN", "PROPN", "VERB"] and not token.is_stop and not token.dep_ == "aux":
            # For nouns, you might want to use the original text or the lemma_ depending on your preference
            # For verbs, using lemma_ to get the base form of the verb
            word = token.text if token.pos_ in ["NOUN", "PROPN"] else token.lemma_

            # Add the noun or verb to the list
            nouns_verbs.append(word)

    return ' '.join(nouns_verbs)


def evaluate_toxicity(text):
    # nb : bof en francais : je vais te violer = not toxic 0.7
    model_path = "citizenlab/distilbert-base-multilingual-cased-toxicity"
    toxicity_classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)
    result = toxicity_classifier(text)
    score = result[0]['score']
    if result[0]['label'] == "toxic":
        score = -score
    return float(score)


def get_lang(text):
    lang, confidence = langid.classify(text)
    return lang
