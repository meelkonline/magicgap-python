import spacy
import langid
import pdfplumber
from docx import Document
from transformers import pipeline
from api_requests import MultipleStringRequest, SentimentRequest

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")
english_emotion_classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier", top_k=2)
french_emotion_classifier = pipeline("text-classification", model="ac0hik/emotions_detection_french", top_k=2)


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


def evaluate_sentiment(request: SentimentRequest):
    if request.lang == 'fr':
        result = french_emotion_classifier(request.strings)
    else:
        result = english_emotion_classifier(request.strings)

    return result


def get_lang(text):
    lang, confidence = langid.classify(text)
    return lang


# BELOW : Chunking for RAG functions


def load_text(filepath):
    """ Load text from a PDF or DOCX file. """
    if filepath.endswith('.pdf'):
        text = extract_text_from_pdf(filepath)
    elif filepath.endswith('.docx'):
        text = extract_text_from_docx(filepath)
    elif filepath.endswith('.txt'):
        text = extract_text_from_txt(filepath)
    else:
        raise ValueError("Unsupported file type. Please use a .pdf or .docx file.")
    return text


def extract_text_from_txt(filepath):
    """ Extract text from a TXT file. """
    file = open(filepath, "r")
    text = file.read()
    file.close()

    return text


def extract_text_from_pdf(filepath):
    """ Extract text from a PDF file. """
    text = ''
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + ' '
    return text


def extract_text_from_docx(filepath):
    """ Extract text from a DOCX file. """
    doc = Document(filepath)
    return ' '.join([paragraph.text for paragraph in doc.paragraphs])


def extract_sentences(text):
    """ Use spaCy to extract sentences from the provided text. """
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]
