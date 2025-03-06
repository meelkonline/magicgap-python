import re
from typing import List

import spacy
import pdfplumber
from docx import Document

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")


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


# def get_toxicity(text):
#     raw_results = Detoxify('multilingual').predict(text)
#     processed_results = {key: float(value) for key, value in raw_results.items()}
#     return processed_results


# def get_lang(text):
#     lang, confidence = langid.classify(text)
#     return lang


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
    file = open(filepath, "r", encoding="utf-8")
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


def extract_sentences(text, chunk_size=100000):
    """ Extract sentences from large text in smaller chunks. """
    if len(text) <= nlp.max_length:
        return process_sentences_with_spacy(text)

    sentences = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        sentences.extend(process_sentences_with_spacy(chunk))  # Process each chunk separately

    return sentences


def process_sentences_with_spacy(text):
    """ Process a single chunk of text to extract sentences. """
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


# Function to limit text to a specified number of sentences
def limit_to_sentences(text, max_sentences=2):
    doc = nlp(text)
    sentences = list(doc.sents)  # Extract sentences using spaCy
    limited_sentences = " ".join([str(sent) for sent in sentences[:max_sentences]])  # Join the first n sentences
    return limited_sentences
