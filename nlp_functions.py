import os

import spacy
import langid
import pdfplumber
from docx import Document
from fastapi import HTTPException

from transformers import pipeline

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


def evaluate_toxicity(text):
    # nb : bof en francais : je vais te violer = not toxic 0.7
    model_path = "citizenlab/distilbert-base-multilingual-cased-toxicity"
    toxicity_classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)
    result = toxicity_classifier(text)
    score = result[0]['score']
    if result[0]['label'] == "not_toxic":
        score = -score
    return float(score)


def extract_document_chunks(filepath, max_words):
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found.")

    # Detecting file format based on file extension
    file_extension = filepath.split('.')[-1].lower()
    try:
        if file_extension == 'pdf':
            chunks = extract_text_from_pdf(filepath, max_words)
        elif file_extension == 'docx':
            chunks = extract_text_from_docx(filepath, max_words)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

    return {"chunks": chunks}


def extract_text_from_pdf(filepath, max_words):
    with pdfplumber.open(filepath) as pdf:
        text_chunks = []
        current_chunk = []
        current_word_count = 0
        for page in pdf.pages:
            text = page.extract_text() or ""
            for word in text.split():
                if current_word_count < max_words:
                    current_chunk.append(word)
                    current_word_count += 1
                else:
                    text_chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_word_count = 1
        if current_chunk:
            text_chunks.append(" ".join(current_chunk))
    return text_chunks


def extract_text_from_docx(filepath, max_words):
    doc = Document(filepath)
    text_chunks = []
    current_chunk = []
    current_word_count = 0
    for paragraph in doc.paragraphs:
        for word in paragraph.text.split():
            if current_word_count < max_words:
                current_chunk.append(word)
                current_word_count += 1
            else:
                text_chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_word_count = 1
    if current_chunk:
        text_chunks.append(" ".join(current_chunk))
    return text_chunks


def get_lang(text):
    lang, confidence = langid.classify(text)
    return lang
