import re
from typing import List

import spacy
import pdfplumber
from docx import Document

# Cache pour éviter de recharger les modèles
_SPACY_CACHE = {}


def load_spacy_for_lang(lang: str):
    """
    Charge un modèle spaCy en fonction de la langue détectée.
    Utilise un cache pour éviter le rechargement multiple.
    """
    global _SPACY_CACHE

    lang = lang.lower()

    # Mapping langues -> modèles spaCy
    MODEL_MAP = {
        "fr": "fr_core_news_md",
        "en": "en_core_web_sm",
        "es": "es_core_news_sm",
        "de": "de_core_news_sm"
    }

    model = MODEL_MAP.get(lang, "en_core_web_sm")  # fallback anglais

    if model not in _SPACY_CACHE:
        try:
            _SPACY_CACHE[model] = spacy.load(model)
        except:
            # fallback ultime
            _SPACY_CACHE[model] = spacy.load("en_core_web_sm")

    return _SPACY_CACHE[model]


def spatie_extract_phrases(text, lang: str):
    """
    NEW VERSION:
    - text: list[str] (conversation messages)
      (user, system, user, system, ...)
    - lang: language code ("fr", "en", ...)

    Returns: list[str]
    One reduced / extracted string per input message.
    """

    # Backward compatibility: if a single string is passed, wrap in list
    if isinstance(text, str):
        messages = [text]
    else:
        messages = text

    extractor = load_spacy_for_lang(lang)

    # Entity labels to keep (geo, proper names, orgs, laws, etc.)
    IMPORTANT_ENTS = {
        "PERSON", "ORG", "GPE", "LOC", "FAC", "NORP",
        "PRODUCT", "EVENT", "WORK_OF_ART", "LAW",
        "MISC",  # often used for named things in some models
        # You can re-add DATE/TIME here if you want all dates too
        "DATE"
    }

    # Words to ignore (greetings, etc.)
    GREETINGS = {
        "bonjour", "salut", "coucou", "hello", "bonsoir",
        "merci"
    }

    # Question words to ignore, to avoid "de quand la loi..."
    QUESTION_WORDS = {
        "comment", "quand", "pourquoi", "où", "ou",
        "quel", "quelle", "quels", "quelles", "quoi",
        "lequel", "laquelle", "lesquels", "lesquelles"
    }

    reduced_list = []

    for msg in messages:
        doc = extractor(msg)

        # Mark tokens we want to keep
        keep = [False] * len(doc)

        # 1) Mark entity tokens
        for ent in doc.ents:
            if ent.label_ in IMPORTANT_ENTS:
                for tok in ent:
                    keep[tok.i] = True

        # 2) Mark key POS tokens (nouns, verbs, proper nouns)
        for token in doc:
            if token.is_space:
                continue

            tlower = token.text.lower()

            # skip greetings
            if tlower in GREETINGS:
                continue

            # skip question words
            if tlower in QUESTION_WORDS:
                continue

            # skip stop words & auxiliaries
            if token.is_stop:
                continue
            if token.pos_ == "AUX" or token.dep_ == "aux":
                continue

            # keep meaningful tokens
            if token.pos_ in ["NOUN", "PROPN", "VERB"]:
                keep[token.i] = True

        # 3) Build the reduced string in ORIGINAL ORDER
        selected_tokens = []
        for token in doc:
            if not keep[token.i]:
                continue

            # lemmatize verbs, keep surface form for nouns/proper nouns
            if token.pos_ == "VERB":
                tok_text = token.lemma_
            else:
                tok_text = token.text

            selected_tokens.append(tok_text)

        # 4) Remove duplicates while preserving order (per message)
        seen = set()
        final_tokens = []
        for tok in selected_tokens:
            if tok not in seen:
                final_tokens.append(tok)
                seen.add(tok)

        phrase = " ".join(final_tokens).strip()
        reduced_list.append(phrase)

    unique_reduced = []
    seen = set()

    for item in reduced_list:
        if item not in seen:
            unique_reduced.append(item)
            seen.add(item)

    return unique_reduced


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


# Function to limit text to a specified number of sentences
def limit_to_sentences(text, max_sentences=2):
    # Load the spaCy model
    nlp = spacy.load("fr_core_news_sm")
    doc = nlp(text)
    sentences = list(doc.sents)  # Extract sentences using spaCy
    limited_sentences = " ".join([str(sent) for sent in sentences[:max_sentences]])  # Join the first n sentences
    return limited_sentences
