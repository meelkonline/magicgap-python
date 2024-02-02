import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Example sentence
text = "Yes, I saw a white rabbit last night. It was hopping everywhere. And then it died."

# Process the text with spaCy
doc = nlp(text)


# Function to lemmatize and remove stop words from a chunk
def spatie_refine_chunk(chunk):
    # Lemmatize and remove stop words within the chunk
    refined = [token.lemma_ for token in chunk if not token.is_stop]
    # Join the refined tokens back into a string
    return " ".join(refined)


# Extract, refine, and filter noun chunks as key phrases
key_phrases = [spatie_refine_chunk(chunk) for chunk in doc.noun_chunks if not all(token.is_stop for token in chunk)]

# Filter out any empty strings that might result from removing stop words
key_phrases = [phrase for phrase in key_phrases if phrase]

print("Refined Key Phrases:", key_phrases)
