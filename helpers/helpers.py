from PyPDF2 import PdfReader
import re
import spacy
import pdfplumber
import textwrap
import openai


def read_file(file_path):
    file_extension = file_path.split('.')[-1].lower()
    if file_extension == 'txt':
        file_content = read_text_file(file_path)
    elif file_extension == 'pdf':
        file_content = read_pdf_file(file_path)
    else:
        print("Unsupported file format")
        return None

    if file_content == "":
        print("File empty or not found")
        return None

    return file_content


def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def read_pdf_file(file_path):
    # Open the file
    pdf = PdfReader(file_path)
    # Read the content of each page
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text


def clean_text(text):
    # Replace multiple spaces with a single space
    curated_text = re.sub(r'\s+', ' ', text)
    curated_text = re.sub(r'[\t]', ' ', curated_text)  # Replace tabs with spaces
    curated_text = re.sub(r'[^\w\s.!?]', '', curated_text)  # Remove special characters, but keep sentence delimiters
    return curated_text


def extract_coherent_chunks(text, max_chunk_size=15000):
    """
    Extracts coherent chunks from a document, concatenating paragraphs without exceeding the maximum chunk size.
    This version replaces tabs with spaces for consistency and trims whitespace.

    :param text: The text content of the document.
    :param max_chunk_size: Maximum size of each chunk in characters.
    :return: A list of text chunks.
    """
    paragraphs = clean_text(text).split('\n')  # Replace tabs with spaces and split text into paragraphs
    chunks = []
    current_chunk = ''
    max_chunk_size = int(max_chunk_size)
    for paragraph in paragraphs:
        paragraph = paragraph.strip()  # Trim whitespace from the paragraph
        if not paragraph:
            continue

        # Check if the paragraph fits into the current chunk
        if len(current_chunk) + len(paragraph) + 1 <= max_chunk_size:
            current_chunk += paragraph + '\n'
        else:
            # Save the current chunk if it's not empty and start a new one
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + '\n'

    # Add any remaining text as a chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def chunk_pdf(document_path, max_chunk_size):
    # Load a spaCy model
    nlp = spacy.load("en_core_web_sm")  # or another model of your choice

    # Read and extract text from the PDF
    with pdfplumber.open(document_path) as pdf:
        full_text = ''
        for page in pdf.pages:
            full_text += page.extract_text() + '\n'

    # Curate the content: remove tabs and special characters
    curated_text = clean_text(full_text)
    # Process the text with spaCy
    doc = nlp(curated_text)

    # Chunk the text
    chunks = chunk_text_spacy(doc, max_chunk_size)

    return chunks


def chunk_text_spacy(doc, max_chunk_size=5000):
    chunks = []
    current_chunk = ''
    current_length = 0
    max_chunk_size = int(max_chunk_size)

    for sent in doc.sents:
        sentence = sent.text.strip()
        sentence_length = len(sentence)

        if current_length + sentence_length <= max_chunk_size:
            current_chunk += sentence + ' '
            current_length += sentence_length
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ' '
            current_length = sentence_length

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def process_document(file_path, max_length=1024):
    with open(file_path, 'r', encoding='utf-8') as file:
        document_content = file.read()

    chunks = textwrap.wrap(document_content, max_length, break_long_words=False, replace_whitespace=False)
    processed_chunks = [clean_and_process_text(chunk) for chunk in chunks]

    return processed_chunks


def clean_and_process_text(text):
    processed_text = text.strip()
    return processed_text


def generate_chunks(text, description, openai_api_key):
    openai.api_key = openai_api_key
    prompt = f"{description}<<<{text}>>>\n\n" \
             f"Please generate relevant chunks for further training and RAG operations. Return a JSON array."

    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=1024,
        response_format={"type": "json_object"}
    )

    chunks = response.choices[0].text.strip().split('\n\n')
    return chunks


def generate_openai_qas(content, description, openai_api_key):
    """
    Generates Q/A pairs from a given text using OpenAI API.

    :param description:
    :param content: A text content.
    :param openai_api_key: OpenAI API key.
    :return: Generated Q/A pairs.
    """
    openai.api_key = openai_api_key

    prompt = f"{description}: {content}.Return a JSON array, with each Q/A saved as SQUAD"

    response = openai.Completion.create(
        engine="gpt-4-1106-preview",
        prompt=prompt,
        max_tokens=1024,  # Adjust based on your needs
        response_format={"type": "json_object"},
        temperature=0.0
    )

    return response.choices[0].text.strip()
